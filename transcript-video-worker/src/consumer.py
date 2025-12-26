import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, NamedTuple

import backoff
import pika
from pika.exceptions import AMQPConnectionError, AMQPChannelError

from .dto import TranscriptionMessage


class AckRequest(NamedTuple):
    """Request để ACK/NACK message từ thread chính"""
    delivery_tag: int
    success: bool
    job_id: str


class TranscriptionConsumer:
    def __init__(self, cfg: dict, service: Any) -> None:
        self._cfg = cfg
        self._service = service
        self._connection: pika.BlockingConnection | None = None
        self._channel: pika.adapters.blocking_connection.BlockingChannel | None = None
        self._should_stop = False
        self._logger = logging.getLogger("transcription.consumer")
        
        # Thread pool để xử lý messages song song
        num_workers = max(1, int(cfg.get("server", {}).get("workers", 2)))
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="transcription-worker")
        self._logger.info(f"📦 ThreadPoolExecutor initialized with {num_workers} workers")
        
        # Queue để gửi ACK/NACK requests về thread chính (thread-safe)
        self._ack_queue: queue.Queue[AckRequest] = queue.Queue()

    def _connect_with_retry(self) -> None:
        """Kết nối RabbitMQ với retry logic (giống transcode worker)"""
        conn_addr = f"amqp://{self._cfg['rabbitmq']['user']}:{self._cfg['rabbitmq']['pass']}@{self._cfg['rabbitmq']['host']}:{self._cfg['rabbitmq']['port']}/"
        
        @backoff.on_exception(
            backoff.expo,
            (AMQPConnectionError, ConnectionError, OSError),
            max_tries=5,
            max_time=60,
            jitter=backoff.full_jitter,
        )
        def _connect():
            host = self._cfg['rabbitmq']['host']
            port = self._cfg['rabbitmq']['port']
            self._logger.info(f"🔌 Connecting to RabbitMQ at {host}:{port}...")
            params = pika.URLParameters(conn_addr)
            conn = pika.BlockingConnection(params)
            self._logger.info(f"✅ Connected to RabbitMQ at {host}:{port}")
            return conn
        
        self._connection = _connect()
        self._channel = self._connection.channel()

    def _setup_queues(self) -> None:
        """Setup exchanges và queues"""
        ex = self._cfg["rabbitmq"]["exchange"]
        dlx = self._cfg["rabbitmq"]["dlx"]
        queue = self._cfg["rabbitmq"]["queue"]
        dlq = self._cfg["rabbitmq"]["dlq"]
        routing_key = self._cfg["rabbitmq"]["routing_key"]

        self._logger.info(f"📋 Setting up queues: exchange={ex}, queue={queue}, routing_key={routing_key}")
        
        self._channel.exchange_declare(exchange=ex, exchange_type="topic", durable=True)
        self._logger.debug(f"   ✓ Exchange declared: {ex}")
        
        self._channel.exchange_declare(exchange=dlx, exchange_type="topic", durable=True)
        self._logger.debug(f"   ✓ DLX exchange declared: {dlx}")

        self._channel.queue_declare(queue=dlq, durable=True)
        self._channel.queue_bind(queue=dlq, exchange=dlx, routing_key=f"dlq.{routing_key}")
        self._logger.debug(f"   ✓ DLQ declared and bound: {dlq}")

        # Declare queue với arguments (idempotent - sẽ không lỗi nếu queue đã tồn tại với cùng config)
        args = {
            "x-dead-letter-exchange": dlx,
            "x-dead-letter-routing-key": f"dlq.{routing_key}",
        }
        try:
            # Declare queue - sẽ tạo mới nếu chưa tồn tại, hoặc verify config nếu đã tồn tại
            self._channel.queue_declare(queue=queue, durable=True, arguments=args)
            self._logger.debug(f"   ✓ Queue declared: {queue}")
        except pika.exceptions.ChannelClosedByBroker as e:
            # Queue có thể đã tồn tại với arguments khác - cần xóa và tạo lại
            self._logger.warning(
                f"   ⚠️  Queue {queue} may exist with different arguments. "
                f"Channel closed: {e}. Will try to recreate channel and declare queue."
            )
            # Tạo channel mới và thử lại
            self._channel = self._connection.channel()
            try:
                self._channel.queue_declare(queue=queue, durable=True, arguments=args)
                self._logger.debug(f"   ✓ Queue declared after channel recreation: {queue}")
            except Exception as e2:
                self._logger.error(
                    f"   ❌ Failed to declare queue after channel recreation: {e2}. "
                    f"Queue may need to be deleted manually if it has conflicting arguments."
                )
                raise
        except Exception as e:
            self._logger.error(
                f"   ❌ Failed to declare queue: {e}. "
                f"Queue may have been created by another process with different arguments."
            )
            raise
        
        # Bind queue với exchange và routing key mới
        try:
            self._channel.queue_bind(queue=queue, exchange=ex, routing_key=routing_key)
            self._logger.debug(f"   ✓ Queue bound to exchange: {queue} -> {ex} ({routing_key})")
        except Exception as e:
            # Binding có thể đã tồn tại, chỉ log warning
            self._logger.warning(f"   ⚠️  Queue binding may already exist: {e}")
        
        prefetch = max(1, int(self._cfg["server"]["workers"]))
        self._channel.basic_qos(prefetch_count=prefetch)
        self._logger.info(f"✅ Queue setup complete: {queue} (prefetch={prefetch})")

    def stop(self) -> None:
        """Graceful shutdown"""
        self._should_stop = True
        
        # Shutdown thread pool gracefully
        self._logger.info("🛑 Shutting down thread pool...")
        self._executor.shutdown(wait=True, timeout=60)
        self._logger.info("✅ Thread pool shut down")
        
        try:
            if self._channel and self._channel.is_open:
                self._channel.stop_consuming()
                self._channel.close()
        except Exception as e:
            self._logger.warning(f"⚠️  Error closing channel: {e}")
        finally:
            try:
                if self._connection and self._connection.is_open:
                    self._connection.close()
                    self._logger.info("🔌 RabbitMQ connection closed")
            except Exception as e:
                self._logger.warning(f"⚠️  Error closing connection: {e}")

    def _is_connected(self) -> bool:
        """Kiểm tra connection và channel còn sống không"""
        return (
            self._connection is not None
            and self._connection.is_open
            and self._channel is not None
            and self._channel.is_open
        )
    
    def _process_message(self, body: bytes) -> tuple[bool, str]:
        """Xử lý message trong thread riêng. Trả về (success, job_id)"""
        job_id = "unknown"
        try:
            data = json.loads(body.decode("utf-8"))
            # Convert jobId từ UUID sang string nếu cần
            job_id = data.get("jobId")
            if not isinstance(job_id, str):
                job_id = str(job_id)
            
            msg = TranscriptionMessage(
                jobId=job_id,
                objectPath=data.get("objectPath", ""),
                language=data.get("language")
            )
            
            @backoff.on_exception(
                backoff.expo,
                Exception,
                max_tries=5,
                max_time=30,
            )
            def handle():
                self._service.process(msg)
            
            handle()
            return (True, job_id)
        except Exception as e:
            self._logger.error(
                f"❌ Failed to process message after retries | jobId={job_id[:8]}... | error={e}",
                exc_info=True
            )
            return (False, job_id)
    
    def _process_ack_queue(self, channel) -> None:
        """Xử lý ACK/NACK requests từ queue (chạy trong thread chính)"""
        try:
            while True:
                # Non-blocking get với timeout ngắn để không block quá lâu
                try:
                    ack_req = self._ack_queue.get(timeout=0.1)
                except queue.Empty:
                    break
                
                try:
                    if ack_req.success:
                        channel.basic_ack(delivery_tag=ack_req.delivery_tag)
                        self._logger.info(f"✅ Successfully processed | jobId={ack_req.job_id[:8]}...")
                    else:
                        channel.basic_nack(delivery_tag=ack_req.delivery_tag, requeue=False)
                        self._logger.warning(f"⚠️  Message sent to DLQ | jobId={ack_req.job_id[:8]}...")
                except Exception as e:
                    self._logger.warning(f"⚠️  Failed to ACK/NACK message: {e}")
        except Exception as e:
            self._logger.error(f"❌ Error processing ACK queue: {e}", exc_info=True)

    def start(self) -> None:
        """Start consumer với auto-reconnect"""
        queue = self._cfg["rabbitmq"]["queue"]
        exchange = self._cfg["rabbitmq"]["exchange"]
        routing_key = self._cfg["rabbitmq"]["routing_key"]
        reconnect_delay = 5  # seconds

        while not self._should_stop:
            try:
                # Kết nối với retry
                if not self._is_connected():
                    self._connect_with_retry()
                    self._setup_queues()

                def on_message(_ch, method, properties, body: bytes):
                    """Callback được gọi khi nhận message từ RabbitMQ (chạy trong thread chính)"""
                    try:
                        data = json.loads(body.decode("utf-8"))
                        job_id = data.get("jobId", "unknown")
                        object_path = data.get("objectPath", "unknown")
                        
                        self._logger.info(
                            f"📨 Received message | jobId={job_id[:8] if isinstance(job_id, str) else str(job_id)[:8]}... | "
                            f"objectPath={object_path}"
                        )
                    except Exception as e:
                        self._logger.error(f"❌ Failed to parse message: {e}")
                        try:
                            _ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
                        except Exception:
                            pass
                        return
                    
                    # Submit task vào thread pool để xử lý song song
                    future = self._executor.submit(self._process_message, body)
                    
                    # Callback để gửi ACK/NACK request về queue (chạy trong thread pool)
                    def on_done(fut: Future):
                        try:
                            success, job_id = fut.result()
                            # Gửi request về queue để thread chính xử lý ACK/NACK
                            self._ack_queue.put(AckRequest(
                                delivery_tag=method.delivery_tag,
                                success=success,
                                job_id=job_id
                            ))
                        except Exception as e:
                            self._logger.error(f"❌ Error in future callback: {e}", exc_info=True)
                            # NACK nếu có lỗi không mong đợi
                            self._ack_queue.put(AckRequest(
                                delivery_tag=method.delivery_tag,
                                success=False,
                                job_id="unknown"
                            ))
                    
                    future.add_done_callback(on_done)

                self._logger.info(f"🚀 Started consuming | queue={queue} | exchange={exchange} | routing_key={routing_key}")
                
                # Polling loop: xử lý messages và ACK queue
                while not self._should_stop:
                    # Process ACK queue trước
                    self._process_ack_queue(self._channel)
                    
                    # Get message từ RabbitMQ (non-blocking)
                    try:
                        method_frame, properties, body = self._channel.basic_get(queue=queue, auto_ack=False)
                        if method_frame:
                            # Có message mới, xử lý
                            on_message(self._channel, method_frame, properties, body)
                    except Exception as e:
                        # Nếu có lỗi, break để reconnect
                        if not isinstance(e, (AMQPConnectionError, AMQPChannelError)):
                            self._logger.warning(f"⚠️  Error getting message: {e}")
                        raise
                    
                    # Sleep ngắn để không chiếm CPU
                    time.sleep(0.1)

            except (AMQPConnectionError, AMQPChannelError, ConnectionError, OSError) as e:
                self._logger.error(f"🔌 Connection lost: {e} | Reconnecting in {reconnect_delay}s...")
                self._cleanup_connection()
                if not self._should_stop:
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff, max 60s
            except KeyboardInterrupt:
                self._logger.info("🛑 Received interrupt signal")
                break
            except Exception as e:
                self._logger.exception(f"💥 Unexpected error in consumer loop: {e}")
                self._cleanup_connection()
                if not self._should_stop:
                    time.sleep(reconnect_delay)

    def _cleanup_connection(self) -> None:
        """Cleanup connection và channel"""
        try:
            if self._channel and self._channel.is_open:
                self._channel.close()
        except Exception:
            pass
        try:
            if self._connection and self._connection.is_open:
                self._connection.close()
        except Exception:
            pass
        self._channel = None
        self._connection = None