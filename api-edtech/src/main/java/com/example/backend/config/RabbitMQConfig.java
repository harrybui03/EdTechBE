package com.example.backend.config;

import lombok.Getter;
import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.core.QueueBuilder;
import org.springframework.amqp.core.TopicExchange;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@Getter
public class RabbitMQConfig {

    @Value("${rabbitmq.exchange.name}")
    private String exchangeName;

    @Value("${rabbitmq.queue.name}")
    private String queueName;

    @Value("${rabbitmq.routing.key}")
    private String routingKey;

    @Value("${rabbitmq.recording.merge.queue.name}")
    private String recordingMergeQueueName;

    @Value("${rabbitmq.recording.merge.routing.key}")
    private String recordingMergeRoutingKey;

    @Value("${rabbitmq.dlx.name}")
    private String dlxName;

    @Value("${rabbitmq.dlq.name}")
    private String dlqName;

    @Value("${rabbitmq.dlq.routing.key}")
    private String dlqRoutingKey;

    @Value("${rabbitmq.recording.merge.dlq.routing.key}")
    private String recordingMergeDlqRoutingKey;

    @Value("${rabbitmq.transcription.queue.name}")
    private String transcriptionQueueName;

    @Value("${rabbitmq.transcription.dlx.name}")
    private String transcriptionDlxName;

    @Value("${rabbitmq.transcription.dlq.name}")
    private String transcriptionDlqName;

    @Value("${rabbitmq.transcription.dlq.routing.key}")
    private String transcriptionDlqRoutingKey;

    @Bean
    public Queue queue() {
        return QueueBuilder.durable(queueName)
                .withArgument("x-dead-letter-exchange", dlxName)
                .withArgument("x-dead-letter-routing-key", dlqRoutingKey)
                .build();
    }

    @Bean
    public TopicExchange exchange() {
        return new TopicExchange(exchangeName);
    }

    @Bean
    public Binding binding(Queue queue, TopicExchange exchange) {
        return BindingBuilder.bind(queue).to(exchange).with(routingKey);
    }

    @Bean
    public Queue recordingMergeQueue() {
        return QueueBuilder.durable(recordingMergeQueueName)
                .withArgument("x-dead-letter-exchange", dlxName)
                .withArgument("x-dead-letter-routing-key", recordingMergeDlqRoutingKey)
                .build();
    }

    @Bean
    public Binding recordingMergeBinding(Queue recordingMergeQueue, TopicExchange exchange) {
        return BindingBuilder.bind(recordingMergeQueue).to(exchange).with(recordingMergeRoutingKey);
    }

    @Bean
    public TopicExchange deadLetterExchange() {
        return new TopicExchange(dlxName);
    }

    @Bean
    public Queue deadLetterQueue() {
        return new Queue(dlqName, true);
    }

    @Bean
    public Binding deadLetterBinding(Queue deadLetterQueue, TopicExchange deadLetterExchange) {
        return BindingBuilder.bind(deadLetterQueue).to(deadLetterExchange).with(dlqRoutingKey);
    }

    @Bean
    public RabbitTemplate rabbitTemplate(final ConnectionFactory connectionFactory) {
        final RabbitTemplate rabbitTemplate = new RabbitTemplate(connectionFactory);
        rabbitTemplate.setMessageConverter(new Jackson2JsonMessageConverter());
        return rabbitTemplate;
    }

    @Bean
    public Queue transcriptionQueue() {
        return QueueBuilder.durable(transcriptionQueueName)
                .withArgument("x-dead-letter-exchange", transcriptionDlxName)
                .withArgument("x-dead-letter-routing-key", transcriptionDlqRoutingKey)
                .build();
    }

    @Bean
    public Binding transcriptionBinding(Queue transcriptionQueue, TopicExchange exchange) {
        return BindingBuilder.bind(transcriptionQueue).to(exchange).with(routingKey);
    }

    @Bean
    public TopicExchange transcriptionDeadLetterExchange() {
        return new TopicExchange(transcriptionDlxName);
    }

    @Bean
    public Queue transcriptionDeadLetterQueue() {
        return new Queue(transcriptionDlqName, true);
    }

    @Bean
    public Binding transcriptionDeadLetterBinding(Queue transcriptionDeadLetterQueue, TopicExchange transcriptionDeadLetterExchange) {
        return BindingBuilder.bind(transcriptionDeadLetterQueue)
                .to(transcriptionDeadLetterExchange)
                .with(transcriptionDlqRoutingKey);
    }
}