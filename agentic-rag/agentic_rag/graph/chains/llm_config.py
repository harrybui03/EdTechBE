"""
LLM configuration with rate limiting support for DeepSeek API.
DeepSeek API is compatible with OpenAI API format.
Handles rate limits and provides better error messages.
"""

import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Only load .env file if not in Docker (override=False prevents overriding existing env vars)
load_dotenv(override=False)

# Rate limiting configuration
# DeepSeek API rate limits vary by tier
# Default: 100 requests per minute
DEEPSEEK_RATE_LIMIT_PER_MINUTE = int(os.getenv("DEEPSEEK_RATE_LIMIT_PER_MINUTE", "100"))
# Add buffer: use 80% of limit to be safe
DEEPSEEK_SAFE_LIMIT = int(DEEPSEEK_RATE_LIMIT_PER_MINUTE * 0.8)
DEEPSEEK_MIN_DELAY_SECONDS = 60.0 / DEEPSEEK_SAFE_LIMIT  # Minimum delay between requests

# Track last request time for rate limiting (shared across all LLM instances)
_last_request_time: Optional[float] = None
_request_count = 0
_window_start_time: Optional[float] = None


def rate_limit_delay():
    """
    Add delay to respect rate limits with sliding window.
    Call this before each LLM invocation.
    """
    global _last_request_time, _request_count, _window_start_time
    
    current_time = time.time()
    
    # Reset window if 1 minute has passed
    if _window_start_time is None or (current_time - _window_start_time) >= 60:
        _window_start_time = current_time
        _request_count = 0
        print(f"[RATE LIMIT] New rate limit window started (limit: {DEEPSEEK_SAFE_LIMIT}/minute)")
    
    # Check if we're approaching the limit
    if _request_count >= DEEPSEEK_SAFE_LIMIT:
        wait_time = 60 - (current_time - _window_start_time)
        if wait_time > 0:
            print(f"[RATE LIMIT] Approaching limit ({_request_count}/{DEEPSEEK_SAFE_LIMIT}). Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            _window_start_time = time.time()
            _request_count = 0
    
    # Add minimum delay between requests
    if _last_request_time is not None:
        elapsed = current_time - _last_request_time
        if elapsed < DEEPSEEK_MIN_DELAY_SECONDS:
            delay = DEEPSEEK_MIN_DELAY_SECONDS - elapsed
            if delay > 0.1:  # Only print if delay is significant
                print(f"[RATE LIMIT] Waiting {delay:.2f}s between requests...")
            time.sleep(delay)
    
    _last_request_time = time.time()
    _request_count += 1


def create_llm(
    model: str = "deepseek-chat",
    temperature: float = 0,
    max_retries: int = 3,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance configured for DeepSeek API.
    DeepSeek API is compatible with OpenAI API format.
    Rate limiting is handled by calling rate_limit_delay() before each invoke.
    
    Args:
        model: Model name (default: deepseek-chat)
        temperature: Temperature for generation (default: 0)
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        ChatOpenAI instance configured for DeepSeek
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable is required. "
            "Please set it in your .env file."
        )
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )



