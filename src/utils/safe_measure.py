import time
import random
from functools import wraps
import openai


def exponential_backoff(max_retries=5, base_delay=2, reset_after=60):
    def decorator(func):
        last_attempt = 0
        retries = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_attempt, retries
            while retries < max_retries:
                try:
                    if time.time() - last_attempt > reset_after:
                        retries = 0
                    return func(*args, **kwargs)
                except openai.RateLimitError:
                    wait = (2 ** retries) * base_delay + random.uniform(0, 1)
                    time.sleep(wait)
                    retries += 1
                    last_attempt = time.time()
            raise Exception("Max retries exceeded")
        return wrapper
    return decorator


def safe_measure(metric, *args, **kwargs):
    @exponential_backoff()
    def wrapped_measure():
        return metric.measure(*args, **kwargs)

    return wrapped_measure()
