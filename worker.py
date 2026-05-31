import os
import random

import nltk

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    LogActionConfig,
    BenchmarkConfig,
)

# --- Model configuration ------------------------------------------------------

MODEL_SERVER_URL  = "http://127.0.0.1"
MODEL_SERVER_PORT = int(os.environ.get("LLAMA_ARG_PORT", 5000))
MODEL_LOG_FILE    = os.environ.get("MODEL_LOG", "/var/log/portal/model.log")
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# llama-specific log messages
MODEL_LOAD_LOG_MSG = [
    "llama_server: model loaded"
]

MODEL_ERROR_LOG_MSGS = [
    "Traceback (most recent call last):",
    "Initialization failed. warmup error: Traceback (most recent call last):",
]

MODEL_INFO_LOG_MSGS = [
    "llama_server: loading model",
    "load_model: speculative decoding will use checkpoints",
    "update_slots: all slots are idle"
]

# --- Benchmark data generation -----------------------------------------------

# For this example we use NLTK's word list to create random prompts
nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

def completions_benchmark_generator() -> dict:
    """Generate one benchmark payload for the /v1/completions endpoint.
    This shape should match what your vLLM server expects.
    """
    prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
    model = os.environ.get("MODEL_NAME")

    if not model:
        raise ValueError("MODEL_NAME environment variable not set")

    return {
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 500,
    }

# --- Worker configuration -----------------------------------------------------

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        # /v1/completions: also used as the benchmark handler
        HandlerConfig(
            route="/v1/completions",
            workload_calculator=lambda payload: float(payload.get("max_tokens", 0)),
            allow_parallel_requests=True,
            max_queue_time=60.0,
            benchmark_config=BenchmarkConfig(
                generator=completions_benchmark_generator,
                concurrency=8,
                runs=2,
            ),
        ),

        # /v1/chat/completions: similar behavior but no benchmark_config
        HandlerConfig(
            route="/v1/chat/completions",
            workload_calculator=lambda payload: float(payload.get("max_tokens", 0)),
            allow_parallel_requests=True,
            max_queue_time=60.0,
        ),
    ],

    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS,
    ),
)

# Run the worker synchronously
Worker(worker_config).run()