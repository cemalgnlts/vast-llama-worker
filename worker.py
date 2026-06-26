import os
import random

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    LogActionConfig,
    BenchmarkConfig,
)

# --- Model configuration ------------------------------------------------------

MODEL_SERVER_URL  = "http://127.0.0.1"
MODEL_SERVER_PORT = int(os.getenv("LLAMA_ARG_PORT", 5000))
MODEL_LOG_FILE    = os.getenv("MODEL_LOG", "/var/log/model.log")
MODEL_HEALTHCHECK_ENDPOINT = "/health"
N_SLOTS = int(os.getenv("LLAMA_ARG_N_PARALLEL", 4))

# llama-specific log messages
MODEL_LOAD_LOG_MSG = [
    "llama_server: model loaded"
]

MODEL_ERROR_LOG_MSGS = [
    "Traceback (most recent call last):",
    "Initialization failed. warmup error: Traceback (most recent call last):",
    "llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache",
    "llama_server: exiting due to model loading error",
    "llama_model_load_from_file_impl: failed to load model",
    "CUDA error: CUDA-capable device(s) is/are busy or unavailable",
    "ggml_cuda_init: failed to initialize CUDA: forward compatibility was attempted on non supported HW",
    "common_fit_params: encountered an error while trying to fit params to free device memory: failed to create llama_context from model",
    "failed to download model from Hugging Face"
]

MODEL_INFO_LOG_MSGS = [
    "init:",
    "- CUDA",
    "system_info:",
    "llama_server: loading model",
    "update_slots: all slots are idle",
    "eval time ="
]

# --- Benchmark data generation -----------------------------------------------

WORD_LIST = [
    "A",
    "a",
    "aa",
    "aal",
    "aalii",
    "aam",
    "Aani",
    "aardvark",
    "aardwolf",
    "Aaron",
    "Aaronic",
    "Aaronical",
    "Aaronite",
    "Aaronitic",
    "Aaru",
    "Ab",
    "aba",
    "Ababdeh",
    "Ababua",
    "abac",
    "abaca",
    "abacate",
    "abacay",
    "abacinate",
    "abacination",
    "abaciscus",
    "abacist",
    "aback",
    "abactinal",
    "abactinally",
    "abaction",
    "abactor",
    "abaculus",
    "abacus",
    "Abadite",
    "abaff",
    "abaft",
    "abaisance",
    "abaiser",
    "abaissed",
    "abalienate",
    "abalienation",
    "abalone",
    "Abama",
    "abampere",
    "abandon",
    "abandonable",
    "abandoned",
    "abandonedly",
    "abandonee",
    "abandoner",
    "abandonment",
    "Abanic",
    "Abantes",
    "abaptiston",
    "Abarambo",
    "Abaris",
    "abarthrosis",
    "abarticular",
    "abarticulation",
    "abas",
    "abase",
    "abased",
    "abasedly",
    "abasedness",
    "abasement",
    "abaser",
    "Abasgi",
    "abash",
    "abashed",
    "abashedly",
    "abashedness",
    "abashless",
    "abashlessly",
    "abashment",
    "abasia",
    "abasic",
    "abask",
    "Abassin",
    "abastardize",
    "abatable",
    "abate",
    "abatement",
    "abater",
    "abatis",
    "abatised",
    "abaton",
    "abator",
    "abattoir",
    "Abatua",
    "abature",
    "abave",
    "abaxial",
    "abaxile",
    "abaze",
    "abb",
    "Abba",
    "abbacomes",
    "abbacy",
    "Abbadide",
    "abbas",
    "abbasi",
    "abbassi",
    "Abbasside",
    "abbatial",
    "abbatical",
    "abbess",
    "abbey",
    "abbeystede",
    "Abbie",
    "abbot",
    "abbotcy",
    "abbotnullius",
    "abbotship",
    "abbreviate",
    "abbreviately",
    "abbreviation",
    "abbreviator",
    "abbreviatory",
    "abbreviature",
    "Abby",
    "abcoulomb",
    "abdal",
    "abdat",
    "Abderian",
    "Abderite",
    "abdest",
    "abdicable",
    "abdicant",
    "abdicate",
    "abdication",
    "abdicative",
    "abdicator",
    "Abdiel",
    "abditive",
    "abditory",
    "abdomen",
    "abdominal",
    "Abdominales",
    "abdominalian",
    "abdominally",
    "abdominoanterior",
    "abdominocardiac",
    "abdominocentesis",
    "abdominocystic",
    "abdominogenital",
    "abdominohysterectomy",
    "abdominohysterotomy",
    "abdominoposterior",
    "abdominoscope",
    "abdominoscopy",
    "abdominothoracic",
    "abdominous",
    "abdominovaginal",
    "abdominovesical",
    "abduce",
    "abducens",
    "abducent",
    "abduct",
    "abduction",
    "abductor",
    "Abe",
    "abeam",
    "abear",
    "abearance",
    "abecedarian",
    "abecedarium",
    "abecedary",
    "abed",
    "abeigh",
    "Abel",
    "abele",
    "Abelia",
    "Abelian",
    "Abelicea",
    "Abelite",
    "abelite",
    "Abelmoschus",
    "abelmosk",
    "Abelonian",
    "abeltree",
    "Abencerrages",
    "abenteric",
    "abepithymia",
    "Aberdeen",
    "aberdevine",
    "Aberdonian",
    "Aberia",
    "aberrance",
    "aberrancy",
    "aberrant",
    "aberrate",
    "aberration",
    "aberrational",
    "aberrator",
    "aberrometer",
    "aberroscope",
    "aberuncator",
    "abet",
    "abetment",
    "abettal",
    "abettor",
    "abevacuation",
    "abey",
    "abeyance",
    "abeyancy",
    "abeyant",
    "abfarad",
    "abhenry",
    "abhiseka",
    "abhominable",
    "abhor",
    "abhorrence",
    "abhorrency",
    "abhorrent",
    "abhorrently",
    "abhorrer",
    "abhorrible",
    "abhorring",
    "Abhorson",
    "abidal",
    "abidance",
    "abide",
    "abider",
    "abidi",
    "abiding",
    "abidingly",
    "abidingness",
    "Abie",
    "Abies",
    "abietate",
    "abietene",
    "abietic",
    "abietin",
    "Abietineae",
    "abietineous",
    "abietinic",
    "Abiezer",
    "Abigail",
    "abigail",
    "abigailship",
    "abigeat",
    "abigeus",
    "abilao",
    "ability",
    "abilla",
    "abilo",
    "abintestate",
    "abiogenesis",
    "abiogenesist",
    "abiogenetic",
    "abiogenetical",
    "abiogenetically",
    "abiogenist",
    "abiogenous",
    "abiogeny",
    "abiological",
    "abiologically",
    "abiology",
    "abiosis",
    "abiotic",
    "abiotrophic",
    "abiotrophy",
    "Abipon",
    "abir",
    "abirritant",
    "abirritate",
    "abirritation",
    "abirritative",
    "abiston",
    "Abitibi",
    "abiuret",
    "abject",
    "abjectedness",
    "abjection",
    "abjective",
    "abjectly",
    "abjectness",
    "abjoint",
    "abjudge",
    "abjudicate",
    "abjudication",
    "abjunction",
    "abjunctive",
    "abjuration",
    "abjuratory",
    "abjure",
    "abjurement",
    "abjurer",
    "abkar",
    "abkari",
    "Abkhas",
    "Abkhasian",
    "ablach",
    "ablactate",
    "ablactation",
    "ablare",
    "ablastemic",
    "ablastous",
    "ablate"
]


def completions_benchmark_generator() -> dict:
    """
    Generate one benchmark payload for the /v1/completions endpoint.
    This shape should match what your llama server expects.
    """
    prompt = " ".join(random.choices(WORD_LIST, k=int(200)))
    model = os.environ.get("MODEL_NAME")

    if not model:
        raise ValueError("MODEL_NAME environment variable not set")

    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512
    }


# --- Worker configuration -----------------------------------------------------

def llm_workload(payload: dict) -> float:
    # /v1/completions
    prompt = payload.get("prompt", "")
    prompt_tokens = len(prompt) / 4.0

    # /v1/chat/completions
    messages = payload.get("messages", [])
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            prompt_tokens += len(content) / 4.0
        elif isinstance(content, list):  # multimodal content array
            for part in content:
                if part.get("type") == "text":
                    prompt_tokens += len(part.get("text", "")) / 4.0

    max_tokens = float(payload.get("max_tokens", 512))

    return prompt_tokens + max_tokens


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        # /v1/completions: also used as the benchmark handler
        HandlerConfig(
            route="/v1/completions",
            workload_calculator=llm_workload,
            allow_parallel_requests=True,
            max_queue_time=90.0,
            benchmark_config=BenchmarkConfig(
                generator=completions_benchmark_generator,
                concurrency=N_SLOTS,
                runs=2,
            ),
        ),

        # /v1/chat/completions: similar behavior but no benchmark_config
        HandlerConfig(
            route="/v1/chat/completions",
            workload_calculator=llm_workload,
            allow_parallel_requests=True,
            max_queue_time=90.0
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