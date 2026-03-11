from asdq.models.llava_onevision import LLaVA_onevision  # noqa: F401 - registers "llava_onevision"
from asdq.utils.registry import MODEL_REGISTRY


def get_process_model(model_name):
    return MODEL_REGISTRY[model_name]
