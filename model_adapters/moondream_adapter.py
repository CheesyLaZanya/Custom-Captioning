# Standard library imports
from typing import List

# Third-party library imports
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# https://moondream.ai/


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    ).to(device).eval()
    return model, processor


def generate_annotation(model: AutoModelForCausalLM, processor: AutoTokenizer, image: Image.Image, prompt: str) -> str:
    device = next(model.parameters()).device

    if image.mode != "RGB":
        image = image.convert("RGB")

    encoded_image = model.encode_image(image).to(device)

    annotation = model.answer_question(encoded_image, prompt, processor)

    return annotation


def get_suggested_models() -> List[str]:
    suggested_models = [
        'vikhyatk/moondream2'
    ]
    return suggested_models


def get_suggested_prompts() -> List[str]:
    suggested_prompts = [
        'Describe the image.'
    ]
    return suggested_prompts
