# Standard library imports
from typing import List

# Third-party library imports
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def load_model(model_name: str) -> tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    # Note: We're not moving the model to a specific device here,
    # as it's not needed for 4-bit mode.
    return model, processor


def generate_annotation(model: LlavaNextForConditionalGeneration, processor: LlavaNextProcessor, image: Image.Image, prompt: str) -> str:
    device = next(model.parameters()).device

    # Prepare the prompt using the appropriate template
    full_prompt = f"[INST] <image>\n{prompt} [/INST]"

    # Process the input
    inputs = processor(full_prompt, image, return_tensors="pt").to(device)

    # Generate the annotation
    output = model.generate(**inputs, max_new_tokens=300)

    # Decode the annotation
    annotation = processor.decode(output[0], skip_special_tokens=True)

    # Remove the initial prompt from the annotation
    prompt_marker_end = annotation.find("[/INST]") + len("[/INST]")
    annotation = annotation[prompt_marker_end:].strip()

    return annotation.strip()


def get_suggested_models() -> List[str]:
    suggested_models = [
        'llava-hf/llava-v1.6-mistral-7b-hf'
    ]
    return suggested_models


def get_suggested_prompts() -> List[str]:
    suggested_prompts = [
        'Describe this image in detail.'
    ]
    return suggested_prompts
