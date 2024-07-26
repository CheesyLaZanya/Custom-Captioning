# Standard library imports
from typing import List

# Third-party library imports
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    ).to(device).eval()
    processor = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    return model, processor


def generate_caption(model: AutoModelForCausalLM, processor: AutoTokenizer, image: Image.Image, prompt: str) -> str:
    device = next(model.parameters()).device
    messages = [
        {"role": "user", "content": f'<image>\n{prompt}'}
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    text_chunks = [processor(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0).to(device)
    image_tensor = model.process_images([image], model.config).to(device=device, dtype=model.dtype)
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=2048,
        use_cache=True
    )[0]
    return processor.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()


def get_suggested_models() -> List[str]:
    suggested_models = [
        'cognitivecomputations/dolphin-vision-7b'
    ]
    return suggested_models


def get_suggested_prompts() -> List[str]:
    suggested_prompts = [
        'Describe this image in detail.'
    ]
    return suggested_prompts
