# Standard library imports
from typing import List

# Third-party library imports
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

# https://huggingface.co/THUDM/cogvlm-chat-hf


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, LlamaTokenizer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    return model, processor


def generate_annotation(model: AutoModelForCausalLM, processor: LlamaTokenizer, image: Image.Image, prompt: str) -> str:
    device = next(model.parameters()).device

    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = model.build_conversation_input_ids(processor, query=prompt, history=[], images=[image])  # chat mode
    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
        'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
    }
    gen_kwargs = {"max_length": 2048, "do_sample": False}

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        annotation = processor.decode(outputs[0])

    return annotation


def get_suggested_models() -> List[str]:
    suggested_models = [
        'THUDM/cogvlm-chat-hf'
    ]
    return suggested_models


def get_suggested_prompts() -> List[str]:
    suggested_prompts = [
        'Describe this image.'
    ]
    return suggested_prompts
