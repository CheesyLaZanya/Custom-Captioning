# Standard library imports
from typing import Dict, List

# Third-party library imports
import gradio as gr


def create_debug_interface(results: List[Dict]) -> gr.Blocks:
    """
    Create a Gradio interface for debugging image captioning results.

    This function generates a visual interface using Gradio to display
    the results of image captioning, including the images and their
    corresponding captions.

    Args:
        results (List[Dict]): A list of dictionaries containing the captioning results.
            Each dictionary should have 'image', 'file_name', 'prompt', and 'caption' keys.

    Returns:
        gr.Blocks: A Gradio Blocks interface for displaying the debug output.
    """
    def display_results():
        prompts = list(set(result['prompt'] for result in results))
        images = list(set(result['file_name'] for result in results))

        components = []

        for image in images:
            components.append(gr.Image(value=[result['image'] for result in results if result['file_name'] == image][0], label=image))

            for prompt in prompts:
                captions = [result['caption'] for result in results if result['file_name'] == image and result['prompt'] == prompt]
                caption = captions[0] if captions else "N/A"
                components.append(gr.Textbox(value=caption, label=prompt))

            components.append(gr.Markdown("---"))

        return components

    with gr.Blocks() as iface:
        gr.Markdown("# Image Captioning Debug Output")
        gr.Markdown("Visual representation of the captioning results")

        num_prompts = len(set(r['prompt'] for r in results))
        num_components = len(set(r['file_name'] for r in results)) * (num_prompts + 2)

        output_components = []
        for i in range(num_components):
            if i % (num_prompts + 2) == 0:
                output_components.append(gr.Image(label="Image"))
            elif i % (num_prompts + 2) <= num_prompts:
                output_components.append(gr.Textbox(label="Caption"))
            else:
                output_components.append(gr.Markdown("---"))

        def load_results():
            return display_results()

        iface.load(load_results, inputs=[], outputs=output_components)

    return iface
