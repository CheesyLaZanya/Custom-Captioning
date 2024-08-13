# Standard library imports
from typing import Dict, List

# Third-party library imports
import gradio as gr


def create_debug_interface(results: List[Dict]) -> gr.Blocks:
    """
    Create a Gradio interface for debugging image annotation results.

    This function generates a visual interface using Gradio to display
    the results of image annotation, including the images and their
    corresponding annotations.

    Args:
        results (List[Dict]): A list of dictionaries containing the annotation results.
            Each dictionary should have 'image', 'file_name', 'prompt', and 'vlm_annotation' keys.

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
                annotations = [result['vlm_annotation'] for result in results if result['file_name'] == image and result['prompt'] == prompt]
                annotation = annotations[0] if annotations else "N/A"
                components.append(gr.Textbox(value=annotation, label=prompt))

            components.append(gr.Markdown("---"))

        return components

    with gr.Blocks() as iface:
        gr.Markdown("# Image Annotation Debug Output")
        gr.Markdown("Visual representation of the annotation results")

        num_prompts = len(set(r['prompt'] for r in results))
        num_components = len(set(r['file_name'] for r in results)) * (num_prompts + 2)

        output_components = []
        for i in range(num_components):
            if i % (num_prompts + 2) == 0:
                output_components.append(gr.Image(label="Image"))
            elif i % (num_prompts + 2) <= num_prompts:
                output_components.append(gr.Textbox(label="Annotation"))
            else:
                output_components.append(gr.Markdown("---"))

        def load_results():
            return display_results()

        iface.load(load_results, inputs=[], outputs=output_components)

    return iface
