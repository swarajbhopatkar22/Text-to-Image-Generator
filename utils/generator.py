# generate.py
import os
from datetime import datetime
from typing import List, Dict, Optional
from utils.prompts import build_full_prompt, build_negative_prompt# Import the missing function
import torch
from diffusers import StableDiffusionPipeline
from utils.prompts import build_full_prompt

# Quick placeholder functions

from PIL import Image  # Make sure pillow is installed

def apply_watermark(image: Image.Image, text: str = "") -> Image.Image:
    """
    Placeholder: currently does nothing, just returns the image.
    """
    return image

def save_with_metadata(image, base_filename, output_dir, **metadata):
    """
    Saves image as PNG and JPEG and creates a metadata JSON file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save PNG
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    image.save(png_path)

    # Save JPEG
    jpg_path = os.path.join(output_dir, f"{base_filename}.jpg")
    image.convert("RGB").save(jpg_path, "JPEG", quality=95)

    # Save metadata JSON (optional, currently just path)
    metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.json")
    return {"image_path": png_path, "jpg_path": jpg_path, "metadata_path": metadata_path}



class TextToImageGenerator:
    """
    Simple wrapper around a Stable Diffusion pipeline.
    Keeps configuration and device handling in one place.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device_preference: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.model_id = model_id
        self.device = self._select_device(device_preference)
        self.torch_dtype = torch_dtype if self.device == "cuda" else torch.float32

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            safety_checker=None,  # using custom filter instead
        )
        self.pipeline.to(self.device)

    @staticmethod
    def _select_device(preference: str) -> str:
        if preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def generate_images(
        self,
        base_prompt: str,
        style: str,
        n_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        negative_prompt_text: Optional[str] = None,
        output_dir: str = "outputs",
    ) -> List[Dict]:
        """
        Generates images and returns a list of dictionaries:
        [{ 'image_path': str, 'metadata_path': str }, ...]
        """
        os.makedirs(output_dir, exist_ok=True)

        full_prompt = build_full_prompt(base_prompt, style)
        negative_prompt = build_negative_prompt(negative_prompt_text)

        # Estimate rough ETA: this is just a heuristic for display
        # num_inference_steps and num_images are the main factors
        approx_seconds_per_image = 1.0 if self.device == "cuda" else 5.0
        estimated_time = approx_seconds_per_image * num_inference_steps * n_images

        results = []

        # Run the model
        with torch.autocast(self.device) if self.device == "cuda" else torch.inference_mode():
            images = self.pipeline(
                [full_prompt] * n_images,
                negative_prompt=[negative_prompt] * n_images if negative_prompt else None,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for idx, img in enumerate(images):
            img = apply_watermark(img, text="AI generated @ YourName")

            base_filename = f"sd_{timestamp}_{idx}"
            img_info = save_with_metadata(
                image=img,
                base_filename=base_filename,
                output_dir=output_dir,
                prompt=base_prompt,
                full_prompt=full_prompt,
                negative_prompt=negative_prompt,
                style=style,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                device=self.device,
                estimated_time=estimated_time,
            )
            results.append(img_info)

        return results, estimated_time
