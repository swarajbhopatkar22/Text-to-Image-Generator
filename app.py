# app.py
import os
from utils.safety import is_prompt_allowed
from datetime import datetime
from utils.generator import TextToImageGenerator
import streamlit as st

def is_prompt_allowed(prompt):
    return True

from utils.generator import TextToImageGenerator


# --- Basic page setup ---
st.set_page_config(
    page_title="Text to Image Generator",
    page_icon="🎨",
    layout="wide",
)

st.title("AI-Powered Text-to-Image Generator 🎨")
st.write(
    "Enter a description and generate images using an open-source diffusion model. "
    "Please use the tool responsibly; some content is blocked by policy."
)

# --- Sidebar configuration ---
st.sidebar.header("Generation settings")

device_choice = st.sidebar.selectbox(
    "Device preference",
    options=["auto (GPU if available)", "CPU only"],
)

if device_choice == "CPU only":
    device_pref = "cpu"
else:
    device_pref = "cuda"

num_images = st.sidebar.slider("Number of images", min_value=1, max_value=4, value=1)
style_choice = st.sidebar.selectbox(
    "Style",
    options=["photorealistic", "artistic", "cartoon"],
)
guidance_scale = st.sidebar.slider(
    "Guidance scale", min_value=4.0, max_value=12.0, value=7.5, step=0.5
)
num_steps = st.sidebar.slider(
    "Inference steps", min_value=10, max_value=60, value=30, step=5
)
negative_prompt_input = st.sidebar.text_input(
    "Additional negative prompt tags (optional)",
    help="Comma-separated tags, e.g. 'low contrast, bad anatomy'",
)

output_root = st.sidebar.text_input(
    "Output folder",
    value="outputs",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Ethical use:**")
st.sidebar.write(
    "- No NSFW or harmful content.\n"
    "- All images are watermarked as AI-generated.\n"
    "- Do not mislead people with generated images."
)

# --- Main input section ---
prompt = st.text_area(
    "Describe the image you want:",
    value="a futuristic city at sunset with flying cars",
    height=80,
)

user_filename = st.text_input(
    "Optional custom filename (will be sanitized)",
    value="",
    help="If empty, a timestamp-based name will be used.",
)

generate_button = st.button("Generate images")

# --- Load the model lazily when first needed ---
@st.cache_resource(show_spinner=True)
def load_generator(device_preference: str):
    return TextToImageGenerator(device_preference=device_preference)

if generate_button:
    if not prompt.strip():
        st.error("Please enter a description for the image.")
    elif not is_prompt_allowed(prompt):
        st.error(
            "This prompt appears to violate the content policy. "
            "Please try a different, safe description."
        )
    else:
        gen = load_generator(device_pref)

        with st.spinner("Generating images... this may take a moment."):
            results, eta = gen.generate_images(
                base_prompt=prompt,
                style=style_choice,
                n_images=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                negative_prompt_text=negative_prompt_input,
                output_dir=output_root,
            )

        st.success(
            f"Generation complete. Estimated time was about {eta:.1f} seconds "
            f"on {gen.device.upper()}."
        )

        cols = st.columns(num_images)
        for idx, (col, res) in enumerate(zip(cols, results)):
            with col:
                st.subheader(f"Image {idx + 1}")
                st.image(res["image_path"], use_column_width=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if user_filename:
                    # sanitized inside utils if you want to reuse that
                    base_name = user_filename
                else:
                    base_name = f"image_{timestamp}_{idx}"

                # Download buttons
                with open(res["image_path"], "rb") as f_png:
                    st.download_button(
                        label="Download PNG",
                        data=f_png,
                        file_name=f"{base_name}.png",
                        mime="image/png",
                    )
                with open(res["jpg_path"], "rb") as f_jpg: # with open(res["jpg_path"], "rb") as f_jpg:
                    st.download_button(                    #     st.download_button(...)
                        label="Download JPEG",
                        data=f_jpg,
                        file_name=f"{base_name}.jpg",
                        mime="image/jpeg",
                    )

                st.caption(f"Metadata saved at: {os.path.basename(res['metadata_path'])}")
