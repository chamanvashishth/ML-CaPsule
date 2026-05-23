import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
import speech_recognition as sr

st.title("Speech/Text to Image Generator (Offline)")
st.markdown("### Streamlit + Offline Speech Recognition + Stable Diffusion")
st.markdown("No external speech API is used. Image generation may take time on CPU.")


@st.cache_resource
def load_pipeline():
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    if device == "cpu":
        pipe.enable_attention_slicing()

    return pipe, device


def recognize_speech_offline():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for speech...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)

    try:
        st.info("Transcribing with offline PocketSphinx...")
        text = recognizer.recognize_sphinx(audio)
        st.success(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError as error:
        st.error(f"Offline recognizer is unavailable: {error}")

    return None


def generate_image(prompt: str):
    pipe, device = load_pipeline()

    if device == "cuda":
        with torch.autocast("cuda"):
            output = pipe(prompt, guidance_scale=8.5)
    else:
        output = pipe(prompt, guidance_scale=8.5)

    return output.images[0]


if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""

st.session_state.prompt_text = st.text_input(
    "Enter a prompt for image generation:",
    value=st.session_state.prompt_text,
)

if st.button("Recognize Speech (Offline)"):
    recognized_text = recognize_speech_offline()
    if recognized_text:
        st.session_state.prompt_text = f"{recognized_text}, 4k, high resolution"
        st.info(f"Using recognized prompt: {st.session_state.prompt_text}")

if st.button("Generate Image"):
    if st.session_state.prompt_text.strip():
        with st.spinner("Generating image..."):
            image = generate_image(st.session_state.prompt_text)
            st.image(image, caption="Generated Image", use_container_width=True)
            image.save("generated_image.png")
            st.success("Image generated successfully!")
    else:
        st.warning("Please enter a prompt or use speech recognition first.")

# Contribution note: prepared for GSSoC participants and keeps contributor identity neutral in-app.
