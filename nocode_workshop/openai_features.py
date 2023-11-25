import streamlit as st
from basecode.authenticate import return_api_key
from st_audiorec import st_audiorec
import os
import openai
import requests
import base64
import tempfile
import io
from openai import OpenAI
import json
import time
import plotly.graph_objects as go

cwd = os.getcwd()
AUDIO_DIRECTORY = os.path.join(cwd, "audio_files")

if not os.path.exists(AUDIO_DIRECTORY):
	os.makedirs(AUDIO_DIRECTORY)

openai.api_key = return_api_key()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=return_api_key(),
)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get file extension
def get_file_extension(file_name):
    return os.path.splitext(file_name)[-1]



def analyse_image():
    st.subheader("Analyse an image")
    api_key = return_api_key()
    # Streamlit: File Uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    img_file_buffer = st.camera_input("Take a picture")
    prompt = st.text_input("Enter a prompt", value="This is a photo of a")
    if st.button("Analyse"):
        if uploaded_file is not None or img_file_buffer is not None:
            # Save the file to a temporary file
            if img_file_buffer is not None:
                uploaded_file = img_file_buffer
            extension = get_file_extension(uploaded_file.name)
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            # Encode the image
            base64_image = encode_image(temp_file_path)

            # Prepare the payload
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }

            # Send the request
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            # Display the response
            if response.status_code == 200:
                st.write(response.json())
                st.write(response.json()["choices"][0]["message"]["content"])
            else:
                st.error("Failed to get response")

            # Clean up the temporary file
            os.remove(temp_file_path)

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="text"
        )
    return transcript

def translate_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.translations.create(
        model="whisper-1", 
        file=audio_file
        )
        return transcript


def upload_audio():
    # Streamlit: File Uploader
    st.subheader("Transcribe an audio file")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Save the file to a temporary file
        extension = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Transcribe the audio
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                transcription_result = transcribe_audio(temp_file_path)
                st.write(transcription_result)

        # Clean up the temporary file
        os.remove(temp_file_path)

def record_myself():
    # Audio recorder
    st.subheader("Record and Transcribe an audio file")
    wav_audio_data = st_audiorec()

    if st.button("Transcribe (Maximum: 30 Seconds)") and wav_audio_data is not None:
        memory_file = io.BytesIO(wav_audio_data)
        memory_file.name = "recorded_audio.wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(wav_audio_data)

        with st.spinner("Transcribing..."):
            transcription_result = transcribe_audio(tmpfile.name)
            os.remove(tmpfile.name)  # Delete the temporary file manually after processing
            st.write(transcription_result)
            return transcription_result
    elif st.button("Translation (Maximum: 30 Seconds)") and wav_audio_data is not None:
        memory_file = io.BytesIO(wav_audio_data)
        memory_file.name = "recorded_audio.wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(wav_audio_data)

        with st.spinner("Translating..."):
            transcription_result = translate_audio(tmpfile.name)
            os.remove(tmpfile.name) 
            st.write(transcription_result)
            return transcription_result

def generate_image():
    st.subheader("Generate an image")
    i_prompt = st.text_input("Enter a prompt", value="Generate a photo of a")
    if st.button("Generate"):
        if i_prompt is not None or i_prompt != "Generate a photo of a":
            response = client.images.generate(
            model="dall-e-3",
            prompt=i_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            )

            image_url = response.data[0].url
            st.image(image_url)
        else:
            st.write("Please enter a prompt")


def text_speech(input_text):
    # Create a temporary file within the 'audio_files' directory
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=AUDIO_DIRECTORY, suffix='.mp3')
    
    # Generate speech
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=input_text
    )

    # Write the response content to the temporary file
    with open(temp_file.name, 'wb') as file:
        file.write(response.content)

    # Return the path of the temporary file
    return temp_file.name


def text_to_speech():
    st.subheader("Text to Speech")
    if 'audio_file_path' not in st.session_state:
        st.session_state.audio_file_path = None

    user_input = st.text_area("Enter your text here:")

    if user_input and st.button("Generate Speech from Text"):
        st.session_state.audio_file_path = text_speech(user_input)
        st.audio(st.session_state.audio_file_path)

    if st.session_state.audio_file_path and st.button("Reset"):
        # Remove the temporary file
        os.remove(st.session_state.audio_file_path)
        st.session_state.audio_file_path = None
        st.experimental_rerun()





