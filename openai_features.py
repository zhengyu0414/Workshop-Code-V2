import streamlit as st
from authenticate import return_api_key
from st_audiorec import st_audiorec
import os
import streamlit as st
import openai
import requests
import base64
import tempfile
import io

openai.api_key = return_api_key()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get file extension
def get_file_extension(file_name):
    return os.path.splitext(file_name)[-1]

def analyse_image():
    api_key = return_api_key()
    # Streamlit: File Uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    prompt = st.text_input("Enter a prompt", value="This is a photo of a")
    if uploaded_file is not None:
        # Save the file to a temporary file
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
        else:
            st.error("Failed to get response")

        # Clean up the temporary file
        os.remove(temp_file_path)

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = openai.Audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="text"
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

