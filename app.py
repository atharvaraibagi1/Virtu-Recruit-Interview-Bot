import re
from pypdf import PdfReader
import io
import streamlit as st
import requests
import time
import uuid
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API keys from environment variables
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    pdf_bytes = io.BytesIO(pdf_file.read())
    reader = PdfReader(pdf_bytes)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def segment_text(text):
    """Segment text into dictionary with headings and their content."""
    headings = ['EDUCATION', 'WORK EXPERIENCE', 'PROJECTS & PAPERS', 
                'CERTIFICATIONS & SKILLS', 'EXTRA CURRICULAR ACTIVITIES']
    
    headings_pattern = '|'.join([re.escape(heading) for heading in headings])
    sections = re.split(rf"(?i)(?<=\n)({headings_pattern})(?=\n)", text)
    
    resume_dict = {}
    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""
        resume_dict[heading] = content

    return resume_dict

def process_resume(uploaded_file):
    """Extract and segment text from a resume."""
    text = extract_text_from_pdf(uploaded_file)
    uploaded_file.seek(0)
    resume_segments = segment_text(text)
    return resume_segments

def text_to_speech(text, output_file="output.mp3"):
    """Convert text to speech using Eleven Labs API."""
    if not ELEVEN_LABS_API_KEY or not VOICE_ID:
        st.error("Missing API keys. Please check your .env file.")
        return False

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_LABS_API_KEY,
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.75,
            "similarity_boost": 0.75,
        },
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error in API call: {str(e)}")
        return False

def play_audio(question):
    """Generate and play audio for a given question."""
    output_file = "question_audio.mp3"
    if text_to_speech(question, output_file):
        try:
            audio_file = open(output_file, "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")
            audio_file.close()
        except Exception as e:
            st.error(f"Error playing audio: {str(e)}")

# Initialize session state
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'next_clicked' not in st.session_state:
    st.session_state.next_clicked = False

# Create output directory for audio files if it doesn't exist
if not os.path.exists('audio_outputs'):
    os.makedirs('audio_outputs')

# Main UI
st.title("Virtu Recruit - AI Interview Bot")
st.write("Please upload your resume (PDF format) below:")

resume_pdf = st.file_uploader("Upload Resume", type=['pdf'])

questions = [
    "Hi, so you are interviewing for the position of Data Scientist. Tell me something about yourself.",
    "Can you walk me through one of the most interesting projects you've worked on?",
    "What motivates you to pursue a career in data science?",
    "How do you handle a situation where the data provided to you is incomplete or inconsistent?",
]

# Function to handle moving to next question
def next_question():
    st.session_state.question_index += 1
    st.session_state.current_response = None
    st.session_state.next_clicked = True

if resume_pdf is not None:
    try:
        resume_dict = process_resume(resume_pdf)
        
        st.header(":scroll: Extracted contents from your resume-")
        for key, value in resume_dict.items():
            st.subheader(key)
            st.write(value)
        
        if not st.session_state.interview_started and st.button("Start Interview!"):
            st.session_state.interview_started = True
            
    except Exception as e:
        st.error(f"Error processing resume: {str(e)}")

if st.session_state.interview_started:
    if st.session_state.question_index < len(questions):
        col1, col2 = st.columns(2)
        
        with col1:
            question = questions[st.session_state.question_index]
            st.subheader(f"Question {st.session_state.question_index + 1}:")
            st.write(question)
            play_audio(question)

        # Handle audio recording
        audio_data = st.audio_input(
            f"Record your response to Question {st.session_state.question_index + 1}",
            key=f"audio_{st.session_state.question_index}"
        )

        # Process new recording
        if audio_data is not None:
            try:
                # Generate unique filename in the audio_outputs directory
                unique_filename = os.path.join('audio_outputs', f"response_{uuid.uuid4().hex}.wav")
                
                # Save the audio data
                if isinstance(audio_data, (bytes, bytearray)):
                    with open(unique_filename, "wb") as f:
                        f.write(audio_data)
                else:
                    with open(unique_filename, "wb") as f:
                        f.write(audio_data.read())
                
                # Update session state
                st.session_state.responses[st.session_state.question_index] = unique_filename
                
                # Display the recorded audio
                # st.write("Your recorded response:")
                # st.audio(unique_filename, format="audio/wav")
                
                # Show next question button
                if st.button("Next Question", on_click=next_question):
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                
    if st.session_state.question_index >= len(questions):
        st.success("🎉 Congratulations! You have completed the interview.")
        
        st.subheader("Your Interview Responses")
        for i, response_file in st.session_state.responses.items():
            st.write(f"Question {i + 1}: {questions[i]}")
            try:
                st.audio(response_file, format="audio/wav")
            except Exception as e:
                st.error(f"Error playing response {i + 1}: {str(e)}")