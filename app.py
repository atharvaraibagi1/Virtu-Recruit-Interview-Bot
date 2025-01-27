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

# Set page configuration
st.set_page_config(
    page_title="VirtuRecruit - AI Interview Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton > button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            background-color: #ff4b4b;
            color: white;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #ff3333;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .css-1d391kg {
            padding: 2rem;
            border-radius: 15px;
            background-color: #f7f7f7;
        }
        .stAudio {
            margin: 1rem 0;
        }
        h1 {
            color: #1E1E1E;
            text-align: center;
            margin-bottom: 2rem;
        }
        .subheader {
            color: #4A4A4A;
            font-size: 1.2em;
            margin: 1.5rem 0;
        }
        .progress-container {
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 10px;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

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
        st.error("🚫 Missing API keys. Please check your .env file.")
        return False

    with st.spinner("🎵 Generating audio..."):
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
            st.error(f"🚫 Error in API call: {str(e)}")
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
            st.error(f"🚫 Error playing audio: {str(e)}")

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

# Create output directory for audio files
if not os.path.exists('audio_outputs'):
    os.makedirs('audio_outputs')

# Main UI
st.markdown("<h1>🤖 VirtuRecruit - AI Interview Bot</h1>", unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.markdown("### 📝 Instructions")
    st.info("""
    1. Upload your resume in PDF format
    2. Review the extracted information
    3. Click 'Start Interview' when ready
    4. Listen to each question
    5. Record your response
    6. Click 'Next Question' to continue
    """)
    
    st.markdown("### 🎯 Interview Progress")
    questions = [
        "Hi, so you are interviewing for the position of Data Scientist. Tell me something about yourself.",
        "Can you walk me through one of the most interesting projects you've worked on?",
        "What motivates you to pursue a career in data science?",
        "How do you handle a situation where the data provided to you is incomplete or inconsistent?",
    ]
    # Fix: Convert percentage to decimal (0.0 to 1.0)
    progress = st.session_state.question_index / len(questions)
    st.progress(progress)
    st.markdown(f"Question {st.session_state.question_index + 1} of {len(questions)}")

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    resume_pdf = st.file_uploader("📄 Upload Your Resume (PDF format)", type=['pdf'])

if resume_pdf is not None:
    try:
        with st.spinner("📑 Processing resume..."):
            resume_dict = process_resume(resume_pdf)
        
        st.markdown("### 📋 Extracted Resume Contents")
        for key, value in resume_dict.items():
            with st.expander(f"📌 {key}"):
                st.write(value)
        
        if not st.session_state.interview_started:
            st.markdown("### Ready to begin?")
            if st.button("🎯 Start Interview!", key="start_interview"):
                st.session_state.interview_started = True
                st.experimental_rerun()
            
    except Exception as e:
        st.error(f"🚫 Error processing resume: {str(e)}")

if st.session_state.interview_started:
    if st.session_state.question_index < len(questions):
        st.markdown("---")
        question = questions[st.session_state.question_index]
        
        # Question display
        st.markdown(f"### 💭 Question {st.session_state.question_index + 1}:")
        st.info(question)
        play_audio(question)
        
        # Recording section
        st.markdown("### 🎤 Your Response")
        audio_data = st.audio_input(
            "Record your answer",
            key=f"audio_{st.session_state.question_index}"
        )

        if audio_data is not None:
            try:
                unique_filename = os.path.join('audio_outputs', f"response_{uuid.uuid4().hex}.wav")
                
                with open(unique_filename, "wb") as f:
                    if isinstance(audio_data, (bytes, bytearray)):
                        f.write(audio_data)
                    else:
                        f.write(audio_data.read())
                
                st.session_state.responses[st.session_state.question_index] = unique_filename
                
                st.success("✅ Response recorded successfully!")
                if st.button("➡️ Next Question"):
                    st.session_state.question_index += 1
                    st.session_state.current_response = None
                    st.session_state.next_clicked = True
                    st.experimental_rerun()
                    
            except Exception as e:
                st.error(f"🚫 Error processing audio: {str(e)}")
                
    else:
        st.markdown("---")
        st.markdown("## 🎉 Interview Complete!")
        st.success("Thank you for completing the interview. Here are your responses:")
        
        for i, response_file in st.session_state.responses.items():
            with st.expander(f"Question {i + 1}"):
                st.write(questions[i])
                try:
                    st.audio(response_file, format="audio/wav")
                except Exception as e:
                    st.error(f"🚫 Error playing response {i + 1}: {str(e)}")