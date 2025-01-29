import re
import librosa
import numpy as np
from pypdf import PdfReader
import io
import streamlit as st
import time
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import soundfile as sf
from tensorflow.keras.models import load_model
import joblib
from scipy.stats import zscore

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        .category-label {
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.9em;
            font-weight: bold;
            margin-bottom: 10px;
            display: inline-block;
        }
        .general {
            background-color: #e3f2fd;
            color: #1565c0;
        }
        .statistics {
            background-color: #f3e5f5;
            color: #7b1fa2;
        }
        .ml {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .resume {
            background-color: #fff3e0;
            color: #e65100;
        }
        .score-card {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .score-excellent {
            background-color: #e8f5e9;
            border: 1px solid #2e7d32;
        }
        .score-good {
            background-color: #e3f2fd;
            border: 1px solid #1565c0;
        }
        .score-improve {
            background-color: #fff3e0;
            border: 1px solid #e65100;
        }
        .feedback-section {
            margin-top: 1rem;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 10px;
        }
        .emotion-card {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .emotion-meter {
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        .emotion-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        .confidence { background-color: #4CAF50; }
        .anxiety { background-color: #FF9800; }
        .fear { background-color: #f44336; }
        .neutral { background-color: #2196F3; }
        .enthusiasm { background-color: #9C27B0; }
    </style>
""", unsafe_allow_html=True)

def play_audio(question):
    """Generate and play audio for a given question."""
    output_file = f"question_audio_{hash(question)}.mp3"
    if text_to_speech(question, output_file):
        try:
            with open(output_file, "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error(f"Error playing audio: {str(e)}")

# Audio processing functions
def extract_audio_features(audio_path):
    """Extract audio features for emotion analysis."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, duration=30)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Energy and zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        energy = np.sum(y**2) / len(y)
        
        # Combine features
        features = np.concatenate([
            mfcc_scaled,
            [np.mean(spectral_centroids)],
            [np.mean(spectral_rolloff)],
            [np.mean(zero_crossing_rate)],
            [energy]
        ])
        
        # Normalize features
        features_normalized = zscore(features)
        
        return features_normalized
    except Exception as e:
        st.error(f"Error extracting audio features: {str(e)}")
        return None

class EmotionAnalyzer:
    def __init__(self):
        """Initialize the emotion analyzer with pre-trained models."""
        try:
            model_path = "emotion_model.h5"
            scaler_path = "emotion_scaler.joblib"
            
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading emotion analysis models: {str(e)}")
            self.model = None
            self.scaler = None
    
    def analyze_emotions(self, audio_path):
        """Analyze emotions in an audio file."""
        try:
            features = extract_audio_features(audio_path)
            if features is None:
                return None
            
            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predict emotions
            emotions_pred = self.model.predict(features_scaled)
            
            # Convert to emotion dictionary
            emotions = {
                'confidence': float(emotions_pred[0][0]),
                'anxiety': float(emotions_pred[0][1]),
                'fear': float(emotions_pred[0][2]),
                'neutral': float(emotions_pred[0][3]),
                'enthusiasm': float(emotions_pred[0][4])
            }
            
            return emotions
        except Exception as e:
            st.error(f"Error analyzing emotions: {str(e)}")
            return None

# Initialize emotion analyzer
emotion_analyzer = EmotionAnalyzer()

# PDF processing functions
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        pdf_bytes = io.BytesIO(pdf_file.read())
        reader = PdfReader(pdf_bytes)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

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

# OpenAI API functions
def get_whisper_transcription(audio_file_path):
    """Transcribe audio to text using OpenAI's Whisper API."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""

def get_ideal_answer(question):
    """Generate an ideal answer using ChatGPT."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data scientist providing concise but comprehensive interview answers."},
                {"role": "user", "content": f"Provide a model answer for the following interview question:\n\n{question}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating ideal answer: {str(e)}")
        return ""

def get_feedback(question, user_answer, ideal_answer, score, emotions):
    """Generate feedback using ChatGPT, including emotion analysis."""
    try:
        emotion_feedback = ""
        if emotions:
            emotion_feedback = "\nVoice Emotion Analysis:\n"
            for emotion, value in emotions.items():
                emotion_feedback += f"- {emotion.title()}: {value*100:.1f}%\n"
        
        feedback_prompt = f"""
        Question: {question}
        Candidate's Answer: {user_answer}
        Ideal Answer: {ideal_answer}
        Score: {score}
        {emotion_feedback}
        
        Please provide specific, constructive feedback on:
        1. Technical content accuracy and completeness
        2. Communication style and confidence level
        3. Areas for improvement in both content and delivery
        4. How to improve emotional aspects (confidence, reduce anxiety if present)
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert interviewer providing constructive feedback."},
                {"role": "user", "content": feedback_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating feedback: {str(e)}")
        return ""

def calculate_similarity_score(user_answer, ideal_answer):
    """Calculate similarity score between user's answer and ideal answer using embeddings."""
    try:
        user_embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=user_answer
        ).data[0].embedding

        ideal_embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=ideal_answer
        ).data[0].embedding

        user_embedding = np.array(user_embedding)
        ideal_embedding = np.array(ideal_embedding)
        
        similarity = np.dot(user_embedding, ideal_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(ideal_embedding))
        
        score = int(similarity * 100)
        return max(0, min(100, score))
    except Exception as e:
        st.error(f"Error calculating similarity score: {str(e)}")
        return 0

def generate_resume_based_questions(resume_content):
    """Generate personalized questions from ChatGPT based on resume content."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an interview assistant helping candidates prepare for a data science interview."},
                {"role": "user", "content": f"Based on the following resume content, generate 3 detailed interview questions (do not put numbering in the start, just text):\n\n{resume_content}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        questions = response.choices[0].message.content.strip().split('\n')
        return [q for q in questions if q.strip()]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []
    
def process_resume(uploaded_file):
    """Extract and segment text from a resume."""
    text = extract_text_from_pdf(uploaded_file)
    if text:
        uploaded_file.seek(0)
        return segment_text(text)
    return {}

def text_to_speech(text, output_file="output.mp3"):
    """Convert text to speech using OpenAI's TTS API."""
    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice="sage",
            input=text
        )
        
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        response.stream_to_file(output_file)
        return True
    except Exception as e:
        st.error(f"Error in TTS API call: {str(e)}")
        return False

def display_category_label(category):
    """Display styled category label."""
    category_class = category.lower().replace(" ", "-")
    st.markdown(f"""
        <div class="category-label {category_class}">
            {category}
        </div>
    """, unsafe_allow_html=True)

def display_emotion_analysis(emotions):
    """Display emotion analysis results with visual meters."""
    st.markdown("""
        <div class="emotion-card">
            <h4>🎭 Voice Emotion Analysis</h4>
    """, unsafe_allow_html=True)
    
    for emotion, value in emotions.items():
        percentage = value * 100
        color_class = emotion.lower()
        
        st.markdown(f"""
            <div>
                <span>{emotion.title()}: {percentage:.1f}%</span>
                <div class="emotion-meter">
                    <div class="emotion-fill {color_class}" 
                         style="width: {percentage}%"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def clear_audio_state(question_index):
    """Clear audio-related session state for a specific question."""
    keys_to_clear = [
        f"audio_recorder_{question_index}",
        f"audio_recorder_{question_index}_key",
        f"audio_recorder_{question_index}_counter"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Initialize session state
if 'question_index' not in st.session_state:
    st.session_state.question_index = 0
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'interview_started' not in st.session_state:
    st.session_state.interview_started = False
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'answer_scores' not in st.session_state:
    st.session_state.answer_scores = {}
if 'transcribed_responses' not in st.session_state:
    st.session_state.transcribed_responses = {}
if 'ideal_answers' not in st.session_state:
    st.session_state.ideal_answers = {}
if 'emotion_analysis' not in st.session_state:
    st.session_state.emotion_analysis = {}

# Create output directory for audio files
os.makedirs('audio_outputs', exist_ok=True)

# Main UI
st.markdown("<h1>🤖 VirtuRecruit - AI Interview Practice Bot</h1>", unsafe_allow_html=True)

# Introduction Section with Markdown
st.markdown("""
📊 Upload your **resume**, and VirtuRecruit will generate **personalized interview questions** based on your experience and key skills. It covers **statistics**, **machine learning**, and **data science** topics. The interactive setup allows you to practice in real-time, simulating a real interview environment. 💼
""")

# Feature List with Bullet Points and Icons
st.subheader("🔑 Key Features:")
st.markdown("""
- **Resume-Based Questions**: Tailored questions based on your resume. 📝  
- **Technical Question Bank**: Includes **text** and **audio** questions on **stats**, **ML**, and **DS**. 🧠  
- **Answer Recording**: Record answers like a real interview. 🎤  
- **AI Evaluation & Feedback**: Receive **scores** and **detailed feedback**. 📈  
""")

# Styled Section for User Experience
st.markdown("""
### 🌟 Why Choose VirtuRecruit?

- **Tailored to You**: VirtuRecruit customizes the interview process based on your unique skills and experiences, ensuring relevant preparation every time.  
- **Realistic Interview Simulation**: Mimic actual interview conditions with both technical and soft-skill practice, giving you the full experience.  
- **Instant AI Feedback**: Receive real-time evaluations of your answers, helping you learn and improve instantly.  
- **Boost Your Confidence**: By practicing with personalized questions and detailed feedback, you'll enter every interview with the confidence to succeed.  
- **Prepare for Any Scenario**: From coding questions to explaining algorithms, VirtuRecruit ensures you're ready for any question, technical or otherwise.  
""")


# Sidebar
with st.sidebar:
    st.header("📝 Instructions")
    st.info("""
    1. Upload your resume (PDF)
    2. Review extracted information
    3. Start the interview
    4. Listen to questions
    5. Record your responses
    6. Get instant feedback and emotional analysis
    """)
    
    if st.session_state.interview_started:
        st.header("🎯 Progress")
        total_questions = len(st.session_state.generated_questions) + 10
        progress = st.session_state.question_index / total_questions
        st.progress(progress)
        st.write(f"Question {st.session_state.question_index + 1} of {total_questions}")

# Main content
resume_pdf = st.file_uploader("📄 Upload Your Resume (PDF format)", type=['pdf'])

if resume_pdf is not None:
    if not st.session_state.interview_started:
        with st.spinner("Processing resume..."):
            resume_dict = process_resume(resume_pdf)
            
            if resume_dict:
                st.success("Resume processed successfully!")
                
                with st.expander("📋 View Extracted Content"):
                    for section, content in resume_dict.items():
                        st.subheader(section)
                        st.write(content)
                
                resume_content = resume_dict.get('WORK EXPERIENCE', '') + '\n' + resume_dict.get('PROJECTS & PAPERS', '')
                st.session_state.generated_questions = generate_resume_based_questions(resume_content)
                
                if st.button("🎯 Start Interview"):
                    st.session_state.interview_started = True
                    st.rerun()

if st.session_state.interview_started:
    # Questions pool
    base_questions = [
        "Tell me about yourself and your interest in data science.",
        "What's the most challenging data science project you've worked on?",
        "How do you handle missing or inconsistent data?",
        "Where do you see yourself in 5 years in the field of data science?"
    ]

    stats_questions = [
        "Can you explain the difference between Type I and Type II errors in hypothesis testing? Provide a real-world example in data science.",
        "How would you explain the Central Limit Theorem and its importance in data analysis? When does it not apply?",
        "What is the difference between correlation and causation? Can you provide an example where correlation might falsely imply causation in a data science context?"
    ]

    ml_questions = [
        "Explain the bias-variance tradeoff in machine learning. How do you handle this tradeoff in practice?",
        "What's the difference between bagging and boosting in ensemble learning? When would you prefer one over the other?",
        "How do you handle class imbalance in a classification problem? What are the pros and cons of different approaches?"
    ]

    all_questions = base_questions + st.session_state.generated_questions + stats_questions + ml_questions
    
    if st.session_state.question_index < len(all_questions):
        current_question = all_questions[st.session_state.question_index]
        
        # Determine question category
        if current_question in base_questions:
            category = "General Question"
        elif current_question in stats_questions:
            category = "Statistics Question"
        elif current_question in ml_questions:
            category = "Machine Learning Question"
        else:
            category = "Resume-Based Question"
            
        # Display question with category
        display_category_label(category)
        st.header(f"Question {st.session_state.question_index + 1}")
        st.info(current_question)
        play_audio(current_question)
        
        # Recording section
        st.subheader("🎤 Your Response")
        audio_response = st.audio_input(
            "Record your answer",
            key=f"audio_recorder_{st.session_state.question_index}"
        )

        if audio_response is not None:
            filename = os.path.join('audio_outputs', f"response_{uuid.uuid4()}.wav")
            with open(filename, "wb") as f:
                f.write(audio_response.read())
            
            st.session_state.responses[st.session_state.question_index] = filename
            
            # Process response immediately
            transcription = get_whisper_transcription(filename)
            ideal_answer = get_ideal_answer(current_question)
            score = calculate_similarity_score(transcription, ideal_answer)
            emotions = emotion_analyzer.analyze_emotions(filename)
            
            # Store results
            st.session_state.transcribed_responses[st.session_state.question_index] = transcription
            st.session_state.ideal_answers[st.session_state.question_index] = ideal_answer
            st.session_state.answer_scores[st.session_state.question_index] = score
            st.session_state.emotion_analysis[st.session_state.question_index] = emotions
            
            st.success("Response recorded and analyzed!")
            
            # Display immediate feedback
            with st.expander("View Answer Analysis"):
                st.write("**Your Response (Transcribed):**")
                st.write(transcription)
                
                st.write("\n**Score:**")
                st.progress(score/100)
                st.write(f"{score}/100")
                
                if emotions:
                    display_emotion_analysis(emotions)
            
            if st.button("Next Question ➡️", key=f"next_{st.session_state.question_index}"):
                clear_audio_state(st.session_state.question_index)
                st.session_state.question_index += 1
                st.rerun()
    
    else:
        st.header("🎉 Interview Complete!")
        st.success("Great job! Here's your comprehensive evaluation:")
        
        # Calculate overall statistics
        total_score = 0
        num_questions = len(st.session_state.responses)
        
        # Create columns for summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Questions Answered", num_questions)
        
        # Calculate average score
        avg_score = sum(st.session_state.answer_scores.values()) / num_questions if num_questions > 0 else 0
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/100")
        
        # Calculate emotional averages
        all_emotions = {
            'confidence': [],
            'anxiety': [],
            'fear': [],
            'neutral': [],
            'enthusiasm': []
        }
        
        for emotions in st.session_state.emotion_analysis.values():
            if emotions:
                for emotion, value in emotions.items():
                    all_emotions[emotion].append(value)
        
        avg_emotions = {
            emotion: np.mean(values) if values else 0
            for emotion, values in all_emotions.items()
        }
        
        # Display dominant emotion
        if avg_emotions:
            dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
            with col3:
                st.metric("Dominant Emotion", dominant_emotion.title())
        
        # Display detailed results for each question
        st.subheader("📊 Detailed Analysis")
        
        for idx in st.session_state.responses.keys():
            question = all_questions[idx]
            transcription = st.session_state.transcribed_responses[idx]
            ideal_answer = st.session_state.ideal_answers[idx]
            score = st.session_state.answer_scores[idx]
            emotions = st.session_state.emotion_analysis[idx]
            
            with st.expander(f"Question {idx + 1}"):
                st.write("**Question:**", question)
                st.audio(st.session_state.responses[idx])
                
                st.write("\n**Your Response (Transcribed):**")
                st.write(transcription)
                
                st.write("\n**Ideal Answer:**")
                st.write(ideal_answer)
                
                score_class = (
                    "score-excellent" if score >= 80
                    else "score-good" if score >= 60
                    else "score-improve"
                )
                
                st.markdown(f"""
                    <div class="score-card {score_class}">
                        <h4>Score: {score}/100</h4>
                        <p>{'🌟 Excellent!' if score >= 80 
                           else '👍 Good job!' if score >= 60 
                           else '💪 Room for improvement'}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                if emotions:
                    display_emotion_analysis(emotions)
                
                feedback = get_feedback(question, transcription, ideal_answer, score, emotions)
                st.markdown("""
                    <div class="feedback-section">
                        <h4>💡 Detailed Feedback:</h4>
                        <div>
                            {}
                        </div>
                    </div>
                """.format(feedback), unsafe_allow_html=True)
        
        # Overall emotional intelligence analysis
        st.subheader("🎭 Overall Emotional Intelligence Analysis")
        if avg_emotions:
            display_emotion_analysis(avg_emotions)
            
            # Generate emotional intelligence recommendations
            emotion_recommendation_prompt = f"""
            Based on the candidate's emotional profile:
            {', '.join(f'{k}: {v*100:.1f}%' for k, v in avg_emotions.items())}
            
            Please provide specific recommendations for:
            1. How to improve confidence in interviews
            2. Managing anxiety and fear
            3. Maintaining professional enthusiasm
            4. Balancing emotions during technical discussions
            """
            
            try:
                emotion_recommendations = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert in interview coaching and emotional intelligence."},
                        {"role": "user", "content": emotion_recommendation_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                ).choices[0].message.content
                
                st.markdown("### 🎯 Emotional Intelligence Recommendations")
                st.write(emotion_recommendations)
            except Exception as e:
                st.error("Error generating emotional intelligence recommendations")
        
        # Option to start new interview
        if st.button("Start New Interview"):
            # Clean up audio files
            for response_file in st.session_state.responses.values():
                if os.path.exists(response_file):
                    try:
                        os.remove(response_file)
                    except Exception:
                        pass
            
            # Reset session state
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()