import re
from pypdf import PdfReader
import io
import streamlit as st
import time
import uuid
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import librosa
import soundfile as sf
from transformers import pipeline
import torch
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")

def extract_audio_features(audio_file):
    """Extract audio features for emotion analysis."""
    try:
        # Load audio file
        y, sr = librosa.load(audio_file, duration=30)
        
        # Check if audio is too short
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 0.1:
            return None, {
                'confident': 20,
                'anxious': 15,
                'fear': 10,
                'neutral': 40,
                'passionate': 15
            }
        
        # Extract features
        features = {
            'mfcc': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0),
            'chroma': np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0),
            'mel': np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0),
            'contrast': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0),
            'tonnetz': np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
        }
        
        # Calculate audio characteristics
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        zero_crossings = librosa.feature.zero_crossing_rate(y=y)[0]
        
        # Use audio characteristics to influence emotion scores
        confidence_score = min(100, max(0, 
            40 + (tempo - 120) * 0.2 + np.mean(rms) * 100
        ))
        
        anxiety_score = min(100, max(0,
            20 + np.mean(zero_crossings) * 100 + (tempo - 100) * 0.3
        ))
        
        passion_score = min(100, max(0,
            30 + np.mean(rms) * 150 + (tempo - 110) * 0.2
        ))
        
        fear_score = min(100, max(0,
            15 + np.mean(zero_crossings) * 80 - np.mean(rms) * 50
        ))
        
        # Calculate neutral as the remaining percentage
        other_emotions = confidence_score + anxiety_score + passion_score + fear_score
        neutral_score = max(0, min(100, (100 - other_emotions) * 0.8))
        
        # Normalize scores to sum to 100
        total = confidence_score + anxiety_score + passion_score + fear_score + neutral_score
        if total > 0:
            emotion_scores = {
                'confident': (confidence_score / total) * 100,
                'anxious': (anxiety_score / total) * 100,
                'fear': (fear_score / total) * 100,
                'neutral': (neutral_score / total) * 100,
                'passionate': (passion_score / total) * 100
            }
        else:
            emotion_scores = {
                'confident': 20,
                'anxious': 15,
                'fear': 10,
                'neutral': 40,
                'passionate': 15
            }
        
        return features, emotion_scores
        
    except Exception as e:
        st.error(f"Error in audio feature extraction: {str(e)}")
        return None, {
            'confident': 20,
            'anxious': 15,
            'fear': 10,
            'neutral': 40,
            'passionate': 15
        }

# [Rest of the code remains the same]

def create_emotion_visualization(emotion_scores, title="Voice Emotion Analysis"):
    """Create visualizations for emotion analysis."""
    # Prepare data for plotting
    emotions = list(emotion_scores.keys())
    scores = list(emotion_scores.values())
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=emotions,
        fill='toself',
        name='Emotion Analysis'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=title
    )
    
    return fig

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
    </style>
""", unsafe_allow_html=True)

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
 
def calculate_similarity_score(user_answer, ideal_answer):
    """Calculate similarity score between user's answer and ideal answer using embeddings."""
    try:
        if not user_answer or not ideal_answer:
            return 50  # Default score for empty answers
            
        # Get embeddings for both texts
        user_embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=user_answer
        ).data[0].embedding

        ideal_embedding = client.embeddings.create(
            model="text-embedding-ada-002",
            input=ideal_answer
        ).data[0].embedding

        # Calculate cosine similarity
        user_embedding = np.array(user_embedding)
        ideal_embedding = np.array(ideal_embedding)
        
        # Add length penalty to prevent very short answers from getting high scores
        user_length = len(user_answer.split())
        ideal_length = len(ideal_answer.split())
        length_ratio = min(user_length / ideal_length, 1.0)
        
        # Calculate base similarity
        similarity = np.dot(user_embedding, ideal_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(ideal_embedding))
        
        # Apply length penalty and scale to 0-100
        score = int((similarity * 0.8 + length_ratio * 0.2) * 100)
        
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
    except Exception as e:
        st.error(f"Error calculating similarity score: {str(e)}")
        return 50  # Default score on error
    
    
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
            voice="alloy",
            input=text
        )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        # Save the audio file
        response.stream_to_file(output_file)
        return True
    except Exception as e:
        st.error(f"Error in TTS API call: {str(e)}")
        return False

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

def generate_resume_based_questions(resume_content):
    """Generate personalized questions from ChatGPT based on resume content."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an interview assistant helping candidates prepare for a data science interview."},
                {"role": "user", "content": f"Based on the following resume content, generate 5 detailed interview questions (do not put numbering in the start, just text):\n\n{resume_content}"}
            ],
            max_tokens=500,
            temperature=0.7
        )
        questions = response.choices[0].message.content.strip().split('\n')
        return [q for q in questions if q.strip()]
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return []

def get_feedback(question, user_answer, ideal_answer, score):
    """Generate feedback using ChatGPT."""
    try:
        feedback_prompt = f"""
        Question: {question}
        Candidate's Answer: {user_answer}
        Ideal Answer: {ideal_answer}
        Score: {score}
        
        Please provide specific, constructive feedback on:
        1. Key strengths in the answer
        2. Areas for improvement
        3. Specific suggestions to enhance the answer
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

def display_category_label(category):
    """Display styled category label."""
    category_class = category.lower().replace(" ", "-")
    st.markdown(f"""
        <div class="category-label {category_class}">
            {category}
        </div>
    """, unsafe_allow_html=True)

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
    6. Get instant feedback and scores
    """)
    
    if st.session_state.interview_started:
        st.header("🎯 Progress")
        total_questions = len(st.session_state.generated_questions) + 10
        # total_questions = len(all_questions)
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
# In the interview complete section, after processing each response:

if st.session_state.interview_started:
    base_questions = [
        "Tell me about yourself and your interest in data science.",
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
        "How do you handle class imbalance in a classification problem? What are the pros and cons of different approaches?",
        "Explaiin the assumptions of Linear Regression."
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
            st.success("Response recorded!")
            
            if st.button("Next Question ➡️", key=f"next_{st.session_state.question_index}"):
                # Clear audio state for current question
                clear_audio_state(st.session_state.question_index)
                st.session_state.question_index += 1
                st.rerun()
       
    else:
        # In the interview complete section, after processing responses:
        st.header("🎉 Interview Complete!")
        st.success("Great job! Here's your comprehensive evaluation:")
        
        total_score = 0
        num_questions = len(st.session_state.responses)
        
        # Create columns for summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Questions Answered", num_questions)
        
        # Process each response and calculate scores
        responses_data = []  # Store processed response data
        overall_emotion_scores = {
            'confident': 0,
            'anxious': 0,
            'fear': 0,
            'neutral': 0,
            'passionate': 0
        }
        
        # Initialize score accumulator
        scores = []
        
        for i, response_file in st.session_state.responses.items():
            # Process response
            current_question = all_questions[i]
            transcription = get_whisper_transcription(response_file)
            ideal_answer = get_ideal_answer(current_question)
            score = calculate_similarity_score(transcription, ideal_answer)
            feedback = get_feedback(current_question, transcription, ideal_answer, score)
            
            # Add score to list
            scores.append(float(score))
            
            # Extract audio features and emotion scores
            audio_features, emotion_scores = extract_audio_features(response_file)
            
            # Accumulate emotion scores
            if emotion_scores:
                for emotion, score in emotion_scores.items():
                    overall_emotion_scores[emotion] += float(score)
            
            # Store processed data
            responses_data.append({
                'question_index': i,
                'question': current_question,
                'transcription': transcription,
                'ideal_answer': ideal_answer,
                'score': score,
                'feedback': feedback,
                'audio_file': response_file,
                'emotion_scores': emotion_scores
            })
        
        # Calculate average score
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Calculate average emotion scores
        num_responses = len(responses_data)
        if num_responses > 0:
            overall_emotion_scores = {k: float(v)/num_responses for k, v in overall_emotion_scores.items()}
        
        # Display average score
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/100")
        
        # Calculate category scores
        category_scores = {
            "General": [],
            "Statistics": [],
            "Machine Learning": [],
            "Resume-Based": []
        }
        
        for data in responses_data:
            q_idx = data['question_index']
            score = float(data['score'])  # Convert to float
            if all_questions[q_idx] in base_questions:
                category_scores["General"].append(score)
            elif all_questions[q_idx] in stats_questions:
                category_scores["Statistics"].append(score)
            elif all_questions[q_idx] in ml_questions:
                category_scores["Machine Learning"].append(score)
            else:
                category_scores["Resume-Based"].append(score)
        
        # Calculate average scores per category
        category_averages = {}
        for category, scores in category_scores.items():
            if scores:
                category_averages[category] = sum(scores) / len(scores)
        
        # Find strongest and weakest areas
        if category_averages:
            strongest_area = max(category_averages.items(), key=lambda x: x[1])[0]
            weakest_area = min(category_averages.items(), key=lambda x: x[1])[0]
            
            with col3:
                st.metric("Strongest Area", strongest_area)
        
        # Display emotion analysis
        st.subheader("🎭 Voice Emotion Analysis")
        
        # Create and display the radar chart
        emotion_fig = create_emotion_visualization(overall_emotion_scores, "Overall Voice Emotion Analysis")
        st.plotly_chart(emotion_fig, key="overall_emotion_chart")
        
        # Add emotion analysis insights
        st.write("### 🔍 Emotion Analysis Insights")
        
        # Determine dominant emotions
        dominant_emotions = sorted(
            overall_emotion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        st.write(f"**Dominant Emotions:** {dominant_emotions[0][0].title()} ({dominant_emotions[0][1]:.1f}%) and {dominant_emotions[1][0].title()} ({dominant_emotions[1][1]:.1f}%)")
        
        # Provide emotion-based feedback
        emotion_feedback = ""
        if dominant_emotions[0][0] == 'confident':
            emotion_feedback += "Your confident tone is a great asset in interviews. Keep maintaining this level of assurance while ensuring to remain humble and receptive."
        elif dominant_emotions[0][0] == 'anxious' or dominant_emotions[0][0] == 'fear':
            emotion_feedback += "Try to work on managing interview anxiety through practice and preparation. Remember to take deep breaths and pace yourself while speaking."
        elif dominant_emotions[0][0] == 'neutral':
            emotion_feedback += "While maintaining composure is good, try to inject more enthusiasm into your responses to demonstrate your passion for the field."
        elif dominant_emotions[0][0] == 'passionate':
            emotion_feedback += "Your enthusiasm for the subject matter clearly shows in your voice. This passion can be very appealing to interviewers."
        
        st.write("**Feedback:**", emotion_feedback)
        
        # Display detailed results for each question
        st.subheader("📊 Detailed Analysis")
    
        for idx, data in enumerate(responses_data):
            with st.expander(f"Question {data['question_index'] + 1}"):
                # Question section
                st.write("**Question:**", data['question'])
                st.audio(data['audio_file'])
                
                # Display individual response emotion analysis with unique key
                # if data['emotion_scores']:
                #     st.write("\n**Response Emotion Analysis:**")
                #     response_emotion_fig = create_emotion_visualization(
                #         data['emotion_scores'],
                #         f"Question {data['question_index'] + 1} Emotion Analysis"
                #     )
                #     st.plotly_chart(
                #         response_emotion_fig,
                #         key=f"emotion_chart_q{data['question_index']}"
                #     )
                
                # Response section
                st.write("\n**Your Response (Transcribed):**")
                st.write(data['transcription'])
                
                # Ideal answer section
                st.write("\n**Ideal Answer:**")
                st.write(data['ideal_answer'])
                
                # Determine score category
                if data['score'] >= 80:
                    score_message = "🌟 **Excellent!**"
                elif data['score'] >= 60:
                    score_message = "👍 **Good job!**"
                else:
                    score_message = "💪 **Room for improvement**"

                rounded_score = np.round(data['score'], 2)

                # Display Score Section
                # st.write(f"### Score: {rounded_score}/100")
                # st.write(score_message)

                # # Feedback Section
                # st.write("### 💡 Detailed Feedback:")
                # st.write(data['feedback'])
        
        # Overall recommendations
        st.subheader("🎯 Key Recommendations")
        recommendation_prompt = f"""
        Based on the following interview performance:
        - Average Score: {avg_score:.1f}/100
        - Strongest Area: {strongest_area}
        - Weakest Area: {weakest_area}
        
        Provide 3 key recommendations for improvement, focusing on:
        1. Technical knowledge
        2. Communication style
        3. Specific areas to study
        """
        
        try:
            recommendations = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert interviewer providing actionable recommendations."},
                    {"role": "user", "content": recommendation_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            ).choices[0].message.content
            
            st.write(recommendations)
        except Exception as e:
            st.error("Error generating recommendations")
        
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