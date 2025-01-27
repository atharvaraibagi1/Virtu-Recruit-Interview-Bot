import re
from pypdf import PdfReader
import os
import streamlit as st


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
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


def process_resume(pdf_path):
    """Extract and segment text from a resume."""
    text = extract_text_from_pdf(pdf_path)
    resume_segments = segment_text(text)
    return resume_segments

st.title("Virtu Recruit - AI Interview Bot")

st.write("Please upload your resume (PDF format) below:")

resume_pdf = st.file_uploader("Upload Resume", type=['pdf'])

if resume_pdf is not None:
    resume_dict = process_resume(resume_pdf)

    st.header(":scroll: Extracted contents from your resume:")

    for key, value in resume_dict.items():
        st.subheader(key)
        st.write(value)








