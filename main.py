import re
from pypdf import PdfReader
import os


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

if __name__ == "__main__":
    pdf_path = r'C:\Users\Atharva\Downloads\Atharva_Raibagi_Resume.pdf'
    resume_dict = process_resume(pdf_path)
    
    print("Extracted Resume Segments:")
    for key, value in resume_dict.items():
        print(f"{key}: \n{value}\n")