# AI-Powered Interview Bot

An intelligent and interactive interview bot that automates the hiring process by parsing resumes, generating tailored questions, analyzing responses, and providing feedback.

---

## Features

- **Resume Parsing**: Extracts key information such as skills, experience, and projects from uploaded resumes.
- **Question Generation**: Creates dynamic, tailored interview questions based on resume content.
- **Speech-to-Text (STT)**: Converts spoken answers to text for further analysis.
- **Response Analysis**: Evaluates answers for relevance, depth, and sentiment using NLP models.
- **Text-to-Speech (TTS)**: Delivers natural-sounding questions to simulate a real interview.
- **Feedback**: Scores responses and provides detailed evaluations.

---

## Technologies Used

- **Language Models**: GPT for generating questions and analyzing responses.
- **Speech Processing**: Whisper or DeepSpeech for STT, and FastSpeech for TTS.
- **NLP Models**: Sentence-BERT for semantic analysis.
- **Web Framework**: Flask or FastAPI for backend development.
- **Frontend**: React for a dynamic and user-friendly interface.
- **Utilities**: PyPDF2, spaCy for resume parsing.

---

## Workflow

1. **Resume Upload**: Candidate uploads their resume (PDF format).
2. **Parsing**: Extract relevant details such as skills, experience, and projects.
3. **Dynamic Question Generation**: Create tailored questions based on resume data.
4. **Interview Process**: 
   - Questions delivered via TTS.
   - Candidate's spoken answers are converted to text using STT.
5. **Response Analysis**: Evaluate answers for content quality and sentiment.
6. **Feedback Generation**: Provide a detailed report including scores and recommendations.

---

## Future Enhancements

- Support for multiple languages in STT and TTS.
- Integration with ATS (Applicant Tracking Systems).
- Real-time interview monitoring for recruiters.
- Behavioral analysis using video processing.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

