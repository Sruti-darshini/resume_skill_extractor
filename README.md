# Resume Information Extractor

This application extracts key information (name, email, phone number, and skills) from resumes in PDF or DOCX format.

## Setup Instructions

### Backend Setup
1. Install Python dependencies:
```bash
pip install -r requirements.txt
```
2. Install spaCy English model:
```bash
python -m spacy download en_core_web_sm
```
3. Run the Flask backend:
```bash
python app.py
```

### Frontend Setup
1. Navigate to the frontend directory:
```bash
cd frontend
```
2. Install Node.js dependencies:
```bash
npm install
```
3. Start the development server:
```bash
npm start
```

The application will be available at http://localhost:3000

## Features
- Upload PDF or DOCX resumes
- Extract key information:
  - Name
  - Email address
  - Phone number
  - Skills
- Modern, responsive UI
- Real-time feedback
- Error handling
