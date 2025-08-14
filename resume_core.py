import os
import re
from typing import Dict, Set, DefaultDict
from collections import defaultdict

# Lightweight deps
import PyPDF2
from docx import Document

# NLTK (small footprint compared to deep learning libs)
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk


# Initialize lightweight NLP utilities (download lazily if needed)
def _ensure_nltk_data() -> None:
    required = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
        ('corpora/words', 'words'),
    ]
    for path, name in required:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    _ensure_nltk_data()
    stop_words = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    return re.sub(r'[^\x00-\x7F]+', ' ', text)


def extract_email(text: str):
    try:
        lines = text.lower().split('\n')
        for i, line in enumerate(lines):
            if any(k in line for k in ['email', 'e-mail', 'mail', '@']):
                search_text = '\n'.join(lines[i:i+3])
                pattern = r'(?i)([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})'
                matches = re.findall(pattern, search_text)
                if matches:
                    return matches[0].strip()
        pattern = r'(?i)([a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,})'
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
        return None
    except Exception:
        return None


def extract_phone(text: str):
    pattern = r'\+?\d{1,3}[-\s]?\(?\d{2,4}\)?[-\s]?\d{3,4}[-\s]?\d{3,4}'
    matches = re.findall(pattern, text)
    for match in matches:
        phone = clean_text(match)
        phone = re.sub(r'[^\d+]', '', phone)
        if phone.startswith('+'):
            return phone
        if len(phone) == 10:
            return f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
        if len(phone) > 10:
            return phone
    return None


def extract_text_from_pdf(pdf_file) -> str:
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ''
            except Exception:
                page_text = ''
            if page_text:
                text += page_text + '\n'
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\x0c', '\n')
        text = '\n'.join(line.strip() for line in text.splitlines())
        return text
    except Exception:
        return ''


def extract_text_from_docx(docx_file) -> str:
    doc = Document(docx_file)
    return '\n'.join(p.text for p in doc.paragraphs)


def extract_name(text: str):
    try:
        first_chunk = '\n'.join(text.split('\n')[:5])
        name_patterns = [
            r'(?i)name\s*[:-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'(?i)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$',
            r'(?i)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*[,\n]'
        ]
        for pattern in name_patterns:
            matches = re.findall(pattern, first_chunk)
            if matches:
                candidate = matches[0].strip()
                if len(candidate.split()) >= 2:
                    return candidate
        _ensure_nltk_data()
        tokens = word_tokenize(first_chunk)
        pos_tags = pos_tag(tokens)
        ne_chunks = ne_chunk(pos_tags)
        for chunk in ne_chunks:
            if hasattr(chunk, 'label'):
                name = ' '.join(c[0] for c in chunk)
                name = clean_text(name)
                if len(name.split()) >= 2:
                    return name
        lines = [line.strip() for line in first_chunk.split('\n')]
        for line in lines:
            if any(skip in line.lower() for skip in ['resume', 'cv', 'curriculum', 'vitae', '@', 'email', 'phone', 'address', 'http', 'www']):
                continue
            clean_line = ' '.join(word for word in line.split() if word.isalpha())
            words = clean_line.split()
            if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words):
                return clean_line
        return None
    except Exception:
        return None


# Skill DB and helpers (lightweight)
SKILL_DB: Dict[str, Set[str]] = {
    'Programming Languages': {
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift',
        'kotlin', 'go', 'rust', 'scala', 'perl', 'r', 'matlab', 'sql', 'bash', 'powershell'
    },
    'Web Technologies': {
        'html', 'css', 'react', 'angular', 'vue.js', 'node.js', 'express.js', 'django',
        'flask', 'spring', 'asp.net', 'jquery', 'bootstrap', 'tailwind', 'webpack',
        'graphql', 'rest api', 'web services', 'microservices'
    },
    'Databases': {
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sql server',
        'sqlite', 'cassandra', 'dynamodb', 'mariadb', 'neo4j', 'firebase', 'nosql'
    },
    'Cloud & DevOps': {
        'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'terraform',
        'ansible', 'circleci', 'github actions', 'gitlab ci', 'prometheus', 'grafana',
        'devops', 'ci/cd', 'cloud computing'
    },
    'AI & Data Science': {
        'machine learning', 'deep learning', 'neural networks', 'nlp', 'computer vision',
        'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'keras', 'opencv',
        'data analysis', 'data visualization', 'big data', 'hadoop', 'spark'
    },
    'Tools & Methodologies': {
        'git', 'jira', 'agile', 'scrum', 'kanban', 'tdd', 'unit testing', 'ci/cd',
        'rest', 'soap', 'design patterns', 'oop', 'functional programming'
    }
}


def extract_skills_rule_based(text: str) -> Dict[str, Set[str]]:
    skills: DefaultDict[str, Set[str]] = defaultdict(set)
    text_lower = text.lower()
    for category, category_skills in SKILL_DB.items():
        for skill in category_skills:
            s = skill.lower()
            if any(p in text_lower for p in [s, s.replace(' ', '-'), s.replace(' ', '_'), s.replace(' ', '')]):
                skills[category].add(skill)
    return skills


def extract_skills_ner(text: str) -> Dict[str, Set[str]]:
    try:
        _ensure_nltk_data()
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        ne_chunks = ne_chunk(pos_tags)
        skills: DefaultDict[str, Set[str]] = defaultdict(set)
        for chunk in ne_chunks:
            if hasattr(chunk, 'label'):
                skill = ' '.join(c[0] for c in chunk)
                skill_lower = skill.lower()
                for category, category_skills in SKILL_DB.items():
                    if any(s.lower() in skill_lower or skill_lower in s.lower() for s in category_skills):
                        skills[category].add(skill)
                        break
        return skills
    except Exception:
        return defaultdict(set)


def extract_skills(text: str) -> Dict[str, Set[str]]:
    # Baseline: rule-based + NER. Semantic (transformers/torch) is disabled by default.
    rule_based = extract_skills_rule_based(text)
    ner_based = extract_skills_ner(text)
    final: DefaultDict[str, Set[str]] = defaultdict(set)
    for category, skills in rule_based.items():
        if skills:
            final[category].update(skills)
    for category, skills in ner_based.items():
        if skills:
            final[category].update(skills)
    # Optional semantic similarity if explicitly enabled
    if os.getenv('RSX_ENABLE_SEMANTIC', '0') == '1':
        try:
            import importlib
            sentence_transformers = importlib.import_module('sentence_transformers')
            SentenceTransformer = sentence_transformers.SentenceTransformer
            # Late import numpy only if needed
            import numpy as np  # noqa: F401
            # Minimal semantic augmentation could be added here if desired
        except Exception:
            # Ignore if unavailable
            pass
    # Post-process: deduplicate and sort
    processed: Dict[str, Set[str]] = {}
    for category, skill_set in final.items():
        if not skill_set:
            continue
        skills_list = sorted(skill_set, key=len, reverse=True)
        filtered: Set[str] = set()
        for skill in skills_list:
            if not any(skill != other and skill in other for other in filtered):
                filtered.add(skill)
        if filtered:
            processed[category] = sorted(filtered)  # type: ignore[assignment]
    return processed


