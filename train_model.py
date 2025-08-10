import json
import numpy as np
import os
import random
from typing import List, Dict, Set
from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def prepare_training_data(json_file_path):
    """Prepare training data for sentence transformer."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    all_skills = set()
    
    # First pass: collect all skills
    for item in data:
        if 'annotation' in item:
            for annotation in item['annotation']:
                if 'Skills' in annotation.get('label', []):
                    for point in annotation.get('points', []):
                        skill = point.get('text', '').strip()
                        if skill:
                            all_skills.add(skill)
    
    all_skills = list(all_skills)
    if not all_skills:
        raise ValueError("No skills found in the training data")
    
    # Second pass: create examples
    for item in data:
        if 'content' in item and 'annotation' in item:
            text = item['content']
            item_skills = set()
            
            # Extract skills from this item
            for annotation in item['annotation']:
                if 'Skills' in annotation.get('label', []):
                    for point in annotation.get('points', []):
                        skill = point.get('text', '').strip()
                        if skill:
                            item_skills.add(skill)
            
            # Create positive examples
            for skill in item_skills:
                examples.append(InputExample(
                    texts=[text, skill],
                    label=1.0
                ))
            
            # Create negative examples (skills not in this item)
            negative_skills = [s for s in all_skills if s not in item_skills]
            if negative_skills and item_skills:
                # Use up to 3 negative examples per item
                for _ in range(min(3, len(negative_skills))):
                    negative_skill = random.choice(negative_skills)
                    examples.append(InputExample(
                        texts=[text, negative_skill],
                        label=0.0
                    ))
    
    if not examples:
        raise ValueError("No training examples could be created from the data")
        
    print(f"Created {len(examples)} training examples ({len([e for e in examples if e.label > 0.5])} positive, {len([e for e in examples if e.label <= 0.5])} negative)")
    return examples

def load_training_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # Try to parse as array first
        try:
            resumes = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to parse each line as a separate JSON object
            resumes = []
            for line in content.split('\n'):
                if line.strip():
                    try:
                        resume = json.loads(line)
                        resumes.append(resume)
                    except json.JSONDecodeError:
                        print(f'Warning: Could not parse line as JSON: {line[:100]}...')
                        continue
    
    # Prepare training data for NER
    ner_training_data = []
    skill_examples = []
    
    for resume in resumes:
        content = resume.get('content', '')
        annotations = resume.get('annotation', [])
        
        # Process annotations for NER training
        entities = []
        for annotation in annotations:
            if 'Skills' in annotation.get('label', []):
                for point in annotation.get('points', []):
                    start = point.get('start', 0)
                    end = point.get('end', 0)
                    text = point.get('text', '').strip()
                    if text and start < end:
                        entities.append((start, end, 'SKILL'))
                        
                        # Add positive examples for sentence transformer
                        skill_examples.append({
                            'text': text,
                            'skill': text,
                            'similarity_score': 1.0
                        })
        
        if entities:
            # Remove overlapping entities before adding to training data
            non_overlapping_entities = remove_overlapping_entities(entities)
            if non_overlapping_entities:
                ner_training_data.append((content, {'entities': non_overlapping_entities}))
            
            # Add some negative examples for sentence transformer
            random_text = content[:100]  # Use first 100 chars as random text
            skill_examples.append({
                'text': random_text,
                'skill': text,
                'similarity_score': 0.1
            })
    
    return ner_training_data, skill_examples

def preprocess_text(text):
    """Basic text preprocessing."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def fine_tune_sentence_transformer(training_data_path, output_path):
    """Fine-tune a sentence transformer model on skill-text pairs."""
    # Load or initialize the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare training data
    train_examples = prepare_training_data(training_data_path)
    
    # Define the dataloader
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=16
    )
    
    # Define the loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path=output_path
    )
    
    return model

def main():
    try:
        print("Starting training process...")
        
        # Download required NLTK data
        print("\nDownloading NLTK resources...")
        nltk.download('punkt', quiet=False)
        nltk.download('stopwords', quiet=False)
        nltk.download('wordnet', quiet=False)
        
        training_file = 'Entity_Recognition_Fixed.json'
        output_dir = './fine_tuned_skill_model'
        
        print(f"\nUsing training data from: {os.path.abspath(training_file)}")
        print(f"Model will be saved to: {os.path.abspath(output_dir)}")
        
        if not os.path.exists(training_file):
            raise FileNotFoundError(f"Training file not found: {training_file}")
        
        # Fine-tune the sentence transformer model
        print("\nFine-tuning sentence transformer model...")
        model = fine_tune_sentence_transformer(
            training_data_path=training_file,
            output_path=output_dir
        )
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nPlease ensure that:")
        print("1. The training file exists and is in the correct format")
        print("2. You have enough disk space")
        print("3. You have a stable internet connection (for downloading models)")
        print("4. You have the required permissions to write to the output directory")
        return 1
    
    return 0
    
    print("Fine-tuned sentence transformer model saved to 'fine_tuned_skill_model'")

if __name__ == "__main__":
    main()
