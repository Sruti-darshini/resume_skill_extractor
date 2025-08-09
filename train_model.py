import json
import spacy
from spacy.training import Example
import numpy as np
import os
import random
from typing import List, Dict, Set
from collections import defaultdict
import re
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def remove_overlapping_entities(entities):
    """Remove overlapping entities by keeping the longest one."""
    if not entities:
        return []
    
    # Sort by start position and then by length (longest first)
    sorted_entities = sorted(entities, key=lambda x: (x[0], -len(x[2])))
    
    # Keep track of non-overlapping entities
    result = []
    last_end = -1
    
    for start, end, label in sorted_entities:
        if start >= last_end:
            result.append((start, end, label))
            last_end = end
    
    return result

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

def fine_tune_spacy_ner(train_data):
    # Load pre-trained model
    nlp = spacy.load('en_core_web_lg')
    
    # Add NER pipe if not exists
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')
    
    # Add skill labels
    for _, annotations in train_data:
        for ent in annotations.get('entities', []):
            ner.add_label(ent[2])
    
    # Disable other pipes during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        
        for itn in range(30):  # 30 iterations
            random.shuffle(train_data)
            losses = {}
            
            for text, annotations in train_data:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses)
            
            print(f"Iteration {itn}, Losses: {losses}")
    
    return nlp

def fine_tune_sentence_transformer(train_examples):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Convert skill examples to sentence transformer format
    train_samples = []
    all_skills = set()
    
    # First pass: collect all unique skills
    for example in train_examples:
        if example['skill']:
            all_skills.add(example['skill'])
    
    all_skills = list(all_skills)
    
    # Second pass: create training samples
    for example in train_examples:
        if example['text'] and example['skill']:
            # Create positive example
            train_samples.append(InputExample(
                texts=[example['text'], example['skill']],
                label=float(example['similarity_score'])))
            
            # Create negative example with a random different skill
            if len(all_skills) > 1:  # Only if we have other skills to choose from
                negative_skill = random.choice(all_skills)
                while negative_skill == example['skill']:
                    negative_skill = random.choice(all_skills)
                
                train_samples.append(InputExample(
                    texts=[example['text'], negative_skill],
                    label=0.0))
    
    print(f"Created {len(train_samples)} training examples")
    
    # Create dataloader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Fine-tune the model
    print("Starting model training...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=5,
              warmup_steps=100,
              output_path='./fine_tuned_skill_model')
    
    print("Training completed!")
    return model

def main():
    # Load and prepare training data
    json_file = 'Entity_Recognition_Fixed.json'
    ner_training_data, skill_examples = load_training_data(json_file)
    
    print(f"Loaded {len(ner_training_data)} resumes for NER training")
    print(f"Generated {len(skill_examples)} examples for sentence transformer training")
    
    # Fine-tune both models
    print("\nFine-tuning spaCy NER model...")
    nlp = fine_tune_spacy_ner(ner_training_data)
    nlp.to_disk('./fine_tuned_ner_model')
    print("Saved fine-tuned NER model to ./fine_tuned_ner_model")
    
    print("\nFine-tuning sentence transformer model...")
    model = fine_tune_sentence_transformer(skill_examples)
    print("Saved fine-tuned sentence transformer model to ./fine_tuned_skill_model")

if __name__ == "__main__":
    main()
