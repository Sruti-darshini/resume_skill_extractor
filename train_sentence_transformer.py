import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import random

def load_training_data(json_file):
    """Load and prepare training data for sentence transformer."""
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
    
    # Prepare training examples for sentence transformer
    skill_examples = []
    
    for resume in resumes:
        content = resume.get('content', '')
        annotations = resume.get('annotation', [])
        
        # Process annotations for sentence transformer training
        for annotation in annotations:
            if 'Skills' in annotation.get('label', []):
                for point in annotation.get('points', []):
                    text = point.get('text', '').strip()
                    if text:
                        # Add positive examples with context
                        start = point.get('start', 0)
                        end = point.get('end', 0)
                        # Get context around the skill (50 chars before and after)
                        context_start = max(0, start - 50)
                        context_end = min(len(content), end + 50)
                        context = content[context_start:context_end]
                        
                        # Add positive example
                        skill_examples.append(InputExample(
                            texts=[context, text],
                            label=1.0
                        ))
                        
                        # Add negative examples with random skills from other resumes
                        other_skills = [
                            p.get('text', '').strip()
                            for r in resumes
                            if r != resume
                            for a in r.get('annotation', [])
                            if 'Skills' in a.get('label', [])
                            for p in a.get('points', [])
                        ]
                        
                        # Add 2 negative examples for each positive
                        if other_skills:
                            for _ in range(2):
                                negative_skill = random.choice(other_skills)
                                while negative_skill == text:  # Make sure it's different
                                    negative_skill = random.choice(other_skills)
                                    
                                skill_examples.append(InputExample(
                                    texts=[context, negative_skill],
                                    label=0.0
                                ))
    
    print(f"Created {len(skill_examples)} training examples")
    return skill_examples

def fine_tune_sentence_transformer(train_examples):
    """Fine-tune a sentence transformer model on skill examples."""
    print("Loading base model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create dataloader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Fine-tune the model
    print("\nStarting model training...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=5,
              warmup_steps=100,
              output_path='./fine_tuned_skill_model',
              show_progress_bar=True)
    
    print("Training completed! Model saved to ./fine_tuned_skill_model")
    return model

def main():
    # Load training data
    print("Loading training data...")
    json_file = 'Entity_Recognition_Fixed.json'
    train_examples = load_training_data(json_file)
    
    # Fine-tune sentence transformer model
    print("\nFine-tuning sentence transformer model...")
    model = fine_tune_sentence_transformer(train_examples)

if __name__ == "__main__":
    main()
