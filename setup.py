import subprocess
import sys

def install_spacy_model():
    print("Installing spaCy model...")
    try:
        import spacy
        spacy.cli.download("en_core_web_sm")
        print("spaCy model installed successfully!")
    except Exception as e:
        print(f"Error installing spaCy model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    install_spacy_model()
