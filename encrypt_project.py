import os
import shutil
from pathlib import Path

def encrypt_project():
    # Project directories
    base_dir = Path(__file__).parent
    src_dir = base_dir
    dist_dir = base_dir / 'dist'
    
    # Create dist directory if it doesn't exist
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to exclude from encryption
    exclude_files = {'encrypt_project.py', 'setup.py', 'launch.py', 'launch.spec'}
    exclude_dirs = {'dist', 'build', '.git', '__pycache__', 'templates', 'nltk_data', 
                   'fine_tuned_ner_model', 'fine_tuned_skill_model'}
    
    # Find all Python files to encrypt
    py_files = []
    for root, dirs, files in os.walk(src_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            if file.endswith('.py') and file not in exclude_files:
                py_files.append(str(Path(root) / file))
    
    if not py_files:
        print("No Python files found to encrypt.")
        return
    
    print(f"Found {len(py_files)} Python files to encrypt.")
    
    # Encrypt the files
    for py_file in py_files:
        rel_path = os.path.relpath(py_file, src_dir)
        target_dir = os.path.dirname(rel_path)
        os.makedirs(dist_dir / target_dir, exist_ok=True)
        
        cmd = f'pyarmor gen -O {dist_dir / target_dir} --recursive {py_file}'
        print(f"Encrypting: {rel_path}")
        os.system(cmd)
    
    print("\nEncryption complete!")
    print(f"Encrypted files are in: {dist_dir}")
    print("\nTo run your application, use the launcher script in the dist directory.")

if __name__ == "__main__":
    encrypt_project()
