import os
import shutil
from pathlib import Path

def encrypt_resume_extractor():
    # Project directories
    base_dir = Path(__file__).parent
    src_dir = base_dir
    dist_dir = base_dir / 'dist'
    
    # Create dist directory if it doesn't exist
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to exclude from encryption
    exclude_files = {
        'encrypt.py', 
        'launch.py', 
        'launch.spec',
        'requirements.txt',
        'setup.py'
    }
    
    # Directories to exclude
    exclude_dirs = {
        'dist', 
        'build', 
        '.git', 
        '__pycache__', 
        'templates',
        'fine_tuned_skill_model',
        'fine_tuned_ner_model'
    }
    
    # Files to copy without encryption
    copy_files = {
        'requirements.txt',
        'launch.spec'
    }
    
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
    
    # Encrypt the files using PyArmor
    for py_file in py_files:
        rel_path = os.path.relpath(py_file, src_dir)
        target_dir = dist_dir / os.path.dirname(rel_path)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = f'pyarmor gen -O {target_dir} --recursive {py_file}'
        print(f"Encrypting: {rel_path}")
        os.system(cmd)
    
    # Copy template and model directories
    dirs_to_copy = ['templates', 'fine_tuned_skill_model', 'fine_tuned_ner_model']
    for dir_name in dirs_to_copy:
        src_path = src_dir / dir_name
        dst_path = dist_dir / dir_name
        if src_path.exists():
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
    
    # Copy non-Python files
    for item in os.listdir(src_dir):
        item_path = src_dir / item
        if (item in copy_files and 
            item_path.is_file() and 
            item not in exclude_dirs):
            shutil.copy2(item_path, dist_dir)
    
    # Create a launcher script
    create_launcher_script(dist_dir)
    
    print("\nEncryption complete!")
    print(f"Encrypted files are in: {dist_dir}")
    print("\nTo run your application, use: python dist/launch_encrypted.py")

def create_launcher_script(dist_dir):
    launcher_content = """import os
import sys
from pathlib import Path

def main():
    # Set up paths
    base_path = Path(__file__).parent
    os.environ['PYTHONPATH'] = str(base_path)
    
    # Import the main application
    from app_desktop import main as app_main
    
    # Run the application
    app_main()

if __name__ == "__main__":
    main()
"""
    launcher_path = dist_dir / "launch_encrypted.py"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)

if __name__ == "__main__":
    # Install pyarmor if not available
    try:
        import pyarmor
    except ImportError:
        print("Installing pyarmor...")
        os.system('pip install pyarmor')
    
    encrypt_resume_extractor()
