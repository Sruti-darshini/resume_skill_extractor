import os
import sys
import shutil
from pathlib import Path

def setup_environment():
    # Set up paths
    base_dir = Path(__file__).parent
    dist_dir = base_dir / 'dist'
    
    # Create dist directory if it doesn't exist
    dist_dir.mkdir(exist_ok=True)
    
    # Add dist directory to Python path
    sys.path.insert(0, str(dist_dir))
    
    # Ensure required directories exist
    for dir_name in ['templates', 'static']:
        src = base_dir / dir_name
        dst = dist_dir / dir_name
        
        # Create empty directories if they don't exist
        if not src.exists():
            src.mkdir(exist_ok=True)
        
        # Copy directories if they exist and destination doesn't
        if src.exists() and not dst.exists():
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
    
    # Copy other required files
    for file_name in ['.env']:
        src = base_dir / file_name
        dst = dist_dir / file_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(dist_dir)
    
    return base_dir, dist_dir

def main():
    try:
        base_dir, dist_dir = setup_environment()
        
        # Change working directory to dist to properly load encrypted files
        os.chdir(dist_dir)
        
        # Import PyArmor runtime first
        try:
            import pyarmor_runtime_000000
        except ImportError:
            print("Error: PyArmor runtime not found. Make sure to run the encryption script first.")
            sys.exit(1)
            
        # Import required modules
        import sys
        from PyQt5.QtWidgets import QApplication
        
        # Import the desktop application from the encrypted files
        print("Starting encrypted desktop application...")
        from app_desktop import ResumeSkillExtractorApp
        
        # Initialize and run the application
        app = QApplication(sys.argv)
        window = ResumeSkillExtractorApp()
        window.show()
        sys.exit(app.exec_())
        
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("\nPlease install the required dependencies:")
        print("pip install PyQt5")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting steps:")
        print("1. Make sure you've run the encryption script first")
        print("2. Check if all required files are in the dist directory")
        print("3. Verify that PyQt5 is installed")
        print("4. Check if all dependencies are installed")
        sys.exit(1)

if __name__ == "__main__":
    main()
