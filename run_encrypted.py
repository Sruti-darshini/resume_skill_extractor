#!/usr/bin/env python3
"""
Launcher for the encrypted application.
This script should be placed in the root directory of your project.
"""
import os
import sys
import shutil
from pathlib import Path

def main():
    try:
        # Set up paths
        base_dir = Path(__file__).parent
        dist_dir = base_dir / 'dist'
        
        # Check if dist directory exists
        if not dist_dir.exists():
            print("Error: 'dist' directory not found. Please run the encryption script first.")
            sys.exit(1)
            
        # Check if PyArmor runtime exists
        if not (dist_dir / 'pyarmor_runtime_000000').exists():
            print("Error: PyArmor runtime not found. Please run the encryption script first.")
            sys.exit(1)
        
        # Add dist directory to Python path
        sys.path.insert(0, str(dist_dir))
        
        # Import PyArmor runtime
        try:
            import pyarmor_runtime_000000
        except ImportError as e:
            print(f"Error importing PyArmor runtime: {e}")
            print("Make sure you've run the encryption script first.")
            sys.exit(1)
            
        # Import and run the desktop application
        try:
            from PyQt5.QtWidgets import QApplication
            from app_desktop import ResumeSkillExtractorApp
            
            print("Starting encrypted desktop application...")
            app = QApplication(sys.argv)
            window = ResumeSkillExtractorApp()
            window.show()
            sys.exit(app.exec_())
            
        except ImportError as e:
            print(f"Error importing application: {e}")
            print("Make sure all dependencies are installed.")
            print("You may need to run: pip install PyQt5")
            sys.exit(1)
            
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
