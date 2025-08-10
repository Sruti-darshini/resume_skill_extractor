import sys
import os

def main():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)

    app_path = os.path.join(base_path, "app_desktop.py")

    # Run app_desktop.py directly in current process
    with open(app_path, 'rb') as f:
        code = compile(f.read(), app_path, 'exec')
        exec(code, {'__name__': '__main__'})

if __name__ == "__main__":
    main()
