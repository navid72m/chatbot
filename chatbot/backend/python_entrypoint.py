# python_entrypoint.py
import os
import sys
import uvicorn
import subprocess

# Add the current directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Function to check and install dependencies
def ensure_dependencies():
    try:
        import fastapi
        import langchain
        # More import checks here...
    except ImportError:
        print("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                              os.path.join(current_dir, "requirements.txt")])

# Ensure dependencies are installed
ensure_dependencies()

# Now import our app
from app import app

if __name__ == "__main__":
    # Get port from command line args or default to 8000
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    # Start the server
    uvicorn.run(app, host="127.0.0.1", port=port)