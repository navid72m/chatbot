#!/usr/bin/env python
# start_app.py - Script to start both the backend and frontend of the Document Chat application

import subprocess
import sys
import os
import time
import signal
import platform
import threading

# Track processes for cleanup
backend_process = None
frontend_process = None

def check_backend_running(url="http://127.0.0.1:8000"):
    """Check if the backend server is already running"""
    import requests
    try:
        response = requests.get(url, timeout=2)
        return response.status_code == 200
    except:
        return False

def find_python_command():
    """Find the correct Python command to use"""
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Using virtual environment Python:", sys.executable)
        return sys.executable

    # If not in a virtual environment, try common commands
    commands = ["python3", "python"]
    for cmd in commands:
        try:
            output = subprocess.check_output([cmd, "--version"], stderr=subprocess.STDOUT)
            print(f"Using Python: {output.decode().strip()}")
            return cmd
        except:
            continue
    return None

def start_backend(python_cmd):
    """Start the FastAPI backend server"""
    global backend_process
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the app.py file
    backend_dirs = [
        os.path.join(current_dir, "backend"),
        os.path.join(current_dir, "chatbot", "backend"),
    ]
    
    app_py_path = None
    for backend_dir in backend_dirs:
        potential_path = os.path.join(backend_dir, "app.py")
        if os.path.exists(potential_path):
            app_py_path = potential_path
            break
    
    if not app_py_path:
        print("Error: Could not find app.py in any of the expected directories.")
        return False
    
    print(f"Starting backend server from {app_py_path}...")
    
    # Start the backend process
    backend_dir = os.path.dirname(app_py_path)
    if platform.system() == "Windows":
        # Use different approach for Windows
        backend_process = subprocess.Popen(
            [python_cmd, "app.py"],
            cwd=backend_dir,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        backend_process = subprocess.Popen(
            [python_cmd, "app.py"],
            cwd=backend_dir
        )
    
    # Wait for the backend to start
    attempts = 0
    while attempts < 30:
        if check_backend_running():
            print("✅ Backend server is running!")
            return True
        print("Waiting for backend server to start...")
        time.sleep(1)
        attempts += 1
    
    print("❌ Failed to start backend server after 30 seconds.")
    return False

def start_frontend():
    """Start the Electron frontend"""
    global frontend_process
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the frontend directory
    frontend_dirs = [
        os.path.join(current_dir, "frontend"),
        os.path.join(current_dir, "chatbot", "frontend"),
    ]
    
    frontend_dir = None
    for potential_dir in frontend_dirs:
        if os.path.exists(os.path.join(potential_dir, "package.json")):
            frontend_dir = potential_dir
            break
    
    if not frontend_dir:
        print("Error: Could not find frontend directory.")
        return False
    
    print(f"Starting frontend from {frontend_dir}...")
    
    # Choose the right npm command based on the platform
    npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
    
    # Start the frontend process
    frontend_process = subprocess.Popen(
        [npm_cmd, "start"],
        cwd=frontend_dir
    )
    
    print("✅ Frontend started!")
    return True

def cleanup(signum=None, frame=None):
    """Cleanup function to terminate processes on exit"""
    print("\nCleaning up processes...")
    
    if backend_process:
        print("Stopping backend server...")
        if platform.system() == "Windows":
            backend_process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            backend_process.terminate()
    
    if frontend_process:
        print("Stopping frontend application...")
        frontend_process.terminate()
    
    print("Cleanup complete. Exiting.")
    sys.exit(0)

def check_required_packages():
    """Check if required packages are installed"""
    required_packages = [
        "fastapi", 
        "langchain", 
        "chromadb", 
        "langchain_chroma", 
        "sentence_transformers"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install the required packages:")
        print("pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required Python packages are installed.")
    return True

def main():
    """Main function to start the application"""
    print("Starting Document Chat Application...")
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Check required packages
    if not check_required_packages():
        return False
    
    # Check if backend is already running
    if check_backend_running():
        print("Backend is already running.")
    else:
        # Find Python command
        python_cmd = find_python_command()
        if not python_cmd:
            print("Error: Could not find Python. Make sure Python is installed and in your PATH.")
            return False
        
        # Start backend
        if not start_backend(python_cmd):
            print("Failed to start backend. Exiting.")
            return False
    
    # Start frontend
    if not start_frontend():
        print("Failed to start frontend. Exiting.")
        cleanup()
        return False
    
    print("\n🚀 Document Chat application is running!")
    print("⚠️  Press Ctrl+C to stop the application\n")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
    
    return True

if __name__ == "__main__":
    main()