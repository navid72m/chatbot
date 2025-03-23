#!/usr/bin/env python
# update_dependencies.py - Script to update dependencies for the Document Chat application

import subprocess
import sys
import os

def update_dependencies():
    """Update the required dependencies for the Document Chat application"""
    print("Updating dependencies for Document Chat application...")
    
    # Get the current directory (assuming the script is being run from the backend directory)
    current_dir = os.getcwd()
    
    # Path to requirements.txt - assuming it's in the current directory
    requirements_path = os.path.join(current_dir, 'requirements.txt')
    
    # Check if the requirements file exists
    if not os.path.exists(requirements_path):
        print(f"Error: Could not find requirements.txt at {requirements_path}")
        print("Please run this script from the directory containing requirements.txt")
        return False
    
    try:
        # Install/update dependencies
        print("Installing/updating Python dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "-r", requirements_path])
        
        # Specifically ensure langchain-chroma is installed
        print("Ensuring langchain-chroma is properly installed...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "langchain-chroma"])
        
        print("\nDependencies updated successfully!")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error updating dependencies: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    update_dependencies()