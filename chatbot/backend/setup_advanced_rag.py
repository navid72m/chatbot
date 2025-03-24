#!/usr/bin/env python
# setup_advanced_rag.py - Setup script for Advanced RAG dependencies

import subprocess
import sys
import os
import logging
import time
import requests
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")

def install_dependencies():
    """Install Python dependencies from requirements.txt"""
    print_header("Installing Python dependencies")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Successfully installed Python dependencies")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def download_spacy_model():
    """Download spaCy language model"""
    print_header("Setting up spaCy language model")
    
    try:
        logger.info("Downloading spaCy English language model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
        logger.info("Successfully downloaded spaCy language model")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error downloading spaCy model: {e}")
        logger.info("Attempting to install spaCy first...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "spacy"])
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
            logger.info("Successfully downloaded spaCy language model")
            return True
        except:
            logger.error("Failed to download spaCy model")
            return False

def check_neo4j():
    """Check if Neo4j is installed and running"""
    print_header("Checking Neo4j installation")
    
    # Check if Neo4j is installed
    neo4j_installed = False
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["where", "neo4j"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            neo4j_installed = result.returncode == 0
        else:  # Unix-like systems
            result = subprocess.run(["which", "neo4j"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            neo4j_installed = result.returncode == 0
    except:
        neo4j_installed = False
    
    if neo4j_installed:
        logger.info("Neo4j is installed on this system")
    else:
        logger.warning("Neo4j does not appear to be installed on this system")
        logger.info("Please install Neo4j from https://neo4j.com/download/")
        logger.info("For Docker users, you can use: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j")
    
    # Check if Neo4j is running by attempting to connect
    neo4j_running = False
    try:
        response = requests.get("http://localhost:7474", timeout=2)
        neo4j_running = response.status_code == 200
    except:
        neo4j_running = False
    
    if neo4j_running:
        logger.info("Neo4j service appears to be running")
    else:
        logger.warning("Neo4j service does not appear to be running")
        logger.info("Please start Neo4j before using the knowledge graph features")
        if platform.system() == "Windows":
            logger.info("Start Neo4j using the Neo4j Desktop application or run: neo4j.bat console")
        else:
            logger.info("Start Neo4j using: neo4j start")
    
    return neo4j_installed and neo4j_running

def check_ollama():
    """Check if Ollama is installed and running"""
    print_header("Checking Ollama installation")
    
    # Check if Ollama is available
    ollama_installed = False
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["where", "ollama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ollama_installed = result.returncode == 0
        else:  # Unix-like systems
            result = subprocess.run(["which", "ollama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ollama_installed = result.returncode == 0
    except:
        ollama_installed = False
    
    if ollama_installed:
        logger.info("Ollama is installed on this system")
    else:
        logger.warning("Ollama does not appear to be installed on this system")
        logger.info("Please install Ollama from https://ollama.com/")
    
    # Check if Ollama is running
    ollama_running = False
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        ollama_running = True  # If no exception, it's responding
    except:
        ollama_running = False
    
    if ollama_running:
        logger.info("Ollama service appears to be running")
        
        # Check for required models
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "").split(":")[0] for model in models]
                
                if "mistral" in model_names:
                    logger.info("Mistral model is available")
                else:
                    logger.warning("Mistral model is not available")
                    logger.info("Please pull the model using: ollama pull mistral")
            else:
                logger.warning("Could not retrieve model list from Ollama")
        except Exception as e:
            logger.error(f"Error checking Ollama models: {e}")
    else:
        logger.warning("Ollama service does not appear to be running")
        logger.info("Please start Ollama using: ollama serve")
    
    return ollama_installed and ollama_running

def create_test_directories():
    """Create necessary directories"""
    print_header("Creating directories")
    
    directories = ["uploads", "chroma_db"]
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")

def main():
    """Main setup function"""
    print_header("Advanced RAG Setup")
    
    steps = [
        ("Installing Python dependencies", install_dependencies),
        ("Downloading spaCy language model", download_spacy_model),
        ("Checking Neo4j installation", check_neo4j),
        ("Checking Ollama installation", check_ollama),
        ("Creating required directories", create_test_directories)
    ]
    
    success_count = 0
    for step_name, step_func in steps:
        try:
            logger.info(f"Starting: {step_name}")
            result = step_func()
            if result:
                success_count += 1
                logger.info(f"Completed: {step_name}")
            else:
                logger.warning(f"Step completed with warnings: {step_name}")
        except Exception as e:
            logger.error(f"Error in {step_name}: {e}")
    
    print_header("Setup Complete")
    logger.info(f"Completed {success_count} out of {len(steps)} steps successfully")
    
    # Final instructions
    print("\nNext steps:")
    print("1. Ensure Neo4j is running (if using knowledge graph features)")
    print("2. Ensure Ollama is running with the mistral model pulled")
    print("3. Start the application with: python app_integration.py")
    print("\nFor more information on advanced RAG features, see the documentation.")

if __name__ == "__main__":
    main()