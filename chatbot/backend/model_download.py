import os
import time
import json
import logging
import threading
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Router for model download endpoints
router = APIRouter()

# In-memory storage for download progress
# In a production app, this should be stored in a database or Redis
download_status = {}

# Base URL for Ollama API
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")

# Model information (size, description, etc.)
MODEL_INFO = {
    "deepseek-r1": {
        "name": "DeepSeek R1",
        "size": "3.4 GB",
        "description": "A high-performance model for reasoning tasks"
    },
    "mistral": {
        "name": "Mistral 7B",
        "size": "4.1 GB",
        "description": "Efficient open-source model with strong general capabilities"
    },
    "llama2": {
        "name": "Llama 2 13B",
        "size": "7.8 GB",
        "description": "Meta's powerful open-source LLM with extensive training"
    }
}

# Request models
class DownloadModelRequest(BaseModel):
    model: str
    quantization: Optional[str] = None

class ModelStatusRequest(BaseModel):
    model: str

# Get list of downloaded models
def get_downloaded_models() -> List[str]:
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name") for model in models]
        else:
            logger.error(f"Failed to list models: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return []

# Download model in a background thread
def download_model_task(model_name: str, download_id: str, quantization: Optional[str] = None):
    try:
        # Update status to downloading
        download_status[download_id] = {
            "model": model_name,
            "progress": 0,
            "status": "downloading",
            "error": None,
            "start_time": time.time()
        }
        
        # Simulate download progress
        # In a real implementation, you would call the Ollama API to pull the model
        # and update progress based on the actual download
        total_size_mb = 5000  # Example: 5GB model
        downloaded_mb = 0
        
        if model_name in MODEL_INFO:
            # Extract size from model info (format: "3.4 GB")
            size_str = MODEL_INFO[model_name].get("size", "5.0 GB")
            try:
                size_gb = float(size_str.split(" ")[0])
                total_size_mb = size_gb * 1024  # Convert to MB
            except ValueError:
                pass

        # Simulate download with progress updates
        download_chunks = 50  # Number of progress updates
        chunk_size = total_size_mb / download_chunks
        
        for i in range(download_chunks):
            # Simulate network fluctuations
            time.sleep(0.2 + (0.3 * (i % 3)))  # Varying sleep time
            downloaded_mb += chunk_size
            progress = min(100, (downloaded_mb / total_size_mb) * 100)
            
            download_status[download_id].update({
                "progress": progress,
                "last_update": time.time()
            })
            
            # Random chance of error for testing purposes
            if i == 30 and download_id.endswith("test-error"):
                download_status[download_id].update({
                    "status": "error",
                    "error": "Simulated download error"
                })
                return
        
        # Mark download as complete
        download_status[download_id].update({
            "progress": 100,
            "status": "completed",
            "end_time": time.time()
        })
        
        # In a real implementation, you'd verify the model is available
        logger.info(f"Model {model_name} download completed")
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {str(e)}")
        download_status[download_id].update({
            "status": "error",
            "error": str(e)
        })

# API Routes
@router.post("/download-model")
async def start_model_download(request: DownloadModelRequest):
    """Start downloading a model"""
    model_name = request.model
    
    # Check if model exists in available models list
    # In a real implementation, you'd check if the model ID is valid
    
    # Generate a unique ID for this download
    download_id = f"{model_name}-{uuid.uuid4()}"
    
    # Start download in a background thread
    thread = threading.Thread(
        target=download_model_task,
        args=(model_name, download_id, request.quantization)
    )
    thread.daemon = True
    thread.start()
    
    return {
        "model": model_name,
        "download_id": download_id,
        "status": "started"
    }

@router.get("/download-status")
async def get_download_status(model: str):
    """Get the status of a model download"""
    # Find the most recent download for this model
    model_downloads = [
        (download_id, status) for download_id, status in download_status.items()
        if status["model"] == model
    ]
    
    if not model_downloads:
        # Check if model is already downloaded
        downloaded_models = get_downloaded_models()
        if model in downloaded_models:
            return {
                "model": model,
                "progress": 100,
                "status": "completed",
                "already_downloaded": True
            }
        else:
            return {
                "model": model,
                "progress": 0,
                "status": "not_started"
            }
    
    # Sort by start time to get the most recent download
    sorted_downloads = sorted(
        model_downloads,
        key=lambda x: x[1].get("start_time", 0),
        reverse=True
    )
    
    download_id, status = sorted_downloads[0]
    
    return {
        "model": model,
        "download_id": download_id,
        "progress": status.get("progress", 0),
        "status": status.get("status", "unknown"),
        "error": status.get("error"),
        "start_time": status.get("start_time"),
        "end_time": status.get("end_time")
    }

@router.get("/models")
async def list_models():
    """List available and downloaded models"""
    downloaded_models = get_downloaded_models()
    
    # In a real implementation, you'd fetch the list of available models
    # from Ollama or Hugging Face API
    available_models = list(MODEL_INFO.keys())
    
    return {
        "models": available_models,
        "downloaded_models": downloaded_models,
        "model_info": MODEL_INFO
    }

# Function to add these routes to the main app
def add_model_download_routes(app):
    app.include_router(router, tags=["models"])