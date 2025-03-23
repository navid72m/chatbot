#!/usr/bin/env python
# ollama_diagnostic.py - Script to diagnose and troubleshoot Ollama API issues

import requests
import sys
import json
import os

def print_header(text):
    """Print a header with the given text"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def check_ollama_running():
    """Check if Ollama server is running"""
    print_header("Checking if Ollama server is running")
    
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        print(f"✅ Ollama server is running! Response: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama server at http://localhost:11434")
        print("   Make sure Ollama is installed and running with 'ollama serve'")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {str(e)}")
        return False

def check_api_version():
    """Check the Ollama API version"""
    print_header("Checking Ollama API version")
    
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            version = response.json().get("version", "unknown")
            print(f"✅ Ollama API version: {version}")
            return version
        elif response.status_code == 404:
            print("❌ API version endpoint not found. This might be an older version of Ollama.")
            print("   Trying alternative method to determine version...")
            
            # Try other endpoints to guess the version
            try:
                response = requests.get("http://localhost:11434/api/tags")
                if response.status_code == 200:
                    print("✅ Found '/api/tags' endpoint - this appears to be Ollama 0.1.0 or newer")
                    return "0.1.0+"
                else:
                    print("❌ Could not access '/api/tags' endpoint")
            except:
                pass
            
            return "unknown"
        else:
            print(f"❌ Unexpected response code: {response.status_code}")
            return "unknown"
    except Exception as e:
        print(f"❌ Error checking API version: {str(e)}")
        return "unknown"

def list_models():
    """List available Ollama models"""
    print_header("Listing available Ollama models")
    
    endpoints = [
        "http://localhost:11434/api/tags",  # Current API
        "http://localhost:11434/api/models"  # Old API (pre-0.1.0)
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint)
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats based on endpoint
                if "models" in data:
                    models = data["models"]
                    print(f"✅ Found {len(models)} models using endpoint: {endpoint}")
                    for model in models:
                        if isinstance(model, dict):
                            name = model.get("name", "unknown")
                            size = model.get("size", "unknown")
                            print(f"   - {name} ({size})")
                        else:
                            print(f"   - {model}")
                    return models
                else:
                    print(f"✅ Response from {endpoint}: {json.dumps(data, indent=2)[:200]}...")
            else:
                print(f"❌ Failed to list models using {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ Error with {endpoint}: {str(e)}")
    
    print("❌ Could not list models from any known endpoint")
    return []

def test_generate_api():
    """Test the generate API endpoint"""
    print_header("Testing the generate API")
    
    # Test data
    data = {
        "model": "mistral",
        "prompt": "What is the capital of France?",
        "stream": False
    }
    
    # Try both known API endpoints
    endpoints = [
        "http://localhost:11434/api/generate",  # Current endpoint
        "http://localhost:11434/api/chat"       # Possible alternative endpoint
    ]
    
    for endpoint in endpoints:
        try:
            print(f"Testing endpoint: {endpoint}")
            print(f"Request data: {json.dumps(data, indent=2)}")
            
            response = requests.post(endpoint, json=data, timeout=10)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No response field found")
                print(f"✅ Success! Response: {response_text[:100]}...")
                return True
            elif response.status_code == 404:
                print(f"❌ Endpoint not found: {endpoint}")
            else:
                print(f"❌ Failed with status {response.status_code}: {response.text[:200]}")
        except Exception as e:
            print(f"❌ Error with {endpoint}: {str(e)}")
    
    return False

def suggest_fixes():
    """Suggest fixes based on diagnostic results"""
    print_header("Suggested fixes")
    
    print("Based on the diagnostic results, here are some suggestions:")
    
    print("1. Make sure your code is using the correct API endpoints:")
    print("   - For Ollama 0.1.0+: Use '/api/generate' for text generation")
    print("   - For older versions: Your endpoints may vary")
    
    print("\n2. For the latest Ollama version, update your llm_interface.py code:")
    print("""
    # Example code for llm_interface.py
    def query_ollama(query, context, model="mistral", temperature=0.7, quantization="4bit"):
        # Format model name with quantization
        model_name = model
        if quantization and quantization.lower() != "none":
            if quantization == "4bit":
                model_name = f"{model}:q4_0" 
            # ... other quantization options ...
        
        data = {
            "model": model_name,
            "prompt": f"Context: {context}\\n\\nQuestion: {query}",
            "temperature": temperature,
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=data)
        response.raise_for_status()
        result = response.json()
        return result["response"].strip()
    """)
    
    print("\n3. If you have an older version of Ollama, consider upgrading:")
    print("   - Visit https://ollama.ai to download the latest version")
    
    print("\n4. Check if you need to pull the model first:")
    print("   - Run 'ollama pull mistral' to ensure the model is available")
    
    print("\n5. For quantization, ensure you're using the correct format:")
    print("   - 4-bit: modelname:q4_0")
    print("   - 8-bit: modelname:q8_0")
    print("   - 1-bit: modelname:q1_1")

def main():
    """Main diagnostic function"""
    print_header("Ollama API Diagnostic Tool")
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("\nOllama server is not running. Please start it with 'ollama serve' and try again.")
        return
    
    # Check API version
    version = check_api_version()
    
    # List models
    list_models()
    
    # Test generate API
    test_generate_api()
    
    # Suggest fixes
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print(" Diagnostic complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()