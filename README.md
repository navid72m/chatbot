# Getting Started with Document Chat

This guide explains how to set up and run the Document Chat application with 4-bit quantization support for Ollama.

## Prerequisites

1. **Python 3.8+** with pip
2. **Node.js and npm**
3. **Ollama** installed and running

## Setting Up Your Environment

### Step 1: Clone the repository

```bash
git clone https://github.com/your-username/document-chat.git
cd document-chat
```

### Step 2: Create a virtual environment (recommended)

```bash
# On macOS/Linux
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
# Install Python dependencies
cd chatbot/backend
pip install -r requirements.txt
pip install langchain-chroma

# Install Node.js dependencies
cd ../frontend
npm install
```

## Running the Application

### Option 1: Using the startup scripts (recommended)

We've provided convenient startup scripts that will handle starting both the backend and frontend components in the correct order:

1. **On macOS/Linux**:
   ```bash
   # Make the script executable first
   chmod +x start-app.sh
   
   # Run the script
   ./start-app.sh
   ```

2. **On Windows**:
   ```
   # Double-click start-app.bat or run:
   start-app.bat
   ```

3. **Using Python directly**:
   ```bash
   python start_app.py
   ```

These scripts will:
- Detect and activate your Python virtual environment if available
- Check for required dependencies
- Start the FastAPI backend server
- Start the Electron frontend application
- Handle graceful shutdown of both components when you're done

### Option 2: Manual startup

If you prefer to start the components manually:

1. **Start the backend server**:
   ```bash
   cd chatbot/backend
   python app.py
   ```

2. **In a separate terminal, start the frontend**:
   ```bash
   cd chatbot/frontend
   npm start
   ```

## Using 4-bit Quantization

The application now supports different quantization levels for Ollama models:

1. From the dropdown menu in the sidebar, select your preferred quantization level:
   - **4-bit** (recommended): Best balance of speed and quality
   - **8-bit**: Better quality but uses more memory
   - **1-bit**: Fastest but lower quality
   - **None**: No quantization (highest quality but more memory usage)

2. Ollama will automatically use the selected quantization level for your model.

## Troubleshooting

### Connection Refused Error

If you see a "Connection Refused" error in the app or logs:

1. Make sure the backend server is running on port 8000
2. Check if another application is using port 8000
3. Use the startup scripts which ensure the backend starts before the frontend

### Missing Dependencies

If you get errors about missing Python packages:

```bash
pip install langchain-chroma fastapi uvicorn chromadb sentence-transformers
```

### Ollama Not Found

If the app can't connect to Ollama:

1. Make sure Ollama is installed and running: `ollama serve`
2. Pull your desired model if it's not already available: `ollama pull mistral`

## Next Steps

Once the application is running:

1. Upload documents using the "Upload Document" button
2. Select your preferred model and quantization level
3. Start chatting with your documents!