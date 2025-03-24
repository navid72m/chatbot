#!/bin/bash
# start_advanced_rag.sh - Script to start the complete Advanced RAG system
# This script starts Neo4j, Ollama, the backend server, and the frontend

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# Configuration
BACKEND_DIR="./backend"
FRONTEND_DIR="./frontend"
BACKEND_PORT=8000
FRONTEND_PORT=3000
NEO4J_PORT=7474
NEO4J_BOLT_PORT=7687
OLLAMA_PORT=11434
NEO4J_CONTAINER_NAME="advanced-rag-neo4j"
LOG_DIR="./logs"

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

# Function to print section headers
print_header() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${RESET}\n"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
is_port_in_use() {
    if command_exists lsof; then
        lsof -i:"$1" >/dev/null 2>&1
        return $?
    elif command_exists netstat; then
        netstat -tuln | grep ":$1 " >/dev/null 2>&1
        return $?
    else
        echo -e "${YELLOW}Warning: Cannot check if port $1 is in use. Please ensure it's available.${RESET}"
        return 1
    fi
}

# Function to check if a process is running
is_process_running() {
    pgrep -f "$1" >/dev/null 2>&1
}

# Function to tail logs with timestamp
tail_with_timestamp() {
    local log_file="$1"
    local process_name="$2"
    tail -f "$log_file" | while read line; do
        echo -e "[$(date '+%H:%M:%S')] ${BOLD}${process_name}${RESET}: $line"
    done
}

# Function to wait for a service to be available with detailed progress
wait_for_service_with_logs() {
    local service_name="$1"
    local url="$2"
    local max_attempts="$3"
    local sleep_time="$4"
    local log_file="$5"
    
    echo -e "Waiting for ${service_name} to be available (this may take a while)..."
    
    local attempt=1
    
    # If we have a log file, show logs in background
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        tail_with_timestamp "$log_file" "$service_name" &
        local tail_pid=$!
        # Make sure to kill the tail process when we're done
        trap "kill $tail_pid 2>/dev/null" EXIT
    fi
    
    while true; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e "\n${GREEN}${service_name} is now available! (Attempt $attempt/$max_attempts)${RESET}"
            if [ -n "$log_file" ] && [ -f "$log_file" ]; then
                kill $tail_pid 2>/dev/null
                trap - EXIT
            fi
            return 0
        fi
        
        if [ $attempt -ge $max_attempts ]; then
            echo -e "\n${RED}${service_name} did not become available after $max_attempts attempts.${RESET}"
            if [ -n "$log_file" ] && [ -f "$log_file" ]; then
                kill $tail_pid 2>/dev/null
                trap - EXIT
            fi
            return 1
        fi
        
        echo -ne "\rAttempt $attempt/$max_attempts: ${service_name} is not yet available. Waiting ${sleep_time}s..."
        sleep $sleep_time
        attempt=$((attempt + 1))
    done
}

# Check for required commands
print_header "Checking dependencies"

MISSING_DEPS=0

# Check for Python
if ! command_exists python && ! command_exists python3; then
    echo -e "${RED}Python is not installed. Please install Python 3.9 or higher.${RESET}"
    MISSING_DEPS=1
else
    PYTHON_CMD=$(command_exists python3 && echo "python3" || echo "python")
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}Python $PYTHON_VERSION is installed.${RESET}"
fi

# Check for pip
if ! command_exists pip && ! command_exists pip3; then
    echo -e "${RED}pip is not installed. Please install pip.${RESET}"
    MISSING_DEPS=1
else
    PIP_CMD=$(command_exists pip3 && echo "pip3" || echo "pip")
    echo -e "${GREEN}pip is installed.${RESET}"
fi

# Check for Node.js
if ! command_exists node; then
    echo -e "${RED}Node.js is not installed. Please install Node.js.${RESET}"
    MISSING_DEPS=1
else
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}Node.js $NODE_VERSION is installed.${RESET}"
fi

# Check for npm
if ! command_exists npm; then
    echo -e "${RED}npm is not installed. Please install npm.${RESET}"
    MISSING_DEPS=1
else
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}npm $NPM_VERSION is installed.${RESET}"
fi

# Check for Docker (optional but recommended for Neo4j)
if ! command_exists docker; then
    echo -e "${YELLOW}Warning: Docker is not installed. It's recommended for running Neo4j.${RESET}"
    echo -e "${YELLOW}If you have Neo4j installed directly, you can ignore this warning.${RESET}"
else
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}$DOCKER_VERSION is installed.${RESET}"
fi

# Check for Ollama
if ! command_exists ollama; then
    echo -e "${RED}Ollama is not installed. Please install Ollama from https://ollama.ai${RESET}"
    MISSING_DEPS=1
else
    echo -e "${GREEN}Ollama is installed.${RESET}"
fi

# Exit if any required dependency is missing
if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "\n${RED}Please install the missing dependencies and try again.${RESET}"
    exit 1
fi

# Check port availability
print_header "Checking port availability"

PORT_CONFLICT=0

if is_port_in_use $BACKEND_PORT; then
    echo -e "${RED}Port $BACKEND_PORT is already in use. Backend server may not start correctly.${RESET}"
    PORT_CONFLICT=1
else
    echo -e "${GREEN}Port $BACKEND_PORT is available for the backend server.${RESET}"
fi

if is_port_in_use $FRONTEND_PORT; then
    echo -e "${RED}Port $FRONTEND_PORT is already in use. Frontend server may not start correctly.${RESET}"
    PORT_CONFLICT=1
else
    echo -e "${GREEN}Port $FRONTEND_PORT is available for the frontend server.${RESET}"
fi

if is_port_in_use $NEO4J_PORT; then
    echo -e "${GREEN}Port $NEO4J_PORT is in use. Neo4j may already be running.${RESET}"
    NEO4J_RUNNING=true
else
    echo -e "${YELLOW}Port $NEO4J_PORT is not in use. Will attempt to start Neo4j.${RESET}"
    NEO4J_RUNNING=false
fi

if is_port_in_use $OLLAMA_PORT; then
    echo -e "${GREEN}Port $OLLAMA_PORT is in use. Ollama may already be running.${RESET}"
    OLLAMA_RUNNING=true
else
    echo -e "${YELLOW}Port $OLLAMA_PORT is not in use. Will attempt to start Ollama.${RESET}"
    OLLAMA_RUNNING=false
fi

if [ $PORT_CONFLICT -eq 1 ]; then
    echo -e "\n${YELLOW}Warning: Some ports are in conflict. The system may not start correctly.${RESET}"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check directory structure
print_header "Checking directory structure"

if [ ! -d "$BACKEND_DIR" ]; then
    echo -e "${RED}Backend directory '$BACKEND_DIR' does not exist.${RESET}"
    read -p "Enter the correct backend directory path: " BACKEND_DIR
    if [ ! -d "$BACKEND_DIR" ]; then
        echo -e "${RED}Directory '$BACKEND_DIR' does not exist. Exiting.${RESET}"
        exit 1
    fi
else
    echo -e "${GREEN}Backend directory '$BACKEND_DIR' exists.${RESET}"
fi

if [ ! -d "$FRONTEND_DIR" ]; then
    echo -e "${RED}Frontend directory '$FRONTEND_DIR' does not exist.${RESET}"
    read -p "Enter the correct frontend directory path: " FRONTEND_DIR
    if [ ! -d "$FRONTEND_DIR" ]; then
        echo -e "${RED}Directory '$FRONTEND_DIR' does not exist. Exiting.${RESET}"
        exit 1
    fi
else
    echo -e "${GREEN}Frontend directory '$FRONTEND_DIR' exists.${RESET}"
fi

# Start Neo4j (if not already running)
print_header "Starting Neo4j"

NEO4J_LOG_FILE="$LOG_DIR/neo4j.log"

if [ "$NEO4J_RUNNING" = false ]; then
    if command_exists docker; then
        echo "Starting Neo4j using Docker..."
        
        # Check if Neo4j container exists already
        if docker ps -a --format "{{.Names}}" | grep -q "$NEO4J_CONTAINER_NAME"; then
            echo "Neo4j container exists. Starting it..."
            docker start "$NEO4J_CONTAINER_NAME" > "$NEO4J_LOG_FILE" 2>&1
        else
            echo "Creating and starting new Neo4j container..."
            docker run -d --name "$NEO4J_CONTAINER_NAME" \
                -p $NEO4J_PORT:7474 -p $NEO4J_BOLT_PORT:7687 \
                -e NEO4J_AUTH=neo4j/password \
                -v neo4j-data:/data \
                neo4j > "$NEO4J_LOG_FILE" 2>&1
        fi
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Neo4j Docker container started.${RESET}"
            echo "Capturing Neo4j logs..."
            
            # Capture container logs in the background
            docker logs -f "$NEO4J_CONTAINER_NAME" > "$NEO4J_LOG_FILE" 2>&1 &
            NEO4J_LOG_PID=$!
            
            # Save PID to file for later cleanup
            echo $NEO4J_LOG_PID > "$LOG_DIR/neo4j_log.pid"
            
            # Wait for Neo4j to become available with log display
            echo -e "${YELLOW}Neo4j is starting up. This may take 1-2 minutes...${RESET}"
            echo -e "${YELLOW}Showing log output to track progress:${RESET}"
            
            # Wait for Neo4j to become available
            wait_for_service_with_logs "Neo4j" "http://localhost:$NEO4J_PORT" 60 5 "$NEO4J_LOG_FILE"
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}Neo4j is ready!${RESET}"
            else
                echo -e "${RED}Neo4j did not start properly. Please check the logs at $NEO4J_LOG_FILE${RESET}"
                echo -e "${YELLOW}You can try to manually check Neo4j status with: docker logs $NEO4J_CONTAINER_NAME${RESET}"
                read -p "Do you want to continue anyway? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
        else
            echo -e "${RED}Failed to start Neo4j Docker container.${RESET}"
            echo -e "Error log:\n$(cat $NEO4J_LOG_FILE)"
            echo -e "${YELLOW}If you have Neo4j installed directly, please start it manually.${RESET}"
            read -p "Press Enter to continue after starting Neo4j..." -r
        fi
    else
        echo -e "${YELLOW}Docker is not installed. If you have Neo4j installed directly, please start it manually.${RESET}"
        read -p "Press Enter to continue after starting Neo4j..." -r
    fi
else
    echo -e "${GREEN}Neo4j appears to be already running on port $NEO4J_PORT.${RESET}"
fi

# Start Ollama (if not already running)
print_header "Starting Ollama"

OLLAMA_LOG_FILE="$LOG_DIR/ollama.log"

if [ "$OLLAMA_RUNNING" = false ]; then
    echo "Starting Ollama server..."
    # Start Ollama in the background
    ollama serve > "$OLLAMA_LOG_FILE" 2>&1 &
    OLLAMA_PID=$!
    
    # Save PID to file for later cleanup
    echo $OLLAMA_PID > "$LOG_DIR/ollama.pid"
    
    # Wait for Ollama to become available with log display
    wait_for_service_with_logs "Ollama" "http://localhost:$OLLAMA_PORT" 30 1 "$OLLAMA_LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Ollama started successfully with PID $OLLAMA_PID.${RESET}"
    else
        echo -e "${RED}Failed to start Ollama.${RESET}"
        echo -e "Error log:\n$(cat $OLLAMA_LOG_FILE)"
        kill $OLLAMA_PID 2>/dev/null
        read -p "Please start Ollama manually and press Enter to continue..." -r
    fi
else
    echo -e "${GREEN}Ollama appears to be already running on port $OLLAMA_PORT.${RESET}"
fi

# Check if the required model is available
print_header "Checking Ollama models"

MODELS_LOG_FILE="$LOG_DIR/models.log"

echo "Checking if mistral model is available..."
if ! curl -s http://localhost:$OLLAMA_PORT/api/tags | grep -q "mistral"; then
    echo -e "${YELLOW}Mistral model not found. Pulling the model (this may take several minutes)...${RESET}"
    echo -e "${YELLOW}Showing pull progress:${RESET}"
    
    # Pull the model and show progress
    ollama pull mistral | tee "$MODELS_LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully pulled mistral model.${RESET}"
    else
        echo -e "${RED}Failed to pull mistral model. The system may not work correctly.${RESET}"
        echo -e "Error log:\n$(cat $MODELS_LOG_FILE)"
    fi
else
    echo -e "${GREEN}Mistral model is available.${RESET}"
fi

# Start Backend Server
print_header "Starting Backend Server"

BACKEND_LOG_FILE="$LOG_DIR/backend.log"

cd "$BACKEND_DIR" || exit 1

# Check if Python virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating Python virtual environment..."
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Failed to activate virtual environment. Continuing without it.${RESET}"
    else
        echo -e "${GREEN}Virtual environment activated.${RESET}"
    fi
fi

# Install backend dependencies if needed
if [ ! -f ".dependencies_installed" ]; then
    echo "Installing backend dependencies..."
    $PIP_CMD install -r requirements_updated.txt | tee -a "$BACKEND_LOG_FILE"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Backend dependencies installed successfully.${RESET}"
        touch .dependencies_installed
    else
        echo -e "${RED}Failed to install backend dependencies.${RESET}"
        echo -e "Error log:\n$(tail -n 20 $BACKEND_LOG_FILE)"
        exit 1
    fi
else
    echo -e "${GREEN}Backend dependencies already installed.${RESET}"
fi

# Install spaCy model if needed
if ! $PYTHON_CMD -c "import spacy; spacy.load('en_core_web_md')" 2>/dev/null; then
    echo "Installing spaCy language model..."
    $PYTHON_CMD -m spacy download en_core_web_md | tee -a "$BACKEND_LOG_FILE"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install spaCy language model.${RESET}"
        echo -e "Error log:\n$(tail -n 20 $BACKEND_LOG_FILE)"
        exit 1
    fi
fi

# Start the backend server
echo "Starting backend server on port $BACKEND_PORT..."
$PYTHON_CMD app_integration.py > "$BACKEND_LOG_FILE" 2>&1 &
BACKEND_PID=$!

# Save backend PID to file for later cleanup
echo $BACKEND_PID > "$LOG_DIR/backend.pid"

# Wait for backend to become available with log display
wait_for_service_with_logs "Backend server" "http://localhost:$BACKEND_PORT" 30 1 "$BACKEND_LOG_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Backend server started successfully with PID $BACKEND_PID.${RESET}"
else
    echo -e "${RED}Backend server did not start correctly.${RESET}"
    echo -e "Last 20 lines of log:\n$(tail -n 20 $BACKEND_LOG_FILE)"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start Frontend Server
print_header "Starting Frontend Server"

FRONTEND_LOG_FILE="$LOG_DIR/frontend.log"

cd "$FRONTEND_DIR" || exit 1

# Install frontend dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install | tee "$FRONTEND_LOG_FILE"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install frontend dependencies.${RESET}"
        echo -e "Error log:\n$(tail -n 20 $FRONTEND_LOG_FILE)"
        exit 1
    fi
    
    # Make sure axios is installed
    echo "Installing axios..."
    npm install axios --save | tee -a "$FRONTEND_LOG_FILE"
fi

# Check if the Advanced RAG components exist
if [ ! -f "src/frontend_integration.js" ] || [ ! -f "src/components/AdvancedRAGComponent.jsx" ]; then
    echo -e "${YELLOW}Advanced RAG components not found in the frontend directory.${RESET}"
    echo -e "${YELLOW}Please make sure to copy the following files:${RESET}"
    echo -e "${YELLOW}  - frontend_integration.js to src/${RESET}"
    echo -e "${YELLOW}  - AdvancedRAGComponent.jsx to src/components/${RESET}"
    
    # Create directories if they don't exist
    mkdir -p src/components
    
    read -p "Would you like to create these files now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Ask for file paths
        read -p "Enter path to frontend_integration.js: " INTEGRATION_PATH
        read -p "Enter path to AdvancedRAGComponent.jsx: " COMPONENT_PATH
        
        if [ -f "$INTEGRATION_PATH" ]; then
            cp "$INTEGRATION_PATH" src/frontend_integration.js
            echo -e "${GREEN}Copied frontend_integration.js${RESET}"
        else
            echo -e "${RED}File not found: $INTEGRATION_PATH${RESET}"
        fi
        
        if [ -f "$COMPONENT_PATH" ]; then
            cp "$COMPONENT_PATH" src/components/AdvancedRAGComponent.jsx
            echo -e "${GREEN}Copied AdvancedRAGComponent.jsx${RESET}"
        else
            echo -e "${RED}File not found: $COMPONENT_PATH${RESET}"
        fi
    fi
    
    read -p "Press Enter to continue (or Ctrl+C to cancel)..." -r
fi

# Start the frontend server
echo "Starting frontend server on port $FRONTEND_PORT..."
npm start > "$FRONTEND_LOG_FILE" 2>&1 &
FRONTEND_PID=$!

# Save frontend PID to file for later cleanup
echo $FRONTEND_PID > "$LOG_DIR/frontend.pid"

# Wait for frontend to become available with log display
wait_for_service_with_logs "Frontend server" "http://localhost:$FRONTEND_PORT" 60 2 "$FRONTEND_LOG_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Frontend server started successfully with PID $FRONTEND_PID.${RESET}"
else
    echo -e "${RED}Frontend server did not start correctly.${RESET}"
    echo -e "Last 20 lines of log:\n$(tail -n 20 $FRONTEND_LOG_FILE)"
    kill $FRONTEND_PID 2>/dev/null
    exit 1
fi

# Final instructions
print_header "System Started"

echo -e "${GREEN}Advanced RAG system is now running!${RESET}"
echo -e "\n${BOLD}Access the application:${RESET}"
echo -e "  Frontend: ${BLUE}http://localhost:$FRONTEND_PORT${RESET}"
echo -e "  Backend API: ${BLUE}http://localhost:$BACKEND_PORT${RESET}"
echo -e "  Neo4j Browser: ${BLUE}http://localhost:$NEO4J_PORT${RESET}"

echo -e "\n${BOLD}Log files:${RESET}"
echo -e "  Backend: ${BLUE}$LOG_DIR/backend.log${RESET}"
echo -e "  Frontend: ${BLUE}$LOG_DIR/frontend.log${RESET}"
echo -e "  Neo4j: ${BLUE}$LOG_DIR/neo4j.log${RESET}"
echo -e "  Ollama: ${BLUE}$LOG_DIR/ollama.log${RESET}"

echo -e "\n${BOLD}To view logs in real-time:${RESET}"
echo -e "  Backend: ${BLUE}tail -f $LOG_DIR/backend.log${RESET}"
echo -e "  Frontend: ${BLUE}tail -f $LOG_DIR/frontend.log${RESET}"
echo -e "  Neo4j: ${BLUE}tail -f $LOG_DIR/neo4j.log${RESET}"
echo -e "  Ollama: ${BLUE}tail -f $LOG_DIR/ollama.log${RESET}"

echo -e "\n${BOLD}To stop the system:${RESET}"
echo -e "  Run ${BLUE}./stop_advanced_rag.sh${RESET} or press Ctrl+C"

# Create stop script
cat > stop_advanced_rag.sh << 'EOF'
#!/bin/bash
# stop_advanced_rag.sh - Script to stop the Advanced RAG system

LOG_DIR="./logs"
NEO4J_CONTAINER_NAME="advanced-rag-neo4j"

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

echo -e "${BOLD}Stopping Advanced RAG system...${RESET}"

# Function to stop a process with PID file
stop_process() {
    local pid_file="$1"
    local process_name="$2"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        echo -e "Stopping $process_name (PID: $pid)..."
        kill $pid 2>/dev/null
        
        # Wait for process to terminate
        for i in {1..5}; do
            if ! ps -p $pid > /dev/null 2>&1; then
                break
            fi
            echo "Waiting for $process_name to terminate..."
            sleep 1
        done
        
        # Force kill if still running
        if ps -p $pid > /dev/null 2>&1; then
            echo -e "${YELLOW}$process_name still running, forcing termination...${RESET}"
            kill -9 $pid 2>/dev/null
        fi
        
        rm "$pid_file"
        echo -e "${GREEN}$process_name stopped.${RESET}"
    else
        echo -e "${YELLOW}No PID file found for $process_name.${RESET}"
    fi
}

# Stop Neo4j log process
if [ -f "$LOG_DIR/neo4j_log.pid" ]; then
    stop_process "$LOG_DIR/neo4j_log.pid" "Neo4j log process"
fi

# Stop frontend
stop_process "$LOG_DIR/frontend.pid" "Frontend server"

# Stop backend
stop_process "$LOG_DIR/backend.pid" "Backend server" 

# Stop Ollama
stop_process "$LOG_DIR/ollama.pid" "Ollama server"

# Stop Neo4j Docker container
if command -v docker >/dev/null 2>&1; then
    if docker ps -q --filter name="$NEO4J_CONTAINER_NAME" | grep -q .; then
        echo -e "Stopping Neo4j Docker container..."
        docker stop "$NEO4J_CONTAINER_NAME" > /dev/null
        echo -e "${GREEN}Neo4j Docker container stopped.${RESET}"
    else
        echo -e "${YELLOW}Neo4j Docker container not running.${RESET}"
    fi
else
    echo -e "${YELLOW}Docker not installed, cannot stop Neo4j container.${RESET}"
    echo -e "${YELLOW}If Neo4j is running, please stop it manually.${RESET}"
fi

echo -e "\n${GREEN}Advanced RAG system stopped.${RESET}"
EOF

chmod +x stop_advanced_rag.sh

# Keep script running until Ctrl+C with log monitoring options
echo -e "\n${BOLD}Press Ctrl+C to stop the system${RESET}"
echo -e "Type ${BLUE}1${RESET} for backend logs, ${BLUE}2${RESET} for frontend logs, ${BLUE}3${RESET} for Neo4j logs, ${BLUE}4${RESET} for Ollama logs (or Ctrl+C to exit)"

# Set up trap for Ctrl+C
trap 'echo -e "\nStopping Advanced RAG system..."; ./stop_advanced_rag.sh; exit 0' INT

# Listen for user input to show different logs
while true; do
    read -n 1 -t 1 key 2>/dev/null || continue
    
    case $key in
        1)
            echo -e "\n${BOLD}Showing backend logs (Ctrl+C to return):${RESET}"
            tail -f "$LOG_DIR/backend.log" || true
            echo -e "\n${BOLD}Press Ctrl+C to stop the system${RESET}"
            echo -e "Type ${BLUE}1${RESET} for backend logs, ${BLUE}2${RESET} for frontend logs, ${BLUE}3${RESET} for Neo4j logs, ${BLUE}4${RESET} for Ollama logs (or Ctrl+C to exit)"
            ;;
        2)
            echo -e "\n${BOLD}Showing frontend logs (Ctrl+C to return):${RESET}"
            tail -f "$LOG_DIR/frontend.log" || true
            echo -e "\n${BOLD}Press Ctrl+C to stop the system${RESET}"
            echo -e "Type ${BLUE}1${RESET} for backend logs, ${BLUE}2${RESET} for frontend logs, ${BLUE}3${RESET} for Neo4j logs, ${BLUE}4${RESET} for Ollama logs (or Ctrl+C to exit)"
            ;;
        3)
            echo -e "\n${BOLD}Showing Neo4j logs (Ctrl+C to return):${RESET}"
            tail -f "$LOG_DIR/neo4j.log" || true
            echo -e "\n${BOLD}Press Ctrl+C to stop the system${RESET}"
            echo -e "Type ${BLUE}1${RESET} for backend logs, ${BLUE}2${RESET} for frontend logs, ${BLUE}3${RESET} for Neo4j logs, ${BLUE}4${RESET} for Ollama logs (or Ctrl+C to exit)"
            ;;
        4)
            echo -e "\n${BOLD}Showing Ollama logs (Ctrl+C to return):${RESET}"
            tail -f "$LOG_DIR/ollama.log" || true
            echo -e "\n${BOLD}Press Ctrl+C to stop the system${RESET}"
            echo -e "Type ${BLUE}1${RESET} for backend logs, ${BLUE}2${RESET} for frontend logs, ${BLUE}3${RESET} for Neo4j logs, ${BLUE}4${RESET} for Ollama logs (or Ctrl+C to exit)"
            ;;
    esac
done