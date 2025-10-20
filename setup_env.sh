#!/bin/bash

# GLOBALS
# -------

# Define ANSI color escape codes
ERROR='\033[38;2;255;173;194m'
INFO='\033[38;2;204;219;253m'
SUCCESS='\033[38;2;207;225;185m'
NC='\033[0m'  # No Color

# Define variables
VENV_PATH=".venv"
REQUIREMENTS_FILE="requirements.txt"

# FUNCTIONS
# ---------

# Function to check Python requirements and install virtualenv if needed
check_python() {
    # Check if Python 3 is available
    if ! command -v python3 &> /dev/null; then
        echo "${ERROR}[ERROR] Python 3 is not installed!${NC}"
        echo "${INFO}[INFO] On Debian/Ubuntu systems, run: sudo apt install python3${NC}"
        return 1
    fi

    # Check if pip is available
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        echo "${ERROR}[ERROR] pip3 is not installed!${NC}"
        echo "${INFO}[INFO] Installing pip in user mode...${NC}"
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        python3 get-pip.py --user
        rm get-pip.py
    fi

    # Check if virtualenv is installed, if not install it
    if ! command -v virtualenv &> /dev/null; then
        echo "${INFO}[INFO] virtualenv not found, installing in user mode...${NC}"
        python3 -m pip install --user virtualenv
    fi

    # Verify virtualenv installation
    if ! command -v virtualenv &> /dev/null; then
        echo "${ERROR}[ERROR] Failed to install virtualenv!${NC}"
        return 1
    fi

    return 0
}

# Function to deactivate and remove the virtual environment
deactivate_venv() {
    local DEACT=$1
    local VENV_EXISTS=false
    
    echo "${INFO}[INFO] Deactivating virtual env - ${VENV_PATH}...${NC}"
    
    # Check if we are in a virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate || true
        VENV_EXISTS=true
    fi
    
    if [ -d "$VENV_PATH" ]; then
        rm -rf "$VENV_PATH"
        echo "${SUCCESS}[SUCCESS] Virtual environment removed!${NC}"
    fi
    
    if [ "$DEACT" = false ] && [ "$VENV_EXISTS" = true ]; then
        echo ""
    fi
}

# Function to create and activate the virtual environment, and install requirements
activate_venv() {
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        echo "${ERROR}[ERROR] ${REQUIREMENTS_FILE} not found!${NC}"
        return 1
    fi

    echo "${INFO}[INFO] Creating virtual env ${VENV_PATH}...${NC}"
    if ! virtualenv "$VENV_PATH" --python=python3 --quiet; then
        echo "${ERROR}[ERROR] Failed to create virtual environment!${NC}"
        return 1
    fi
    
    if [ ! -f "${VENV_PATH}/bin/activate" ]; then
        echo "${ERROR}[ERROR] Virtual environment was created but activate script is missing!${NC}"
        rm -rf "$VENV_PATH"
        return 1
    fi
    
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"

    echo "${INFO}[INFO] Installing requirements...${NC}"
    python3 -m pip install --upgrade pip --quiet
    if python3 -m pip install -r "$REQUIREMENTS_FILE" --quiet; then
        echo "${SUCCESS}[SUCCESS] Requirements installed successfully!${NC}"
    else
        echo "${ERROR}[ERROR] Failed to install requirements.${NC}"
        deactivate_venv true
        return 1
    fi

    echo "${SUCCESS}[SUCCESS] Virtual environment is ready!${NC}"
    return 0
}

# Function to clean Python cache files
clean_cache() {
    echo "${INFO}[INFO] Cleaning Python cache files...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
}

# SCRIPT
# ------

# First check if Python is available
check_python || exit 1

# Main script logic
case "$1" in
    ("--activate"|"")
        if [ -n "$VIRTUAL_ENV" ]; then
            deactivate_venv false
        fi
        activate_venv
        ;;
    ("--deactivate")
        deactivate_venv true
        clean_cache
        ;;
    ("--clean")
        deactivate_venv true
        clean_cache
        #rm -rf plots/* saved_models/* 2>/dev/null || true
        echo "${SUCCESS}[SUCCESS] Environment and generated files cleaned!${NC}"
        ;;
    (*)
        echo "${ERROR}[ERROR] Invalid argument: $1${NC}"
        echo "${INFO}[USAGE] source ./setup_env.sh [--activate|--deactivate|--clean]${NC}"
        echo "${INFO} - Use --activate or no argument to activate virtual env${NC}"
        echo "${INFO} - Use --deactivate to deactivate virtual env${NC}"
        echo "${INFO} - Use --clean to deactivate and clean all generated files${NC}"
        ;;
esac
