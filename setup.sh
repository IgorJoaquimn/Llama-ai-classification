#!/bin/bash

# ==============================================================================
# Llama AI Classification - Setup Script
# ==============================================================================
# This script automates the environment setup using 'uv'.
# It handles: uv detection, dependency syncing, and model caching.
# ==============================================================================

# ANSI Color Codes for organized output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

set -e # Exit immediately if a command exits with a non-zero status

print_header() {
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ Error: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! Warning: $1${NC}"
}

# 1. Check for 'uv' installation
print_header "Step 1: Checking Environment"
if ! command -v uv &> /dev/null; then
    print_warning "'uv' is not installed."
    echo -e "Installing 'uv' via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the cargo env to make uv available in current session if possible
    source $HOME/.cargo/env || true
else
    print_success "'uv' is already installed: $(uv --version)"
fi

# 2. Sync dependencies
print_header "Step 2: Syncing Dependencies"
echo -e "Installing Python and all project dependencies from pyproject.toml..."
if uv sync; then
    print_success "Dependencies synchronized successfully."
else
    print_error "Failed to sync dependencies."
    exit 1
fi

# 3. Create necessary directories
print_header "Step 3: Preparing Workspace"
mkdir -p data/input data/output logs tmp
print_success "Directories 'data/input/', 'data/output/', 'logs/' and 'tmp/' are ready."

# 4. Hugging Face Authentication check
print_header "Step 4: Hugging Face Authentication"
echo -e "Some models (like Llama-3) require a Hugging Face login."
echo -e "Would you like to log in now? (y/N)"
read -r -n 1 -t 30 response
echo # Move to a new line
if [[ "$response" =~ ^([yY])$ ]]; then
    uv run huggingface-cli login
else
    print_warning "Skipping HF login. Ensure you have access to the model defined in config.yaml."
fi

# 5. Pre-downloading Model
print_header "Step 5: Model Pre-caching"
echo -e "Would you like to download and cache the model now? (y/N)"
read -r -n 1 -t 30 response
echo # Move to a new line
if [[ "$response" =~ ^([yY])$ ]]; then
    echo -e "Running download_model.py..."
    uv run python src/download_model.py
    print_success "Model cached successfully."
else
    print_warning "Skipping pre-download. The model will be downloaded during the first run."
fi

# Summary
chmod +x test_gpu.sh
print_header "Setup Complete!"
echo -e "${GREEN}${BOLD}Your environment is ready!${NC}"
echo -e ""
echo -e "${BOLD}Quick Start:${NC}"
echo -e "  1. Place your data in ${CYAN}data/input/df_videos_transcript.parquet${NC}"
echo -e "  2. Run the pipeline:  ${YELLOW}uv run python src/main.py${NC}"
echo -e "  3. Run a GPU test:    ${YELLOW}./test_gpu.sh${NC}"
echo -e ""
echo -e "${BLUE}======================================================================${NC}"
