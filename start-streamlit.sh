#!/bin/bash

# IMPORTANT: Replace the path below with the actual path to your miniconda/anaconda installation
# This ensures the 'conda' command is available to the script.
source /Users/mariamalmutairi/miniconda3/etc/profile.d/conda.sh

# Activate your specific Python environment
conda activate nlp

# Run the streamlit app. The '--server.headless true' flag is a good practice
# as it prevents Streamlit from opening a new browser tab on its own.
streamlit run app.py --server.headless true