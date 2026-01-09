# run.py

import streamlit.web.cli as stcli
import os
import sys
from resource_path import resource_path # Import resource_path

def run_streamlit():
    # Determine the correct base path at runtime
    if hasattr(sys, '_MEIPASS'):
        # In a PyInstaller bundle, the resource is in the temp folder
        base_path = sys._MEIPASS
    else:
        # In development, the resource is in the current directory
        base_path = os.path.abspath(os.path.dirname(__file__))

    app_path = os.path.join(base_path, 'app.py')
    
    # --- ADD DEBUG PRINT HERE ---
    print(f"DEBUG: Calculated Streamlit app_path: {app_path}")
    
    # Check if the file actually exists at the calculated path (for debugging the build)
    if not os.path.exists(app_path):
        print(f"FATAL: The file does NOT exist at the expected path: {app_path}")
        # We can stop here and force the user to see the error
        sys.exit(1)
    
    # Set the command-line arguments for Streamlit
    sys.argv = [
        "streamlit",
        "run",
        app_path, # Use the correctly calculated path
        "--server.port=8501",
        "--server.headless=true",
        "--global.developmentMode=false",
    ]

    # Run the Streamlit CLI
    sys.exit(stcli.main())

if __name__ == "__main__":
    run_streamlit()