# Use Python base image (avoid slim on HF)
FROM python:3.11

# Set working directory
WORKDIR /app

# Environment variables (use port 7860 for HF Spaces)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

# Install system dependencies (HF-safe)
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    fontconfig \
    fonts-dejavu-core && \
    fc-cache -f && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first (better cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy models (required for text preprocessing)
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download xx_ent_wiki_sm

# Download NLTK data (required for coherence calculation)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy application files
COPY app.py .
COPY topic_modeling.py .
COPY text_preprocessor.py .
COPY gini_calculator.py .
COPY topic_evolution.py .
COPY narrative_similarity.py .
COPY resource_path.py .
COPY sample_data.csv .

# Copy Streamlit config (fixes 403 upload error)
COPY .streamlit/config.toml .streamlit/config.toml

# Create non-root user (HF compatible)
RUN useradd -m appuser
USER appuser

# Expose Streamlit port (7860 for HF Spaces)
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
