# Social Media Topic Modeling System

A comprehensive topic modeling system for social media analysis built with Streamlit and BERTopic. This application supports flexible CSV column mapping, multilingual topic modeling, Gini coefficient calculation, and topic evolution analysis.

## Features

- **📊 Topic Modeling**: Uses BERTopic for state-of-the-art topic modeling.
- **⚙️ Flexible Configuration**:
    - **Custom Column Mapping**: Use any CSV file by mapping your columns to `user_id`, `post_content`, and `timestamp`.
    - **Topic Number Control**: Let the model find topics automatically or specify the exact number you need.
- **🌍 Multilingual Support**: Handles English and 50+ other languages.
- **📈 Gini Coefficient Analysis**: Calculates topic distribution inequality per user and per topic.
- **⏰ Topic Evolution**: Tracks how topics change over time.
- **🎯 Interactive Visualizations**: Built-in charts and data tables using Plotly.
- **📱 Responsive Interface**: Clean, modern Streamlit interface with a control sidebar.

## Requirements

### CSV File Format

Your CSV file must contain columns that can be mapped to the following roles:
- **User ID**: A column with unique identifiers for each user (string).
- **Post Content**: A column with the text content of the social media post (string).
- **Timestamp**: A column with the date and time of the post (e.g., "2023-01-15 14:30:00").

The application will prompt you to select the correct column for each role after you upload your file.

### Dependencies

See `requirements.txt` for a full list of dependencies.

## Installation

### Option 1: Local Installation

1.  **Clone or download the project files.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Option 2: Docker Installation (Recommended)

1.  **Using Docker Compose (easiest):**
    ```bash
    docker-compose up --build
    ```
2.  **Access the application:**
    ```
    http://localhost:8501
    ```

## Usage

1.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
2.  **Open your browser** and navigate to `http://localhost:8501`.
3.  **Follow the steps in the sidebar:**
    - **1. Upload CSV File**: Click "Browse files" to upload your dataset.
    - **2. Map Data Columns**: Once uploaded, select which of your columns correspond to `User ID`, `Post Content`, and `Timestamp`.
    - **3. Configure Analysis**:
        - **Language Model**: Choose `english` for English-only data or `multilingual` for other languages.
        - **Number of Topics**: Enter a specific number of topics to find, or use `-1` to let the model decide automatically.
        - **Custom Stopwords**: (Optional) Enter comma-separated words to exclude from analysis.
    - **4. Run Analysis**: Click the "🚀 Analyze Topics" button.

4.  **Explore the results** in the five interactive tabs in the main panel.

### Using the Interface

The application provides five main tabs:

#### 📋 Overview
- Key metrics, dataset preview, and average Gini coefficient.

#### 🎯 Topics
- Topic information table and topic distribution bar chart.

#### 📊 Gini Analysis
- Analysis of topic diversity for each user and user concentration for each topic.

#### 📈 Topic Evolution
- Timelines showing how topic popularity changes over time, for all users and for individual users.

#### 📄 Documents
- A detailed view of your original data with assigned topics and probabilities.

## Understanding the Results

### Gini Coefficient
- **Range**: 0 to 1
- **User Gini**: Measures how diverse a user's topics are. **0** = perfectly diverse (posts on many topics), **1** = perfectly specialized (posts on one topic).
- **Topic Gini**: Measures how concentrated a topic is among users. **0** = widely discussed by many users, **1** = dominated by a few users.

---

**Built with ❤️ using Streamlit and BERTopic**