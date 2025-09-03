# Social Media Topic Modeling System

A comprehensive topic modeling system for social media analysis built with Streamlit and BERTopic. This application supports flexible CSV column mapping, multilingual topic modeling, Gini coefficient calculation for diversity analysis, topic evolution tracking, and semantic narrative overlap detection.

## Features

- **📊 Topic Modeling**: Uses BERTopic for state-of-the-art, transformer-based topic modeling.
- **⚙️ Flexible Configuration**:
    - **Custom Column Mapping**: Use any CSV file by mapping your columns to `user_id`, `post_content`, and `timestamp`.
    - **Topic Number Control**: Let the model find topics automatically or specify the exact number you need.
- **🌍 Multilingual Support**: Handles English and 50+ other languages using appropriate language models.
- **📈 Gini Index Analysis**: Calculates topic and user diversity.
- **⏰ Topic Evolution**: Tracks how topic popularity and user interests change over time with interactive charts.
- **🤝 Narrative Overlap Analysis**: Identifies users with semantically similar posting patterns (shared narratives), even when their wording differs.
- **✍️ Interactive Topic Refinement**: Fine-tune topic quality by adding words to a custom stopword list directly from the dashboard.
- **🎯 Interactive Visualizations**: A rich dashboard with built-in charts and data tables using Plotly.
- **📱 Responsive Interface**: Clean, modern Streamlit interface with a control panel for all settings.

## Requirements

### CSV File Format

Your CSV file must contain columns that can be mapped to the following roles:
- **User ID**: A column with unique identifiers for each user (string).
- **Post Content**: A column with the text content of the social media post (string).
- **Timestamp**: A column with the date and time of the post.

The application will prompt you to select the correct column for each role after you upload your file.

#### A Note on Timestamp Formatting

The application is highly flexible and can automatically parse many common date and time formats thanks to the powerful Pandas library. However, to ensure 100% accuracy and avoid errors, please follow these guidelines for your timestamp column:

*   **Best Practice (Recommended):** Use a standard, unambiguous format like ISO 8601.
    - `YYYY-MM-DD HH:MM:SS` (e.g., `2023-10-27 15:30:00`)
    - `YYYY-MM-DDTHH:MM:SS` (e.g., `2023-10-27T15:30:00`)

*   **Supported Formats:** Most common formats will work, including:
    - `MM/DD/YYYY HH:MM` (e.g., `10/27/2023 15:30`)
    - `DD/MM/YYYY HH:MM` (e.g., `27/10/2023 15:30`)
    - `Month D, YYYY` (e.g., `October 27, 2023`)

*   **Potential Issues to Avoid:**
    - **Ambiguous formats:** A date like `01/02/2023` can be interpreted as either Jan 2nd or Feb 1st. Using a `YYYY-MM-DD` format avoids this.
    - **Mixed formats in one column:** Ensure all timestamps in your column follow the same format for best performance and reliability.
    - **Timezone information:** Formats with timezone offsets (e.g., `2023-10-27 15:30:00+05:30`) are fully supported.

### Dependencies

See `requirements.txt` for a full list of dependencies.

## Installation

### Option 1: Local Installation

1.  **Clone or download the project files.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download spaCy models:**
    ```bash
    python -m spacy download en_core_web_sm
    python -m spacy download xx_ent_wiki_sm
    ```

### Option 2: Docker Installation (Recommended)

1.  **Using Docker Compose (easiest):**
    ```bash
    docker-compose up --build
    ```
2.  **Access the application:**
    Open your browser and go to `http://localhost:8501`.

## Usage

1.  **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
2.  **Open your browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
3.  **Follow the steps in the application:**
    - **1. Upload CSV File**: Click "Browse files" to upload your dataset.
    - **2. Map Data Columns**: Once uploaded, select which of your columns correspond to `User ID`, `Post Content`, and `Timestamp`.
    - **3. Configure Analysis**:
        - **Language Model**: Choose `english` for English-only data or `multilingual` for other languages.
        - **Number of Topics**: Enter a specific number of meaningful topics to find, or use `-1` to let the model decide automatically.
        - **Text Preprocessing**: Expand the advanced options to select cleaning steps like lowercasing, punctuation removal, and more.
        - **Custom Stopwords**: (Optional) Enter comma-separated words to exclude from analysis.
    - **4. Run Analysis**: Click the "🚀 Run Full Analysis" button.

4.  **Explore the results** in the interactive sections of the main panel.

### Exploring the Interface

The application provides a series of detailed sections:

#### 📋 Overview & Preprocessing
- Key metrics (total posts, unique users), dataset time range, and a topic coherence score.
- A sample of your data showing the original and processed text.

#### 🎯 Topic Visualization & Refinement
- **Word Clouds**: Visual representation of the most important words for top topics.
- **Interactive Word Lists**: Interactively select words from topic lists to add them to your custom stopwords for re-analysis.

#### 📈 Topic Evolution
- An interactive line chart showing how topic frequencies change over the entire dataset's timespan.

#### 🧑‍🤝‍🧑 User Engagement Profile
- A scatter plot visualizing the relationship between the number of posts a user makes and the diversity of their topics.
- An expandable section showing the distribution of users by their post count.

#### 👤 User Deep Dive
- Select a specific user to analyze.
- View their key metrics, overall topic distribution pie chart, and their personal topic evolution over time.
- See detailed tables of their topic breakdown and their most recent posts.

#### 🤝 Narrative Overlap Analysis
- Select a user to find other users who discuss a similar mix of topics.
- Use the slider to adjust the similarity threshold.
- The results table shows the overlap score and post count of similar users, providing context on both narrative alignment and engagement level.

## Understanding the Results

### Gini Impurity Index
This application uses the **Gini Impurity Index**, a measure of diversity.
- **Range**: 0 to 1
- **User Gini (Topic Diversity)**: Measures how diverse a user's topics are. **0** = perfectly specialized (posts on only one topic), **1** = perfectly diverse (posts spread evenly across all topics).
- **Topic Gini (User Diversity)**: Measures how concentrated a topic is among users. **0** = dominated by a single user, **1** = widely and evenly discussed by many users.

### Narrative Overlap Score
- **Range**: 0 to 1
- This score measures the **cosine similarity** between the topic distributions of two users.
- A score of **1.0** means the two users have an identical proportional interest in topics (e.g., both are 100% focused on Topic 3).
- A score of **0.0** means their topic interests are completely different.
- This helps identify users with similar narrative focus, regardless of their total post count.

