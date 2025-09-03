import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired


def analyze_general_topic_evolution(topic_model, docs, timestamps):
    """
    Analyzes general topic evolution over time.

    Args:
        topic_model: Trained BERTopic model.
        docs (list): List of documents.
        timestamps (list): List of timestamps corresponding to the documents.

    Returns:
        pd.DataFrame: DataFrame with topic evolution information.
    """
    try:
        topics_over_time = topic_model.topics_over_time(docs, timestamps, global_tuning=True)
        return topics_over_time
    except Exception:
        # Fallback for small datasets or cases where evolution can't be computed
        return pd.DataFrame(columns=['Topic', 'Words', 'Frequency', 'Timestamp'])


def analyze_user_topic_evolution(df: pd.DataFrame, topic_model):
    """
    Analyzes topic evolution per user.

    Args:
        df (pd.DataFrame): DataFrame with (
            "user_id", "post_content", "timestamp", and "topic_id" columns.
        topic_model: Trained BERTopic model.

    Returns:
        dict: A dictionary where keys are user_ids and values are DataFrames of topic evolution for that user.
    """
    user_topic_evolution = {}
    for user_id in df["user_id"].unique():
        user_df = df[df["user_id"] == user_id].copy()
        if not user_df.empty and len(user_df) > 1:
            try:
                # Ensure timestamps are sorted for topics_over_time
                user_df = user_df.sort_values(by="timestamp")
                docs = user_df["post_content"].tolist()
                timestamps = user_df["timestamp"].tolist()
                selected_topics = user_df["topic_id"].tolist() # Get topic_ids for the user's posts
                topics_over_time = topic_model.topics_over_time(docs, timestamps, topics=selected_topics, global_tuning=True)
                user_topic_evolution[user_id] = topics_over_time
            except Exception:
                user_topic_evolution[user_id] = pd.DataFrame(columns=['Topic', 'Words', 'Frequency', 'Timestamp'])
        else:
             user_topic_evolution[user_id] = pd.DataFrame(columns=['Topic', 'Words', 'Frequency', 'Timestamp'])
    return user_topic_evolution

if __name__ == "__main__":
    # Example Usage:
    data = {
        "user_id": ["user1", "user2", "user1", "user3", "user2", "user1", "user4", "user3", "user2", "user1", "user5", "user4", "user3", "user2", "user1"],
        "post_content": [
            "This is a great movie, I loved the acting and the plot. It was truly captivating.",
            "The new phone has an amazing camera and long battery life. Highly recommend it.",
            "I enjoyed the film, especially the special effects and the soundtrack. A must-watch.",
            "Learning about AI and machine learning is fascinating. The future is here.",
            "My old phone is so slow, I need an upgrade soon. Thinking about the latest model.",
            "The best part of the movie was the soundtrack and the stunning visuals. Very immersive.",
            "Exploring the vastness of space is a lifelong dream. Astronomy is amazing.",
            "Data science is revolutionizing industries. Predictive analytics is key.",
            "I need a new laptop for work. Something powerful and portable.",
            "Just finished reading a fantastic book on quantum physics. Mind-blowing concepts.",
            "Cooking new recipes is my passion. Today, I tried a spicy Thai curry.",
            "The universe is full of mysteries. Black holes and dark matter are intriguing.",
            "Deep learning models are becoming incredibly sophisticated. Image recognition is impressive.",
            "My current laptop is crashing frequently. Time for an upgrade.",
            "Science fiction movies always make me think about the future of humanity."
        ],
        "timestamp": [
            "2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-02 10:30:00",
            "2023-01-02 14:00:00", "2023-01-03 09:00:00", "2023-01-03 16:00:00",
            "2023-01-04 08:00:00", "2023-01-04 12:00:00", "2023-01-05 10:00:00",
            "2023-01-05 15:00:00", "2023-01-06 09:30:00", "2023-01-06 13:00:00",
            "2023-01-07 11:00:00", "2023-01-07 14:30:00", "2023-01-08 10:00:00"
        ]
    }
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Performing topic modeling (English)...")
    model_en, topics_en, probs_en = perform_topic_modeling(df, language="english")
    df["topic_id"] = topics_en

    print("\nAnalyzing general topic evolution...")
    general_evolution_df = analyze_general_topic_evolution(model_en, df["post_content"].tolist(), df["timestamp"].tolist())
    print(general_evolution_df.head())

    print("\nAnalyzing per user topic evolution...")
    user_evolution_dict = analyze_user_topic_evolution(df, model_en)
    for user_id, evolution_df in user_evolution_dict.items():
        print(f"\nTopic evolution for {user_id}:")
        print(evolution_df.head())