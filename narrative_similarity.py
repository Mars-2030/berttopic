# narrative_similarity.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_narrative_similarity(df: pd.DataFrame):
    """
    Calculates the narrative overlap between users based on their topic distributions.

    Args:
        df (pd.DataFrame): DataFrame containing 'user_id' and 'topic_id' columns.
                          Should already be filtered to exclude outliers (topic_id == -1).

    Returns:
        pd.DataFrame: A square DataFrame where rows and columns are user_ids
                      and values are the cosine similarity of their topic distributions.
    """
    # Filter out outlier posts if any remain
    df_meaningful = df[df['topic_id'] != -1] if 'topic_id' in df.columns else df

    if df_meaningful.empty:
        return pd.DataFrame()

    # Create the "narrative vector" for each user
    # Rows: user_id, Columns: topic_id, Values: count of posts
    user_topic_matrix = pd.crosstab(df_meaningful['user_id'], df_meaningful['topic_id'])

    # Need at least 2 users for meaningful comparison
    if len(user_topic_matrix) < 2:
        return pd.DataFrame()

    # Normalize rows to get proportions (important for meaningful cosine similarity)
    # This ensures users with different post counts can still be compared fairly
    row_sums = user_topic_matrix.sum(axis=1)
    user_topic_proportions = user_topic_matrix.div(row_sums, axis=0)

    # Calculate pairwise cosine similarity between all users
    similarity_matrix = cosine_similarity(user_topic_proportions)

    # Convert the result back to a DataFrame with user_ids as labels
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_topic_matrix.index,
        columns=user_topic_matrix.index
    )

    return similarity_df


def calculate_text_similarity_tfidf(df: pd.DataFrame):
    """
    Calculates text similarity between users using TF-IDF vectorization.

    Combines all posts from each user into a single document, then compares
    the word frequencies using TF-IDF and cosine similarity.

    Args:
        df (pd.DataFrame): DataFrame containing 'user_id' and 'post_content' columns.

    Returns:
        pd.DataFrame: A square DataFrame where rows and columns are user_ids
                      and values are the cosine similarity of their text content.
    """
    if df.empty or 'post_content' not in df.columns:
        return pd.DataFrame()

    # Combine all posts from each user into a single document
    user_docs = df.groupby('user_id')['post_content'].apply(
        lambda posts: ' '.join(posts.astype(str))
    ).reset_index()
    user_docs.columns = ['user_id', 'combined_text']

    # Need at least 2 users for meaningful comparison
    if len(user_docs) < 2:
        return pd.DataFrame()

    # Create TF-IDF vectors for each user's combined text
    tfidf = TfidfVectorizer(
        max_features=5000,  # Limit vocabulary size for performance
        stop_words='english',
        min_df=1,
        max_df=0.95
    )

    try:
        tfidf_matrix = tfidf.fit_transform(user_docs['combined_text'])
    except ValueError:
        # Empty vocabulary (all stop words or empty texts)
        return pd.DataFrame()

    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Convert to DataFrame with user_ids as labels
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_docs['user_id'],
        columns=user_docs['user_id']
    )

    return similarity_df