# narrative_similarity.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def calculate_narrative_similarity(df: pd.DataFrame):
    """
    Calculates the narrative overlap between users based on their topic distributions.

    Args:
        df (pd.DataFrame): DataFrame containing 'user_id' and 'topic_id' columns.

    Returns:
        pd.DataFrame: A square DataFrame where rows and columns are user_ids
                      and values are the cosine similarity of their topic distributions.
    """
    # 1. Filter out outlier posts for a more meaningful similarity score
    df_meaningful = df[df['topic_id'] != -1]

    # 2. Create the "narrative vector" for each user
    #    Rows: user_id, Columns: topic_id, Values: count of posts
    user_topic_matrix = pd.crosstab(df_meaningful['user_id'], df_meaningful['topic_id'])

    # 3. Calculate pairwise cosine similarity between all users
    similarity_matrix = cosine_similarity(user_topic_matrix)

    # 4. Convert the result back to a DataFrame with user_ids as labels
    similarity_df = pd.DataFrame(similarity_matrix, index=user_topic_matrix.index, columns=user_topic_matrix.index)
    
    return similarity_df