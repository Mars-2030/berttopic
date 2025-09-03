# import numpy as np
# import pandas as pd
# from typing import List

# def calculate_gini(array):
#     """
#     Calculates the Gini coefficient of a numpy array.
#     Based on: http://www.statsdirect.com/help/default.htm#nonparametric_tests/gini.htm
#     """
#     array = np.array(array)
#     if array.size == 0 or np.all(array == 0): # Check if all elements are zero
#         return 0.0
#     array = array.flatten()
#     if np.amin(array) < 0: # Values cannot be negative: https://en.wikipedia.org/wiki/Gini_coefficient
#         array -= np.amin(array)
#     array = np.sort(array)
#     index = np.arange(1, array.shape[0] + 1)
#     n = array.shape[0]
#     if np.sum(array) == 0: # Avoid division by zero for empty arrays or arrays with all zeros
#         return 0.0
#     return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

# # def calculate_gini_per_user(df: pd.DataFrame):
# #     """
# #     Calculates the Gini coefficient for topic distribution per user.

# #     Args:
# #         df (pd.DataFrame): DataFrame with 'user_id' and 'topic_id' columns.

# #     Returns:
# #         pd.DataFrame: DataFrame with 'user_id' and 'gini_coefficient'.
# #     """
# #     user_gini = []
# #     for user_id in df["user_id"].unique():
# #         user_posts = df[df["user_id"] == user_id]
# #         topic_counts = user_posts["topic_id"].value_counts().values
# #         gini = calculate_gini(topic_counts)
# #         user_gini.append({"user_id": user_id, "gini_coefficient": gini})
# #     return pd.DataFrame(user_gini)

# # def calculate_gini_per_topic(df: pd.DataFrame):
# #     """
# #     Calculates the Gini coefficient for topic distribution per topic across users.

# #     Args:
# #         df (pd.DataFrame): DataFrame with 'user_id' and 'topic_id' columns.

# #     Returns:
# #         pd.DataFrame: DataFrame with 'topic_id' and 'gini_coefficient'.
# #     """
# #     topic_gini = []
# #     for topic_id in df["topic_id"].unique():
# #         topic_posts = df[df["topic_id"] == topic_id]
# #         user_counts = topic_posts["user_id"].value_counts().values
# #         gini = calculate_gini(user_counts)
# #         topic_gini.append({"topic_id": topic_id, "gini_coefficient": gini})
# #     return pd.DataFrame(topic_gini)

# def calculate_gini_per_user(df: pd.DataFrame, all_topics: List[int]):
#     user_gini = []
#     for user_id in df["user_id"].unique():
#         user_posts = df[df["user_id"] == user_id]
        
#         # Get counts for topics the user posted in
#         existing_topic_counts = user_posts["topic_id"].value_counts()
        
#         # Create a full series with all topics, filling missing with 0
#         full_topic_counts = pd.Series(0, index=all_topics)
#         full_topic_counts.update(existing_topic_counts)
        
#         gini = calculate_gini(full_topic_counts.values)
#         user_gini.append({"user_id": user_id, "gini_coefficient": gini})
#     return pd.DataFrame(user_gini)

# def calculate_gini_per_topic(df: pd.DataFrame, all_users: List[str]):
#     topic_gini = []
#     for topic_id in df["topic_id"].unique(): # Or iterate over all_topics if you want Gini for topics with no posts
#         topic_posts = df[df["topic_id"] == topic_id]
        
#         # Get counts for users who posted in this topic
#         existing_user_counts = topic_posts["user_id"].value_counts()
        
#         # Create a full series with all users, filling missing with 0
#         full_user_counts = pd.Series(0, index=all_users)
#         full_user_counts.update(existing_user_counts)
        
#         gini = calculate_gini(full_user_counts.values)
#         topic_gini.append({"topic_id": topic_id, "gini_coefficient": gini})
#     return pd.DataFrame(topic_gini)

# if __name__ == "__main__":
#     # Example Usage with more diverse data:
#     data = {
#         'user_id': ['userA', 'userA', 'userA', 'userB', 'userB', 'userC', 'userC', 'userC', 'userC', 'userD'],
#         'topic_id': [1, 1, 2, 1, 3, 2, 2, 3, 4, 1]
#     }
#     df = pd.DataFrame(data)

#     print("Calculating Gini per user...")
#     gini_per_user_df = calculate_gini_per_user(df)
#     print(gini_per_user_df)

#     print("\nCalculating Gini per topic...")
#     gini_per_topic_df = calculate_gini_per_topic(df)
#     print(gini_per_topic_df)

# gini_calculator.py

import pandas as pd
from math import isnan
import math
from typing import List

def calculate_gini(counts, *, min_posts=None, normalize=False):
    """
    Compute 1 - sum(p_i^2) where p_i are category probabilities (Gini Impurity).
    Handles: list/tuple of counts, dict {cat: count}, numpy array, pandas Series.

    Edge cases:
      - total == 0  -> return float('nan')
      - total == 1  -> return 0.0
      - min_posts set and total < min_posts -> return float('nan')
      - normalize=True -> divide by (1 - 1/k_nonzero) when k_nonzero > 1

    Parameters
    ----------
    counts : Iterable[int] | dict | pandas.Series | numpy.ndarray
        Nonnegative counts per category.
    min_posts : int | None
        If provided and total posts < min_posts, returns NaN.
    normalize : bool
        If True, returns Gini / (1 - 1/k_nonzero) for k_nonzero > 1.

    Returns
    -------
    float
    """
    # Convert to a flat list of counts
    if counts is None:
        return float('nan')

    if isinstance(counts, dict):
        vals = list(counts.values())
    else:
        # Works for list/tuple/np.array/Series
        try:
            vals = list(counts)
        except TypeError:
            return float('nan')

    # Validate & clean
    vals = [float(v) for v in vals if v is not None and not math.isnan(v)]
    if any(v < 0 for v in vals):
        raise ValueError("Counts must be nonnegative.")
    total = sum(vals)

    # Edge cases
    if total == 0:
        return float('nan')
    if min_posts is not None and total < min_posts:
        return float('nan')
    if total == 1:
        base = 0.0
    else:
        # Compute 1 - sum p_i^2
        s2 = sum((v / total) ** 2 for v in vals)
        base = 1.0 - s2

    if not normalize:
        return base

    # Normalization by maximum possible diversity for observed nonzero categories
    k_nonzero = sum(1 for v in vals if v > 0)
    if k_nonzero <= 1:
        # If only one category has posts, diversity is 0 and normalization isn't defined—return 0
        return 0.0
    denom = 1.0 - 1.0 / k_nonzero
    # Guard against floating tiny negatives due to FP
    return max(0.0, min(1.0, base / denom))


def calculate_gini_per_user(df: pd.DataFrame, all_topics: List[int]):
    """
    Calculates the Gini Impurity for topic distribution per user.
    A high value indicates high topic diversity.
    """
    user_gini = []
    for user_id in df["user_id"].unique():
        user_posts = df[df["user_id"] == user_id]
        existing_topic_counts = user_posts["topic_id"].value_counts()
        full_topic_counts = pd.Series(0, index=all_topics)
        full_topic_counts.update(existing_topic_counts)
        # We use normalize=True to make scores more comparable
        gini = calculate_gini(full_topic_counts.values, normalize=True)
        user_gini.append({"user_id": user_id, "gini_coefficient": gini})
    # The new function returns NaN for zero counts, so we fill with 0
    return pd.DataFrame(user_gini).fillna(0)


def calculate_gini_per_topic(df: pd.DataFrame, all_users: List[str]):
    """
    Calculates the Gini Impurity for user distribution per topic.
    A high value indicates the topic is discussed by a diverse set of users.
    """
    topic_gini = []
    for topic_id in df["topic_id"].unique():
        topic_posts = df[df["topic_id"] == topic_id]
        existing_user_counts = topic_posts["user_id"].value_counts()
        full_user_counts = pd.Series(0, index=all_users)
        full_user_counts.update(existing_user_counts)
        gini = calculate_gini(full_user_counts.values, normalize=True)
        topic_gini.append({"topic_id": topic_id, "gini_coefficient": gini})
    return pd.DataFrame(topic_gini).fillna(0)