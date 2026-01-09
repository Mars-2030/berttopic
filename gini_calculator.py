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
    Optimized with groupby for better performance.
    """
    def compute_user_gini(group):
        existing_topic_counts = group["topic_id"].value_counts()
        full_topic_counts = pd.Series(0, index=all_topics)
        full_topic_counts.update(existing_topic_counts)
        return calculate_gini(full_topic_counts.values, normalize=True)

    # Use groupby instead of loop for O(n) instead of O(n*m) complexity
    user_gini = df.groupby("user_id").apply(compute_user_gini).reset_index()
    user_gini.columns = ["user_id", "gini_coefficient"]
    return user_gini.fillna(0)


def calculate_gini_per_topic(df: pd.DataFrame, all_users: List[str]):
    """
    Calculates the Gini Impurity for user distribution per topic.
    A high value indicates the topic is discussed by a diverse set of users.
    Optimized with groupby for better performance.
    """
    def compute_topic_gini(group):
        existing_user_counts = group["user_id"].value_counts()
        full_user_counts = pd.Series(0, index=all_users)
        full_user_counts.update(existing_user_counts)
        return calculate_gini(full_user_counts.values, normalize=True)

    # Use groupby instead of loop for O(n) instead of O(n*m) complexity
    topic_gini = df.groupby("topic_id").apply(compute_topic_gini).reset_index()
    topic_gini.columns = ["topic_id", "gini_coefficient"]
    return topic_gini.fillna(0)