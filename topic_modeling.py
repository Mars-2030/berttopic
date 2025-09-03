# topic_modeling.py

import pandas as pd
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from typing import List
from sklearn.feature_extraction.text import CountVectorizer # <-- Make sure this is imported

def perform_topic_modeling(
    docs: List[str],
    language: str = "english",
    nr_topics=None,
    remove_stopwords_bertopic: bool = False, # New parameter to control behavior
    custom_stopwords: List[str] = None
):
    """
    Performs topic modeling on a list of documents.

    Args:
        docs (List[str]): A list of documents. Stopwords should be INCLUDED for best results.
        language (str): Language for the BERTopic model ('english', 'multilingual').
        nr_topics: The number of topics to find ("auto" or an int).
        remove_stopwords_bertopic (bool): If True, stopwords will be removed internally by BERTopic.
        custom_stopwords (List[str]): A list of custom stopwords to use.

    Returns:
        tuple: BERTopic model, topics, probabilities, and coherence score.
    """
    vectorizer_model = None  # Default to no custom vectorizer

    if remove_stopwords_bertopic:
        stop_words_list = []
        if language == "english":
            # Start with the built-in English stopword list from scikit-learn
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            stop_words_list = list(ENGLISH_STOP_WORDS)

        # Add any custom stopwords provided by the user
        if custom_stopwords:
            stop_words_list.extend(custom_stopwords)
        
        # Only create a vectorizer if there's a list of stopwords to use
        if stop_words_list:
            vectorizer_model = CountVectorizer(stop_words=stop_words_list)

    # Instantiate BERTopic, passing the vectorizer_model if it was created
    if language == "multilingual":
        topic_model = BERTopic(language="multilingual", nr_topics=nr_topics, vectorizer_model=vectorizer_model)
    else:
        topic_model = BERTopic(language=language, nr_topics=nr_topics, vectorizer_model=vectorizer_model)

    # The 'docs' passed here should contain stopwords for the embedding model to work best
    topics, probs = topic_model.fit_transform(docs)

    # --- Calculate Coherence Score ---
    # This part remains the same.
    tokenized_docs = [word_tokenize(doc) for doc in docs]
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    topic_words = topic_model.get_topics()
    topics_for_coherence = []
    for topic_id in sorted(topic_words.keys()):
        if topic_id != -1:
            words = [word for word, _ in topic_model.get_topic(topic_id)]
            topics_for_coherence.append(words)
    coherence_score = None
    if topics_for_coherence and corpus:
        try:
            coherence_model = CoherenceModel(
                topics=topics_for_coherence,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
        except Exception as e:
            print(f"Could not calculate coherence score: {e}")
            
    return topic_model, topics, probs, coherence_score