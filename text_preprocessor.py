import re
import string
import pandas as pd
import spacy
import emoji
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex

class MultilingualPreprocessor:
    """
    A robust text preprocessor using spaCy for multilingual support.
    """
    def __init__(self, language: str):
        """
        Initializes the preprocessor and loads the appropriate spaCy model.
        
        Args:
            language (str): 'english' or 'multilingual'.
        """
        model_map = {
            'english': 'en_core_web_sm',
            'multilingual': 'xx_ent_wiki_sm'
        }
        self.model_name = model_map.get(language, 'xx_ent_wiki_sm')
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            print(f"spaCy model '{self.model_name}' not found.")
            print(f"Please run: python -m spacy download {self.model_name}")
            raise  # Re-raise the error to be caught by the Streamlit app

        # Customize tokenizer to not split on hyphens in words
        # CORRECTED LINE: CONCAT_QUOTES is wrapped in a list []
        infixes = LIST_ELLIPSES + LIST_ICONS + [CONCAT_QUOTES]
        infix_regex = compile_infix_regex(infixes)
        self.nlp.tokenizer.infix_finditer = infix_regex.finditer

    def preprocess_series(self, text_series: pd.Series, options: dict, n_process_spacy: int = -1) -> pd.Series:
        """
        Applies a series of cleaning steps to a pandas Series of text.
        
        Args:
            text_series (pd.Series): The text to be cleaned.
            options (dict): A dictionary of preprocessing options.

        Returns:
            pd.Series: The cleaned text Series.
        """
        # --- Stage 1: Fast, Regex-based cleaning ---
        processed_text = text_series.copy().astype(str)
        if options.get("remove_html"):
            processed_text = processed_text.str.replace(r"<.*?>", "", regex=True)
        if options.get("remove_urls"):
            processed_text = processed_text.str.replace(r"http\S+|www\.\S+", "", regex=True)

        emoji_option = options.get("handle_emojis", "Keep Emojis")
        if emoji_option == "Remove Emojis":
            processed_text = processed_text.apply(lambda s: emoji.replace_emoji(s, replace=''))
        elif emoji_option == "Convert Emojis to Text":
            processed_text = processed_text.apply(emoji.demojize)
            
        if options.get("handle_hashtags") == "Remove Hashtags":
            processed_text = processed_text.str.replace(r"#\w+", "", regex=True)
        if options.get("handle_mentions") == "Remove Mentions":
            processed_text = processed_text.str.replace(r"@\w+", "", regex=True)

        # --- Stage 2: spaCy-based advanced processing ---
        # Using nlp.pipe for efficiency on a Series
        cleaned_docs = []
        # docs = self.nlp.pipe(processed_text, n_process=-1, batch_size=500)
        docs = self.nlp.pipe(processed_text, n_process=n_process_spacy, batch_size=500)

        
        # Get custom stopwords and convert to lowercase set for fast lookups
        custom_stopwords = set(options.get("custom_stopwords", []))

        for doc in docs:
            tokens = []
            for token in doc:
                # Punctuation and Number handling
                if options.get("remove_punctuation") and token.is_punct:
                    continue
                if options.get("remove_numbers") and (token.is_digit or token.like_num):
                    continue
                
                # Stopword handling (including custom stopwords)
                is_stopword = token.is_stop or token.text.lower() in custom_stopwords
                if options.get("remove_stopwords") and is_stopword:
                    continue
                
                # Use lemma if lemmatization is on, otherwise use the original text
                token_text = token.lemma_ if options.get("lemmatize") else token.text
                
                # Lowercasing (language-aware)
                if options.get("lowercase"):
                    token_text = token_text.lower()
                
                # Remove any leftover special characters or whitespace
                if options.get("remove_special_chars"):
                    token_text = re.sub(r'[^\w\s-]', '', token_text)

                if token_text.strip():
                    tokens.append(token_text.strip())
            
            cleaned_docs.append(" ".join(tokens))
            
        return pd.Series(cleaned_docs, index=text_series.index)