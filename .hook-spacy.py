# hook-spacy.py

from PyInstaller.utils.hooks import collect_data_files, collect_all

# This hook explicitly collects everything spaCy needs, including
# internal language classes, data, and models.

# We will let the spec file handle the models via Tree, 
# but this will ensure all required modules are imported.
# It also provides a robust list of hidden imports.

datas = collect_data_files('spacy')

# Add all necessary language classes and pipelines as hidden imports
hiddenimports = [
    'spacy.lang.en', 
    'spacy.lang.en.stop_words', 
    'spacy.lang.xx',
    'spacy.lang.xx.stop_words',
    'spacy.pipeline',
    'spacy.util',
    'en_core_web_sm', 
    'xx_ent_wiki_sm',
    'spacy.parts_of_speech' ,
    'spacy.lang.lex_attrs',
    'spacy.lang.char_classes',
    'spacy.lang.en.tokenizer_exceptions',
    'spacy.lang.xx.tokenizer_exceptions',
    'spacy.lang.en.stop_words',
    'spacy.lang.xx.stop_words',
    'spacy.lang.en.punctuation',
    'spacy.lang.xx.punctuation',
    'spacy.lang.en.syntax_iterators',
    'spacy.lang.xx.syntax_iterators',
    'spacy.lang.en.syntax_iterators',
    'spacy.lang.norm_exceptions',
    'spacy.lang.en.lemmatizer',
    'spacy.lang.xx.lemmatizer',
    'spacy.lang.en.lookups',
    'spacy.lang.xx.lookups',
    'spacy.lang.en.morphology',
    'spacy.lang.xx.morphology',
    'spacy.lang.en.syntax_iterators',
    'spacy.lang.xx.syntax_iterators',
    'spacy.lang.en.syntax_iterators',
    'spacy.lang.xx.syntax_iterators',
    'spacy.pipeline.transition_parser'

    'thinc.extra.wrappers',
    'cymem.cymem',
    'preshed.maps',
    'blis.py'

]