import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from natsort import natsorted
from nltk.stem import PorterStemmer

# Function to read and return sorted list of filenames in a directory
def get_file_names(path):
    return natsorted(os.listdir(path))  # path should be the directory containing the files

# --------------------------------------------------- TOKENIZATION ----------------------------------------------------
def tokenize_documents(path):
    """Tokenizes the content of each file in the given path."""
    files_name = get_file_names(path)
    document_terms = []  # List to store tokens for each document

    for file_name in files_name:
        # Read the content of the file
        with open(os.path.join(path, file_name), "r") as f:
            document = f.read()
        # Tokenize content by words
        tokenized_terms = word_tokenize(document)
        document_terms.append(tokenized_terms)  # Append tokens to the main list

    return document_terms

# --------------------------------------------------- STEMMING -------------------------------------------------------
def stem_documents(document_terms):
    """Stems each word in the tokenized document terms using Porter Stemmer."""
    stemmer = PorterStemmer()
    stemmed_documents = []  # List to store stemmed terms for each document

    for terms in document_terms:
        stemmed_terms = [stemmer.stem(word) for word in terms]  # Stem each word in the terms
        stemmed_documents.append(stemmed_terms)

    return stemmed_documents

# --------------------------------------------------- TEST CASE -------------------------------------------------------
# document_terms = tokenize_documents('files')
# print("Tokenized Documents:", document_terms)
# print('-' * 50)
# stemmed_documents = stem_documents(document_terms)
# print("Stemmed Documents:", stemmed_documents)
