import pandas as pd
import math as m
import numpy as np
import step_1

# Calculate weighted term frequency
def get_weighted_term_freq(x):
    return m.log(x) + 1 if x > 0 else 0

# Calculate term frequency and weighted term frequency for documents
def term_frequence():
    pathT = 'files'
    tokenized_docs = step_1.Tokenization(pathT)
    stemmed_docs = step_1.stemming(tokenized_docs)

    # Create a list of all unique terms
    all_words = list({word for doc in stemmed_docs for word in doc})
    
    def get_term_freq(doc):
        word_count = dict.fromkeys(all_words, 0)
        for word in doc:
            word_count[word] += 1
        return word_count

    term_freq = pd.DataFrame(get_term_freq(stemmed_docs[0]).values(), index=get_term_freq(stemmed_docs[0]).keys())

    # Populate DataFrame with term frequencies for each document
    for i in range(1, len(stemmed_docs)):
        term_freq[i] = get_term_freq(stemmed_docs[i]).values()

    # Rename columns to represent documents (d1, d2, ...)
    term_freq.columns = [f'd{i}' for i in range(1, len(stemmed_docs) + 1)]

    # Calculate weighted term frequency for each document
    weighted_term_freq = term_freq.applymap(get_weighted_term_freq)
    return term_freq, weighted_term_freq

# Calculate document frequency (DF) and inverse document frequency (IDF)
def term_frequence_doc():
    term_freq, weighted_tf = term_frequence()
    df_idf = pd.DataFrame(columns=['df', 'idf'])

    for i in range(len(term_freq)):
        df = term_freq.iloc[i].astype(bool).sum(axis=0)
        df_idf.loc[i, 'df'] = df
        df_idf.loc[i, 'idf'] = m.log10(len(term_freq.columns) / df) if df else 0

    df_idf.index = term_freq.index
    tf_idf = term_freq.multiply(df_idf['idf'], axis=0)
    return df_idf, tf_idf

# Calculate document length and normalize term frequency
def normalization():
    df_idf, tf_idf = term_frequence_doc()
    document_length = pd.DataFrame({col + '_len': [np.sqrt((tf_idf[col]**2).sum())] for col in tf_idf.columns})
    normalized_term = tf_idf.div(document_length.iloc[0], axis=1)
    return document_length, normalized_term

# Stem and tokenize query terms
def query_stemming(query):
    ps = step_1.PorterStemmer()
    tokenized_query = step_1.word_tokenize(query)
    return [ps.stem(word) for word in tokenized_query]

# Calculate similarity between query and documents
def similarity(query_str, normalized_term, df_idf):
    query_terms = query_stemming(query_str)
    query = pd.DataFrame(index=normalized_term.index)
    query['tf'] = [1 if term in query_terms else 0 for term in normalized_term.index]
    query['w_tf'] = query['tf'].apply(get_weighted_term_freq)
    query['idf'] = df_idf['idf'] * query['w_tf']
    query_length = m.sqrt((query['idf'] ** 2).sum())
    query['norm'] = query['idf'] / query_length if query_length else 0

    product = normalized_term.multiply(query['norm'], axis=0)
    scores = {col: product[col].sum() for col in product.columns if all(query['tf'][normalized_term.index.get_loc(term)] for term in query_terms)}
    return scores, query, product, query_length

# Example usage:
# term_freq, weighted_tf = term_frequence()
# print("Term Frequency:\n", term_freq)
# print("Weighted Term Frequency:\n", weighted_tf)
# df_idf, tf_idf = term_frequence_doc()
# doc_length, normalized_tf = normalization()
# query_str = 'antony brutus'
# scores, query, product, query_len = similarity(query_str, normalized_tf, df_idf)
# print("Scores:", scores)
# print("Query vector:", query)
# print("Product matrix:", product)
# print("Query length:", query_len)
