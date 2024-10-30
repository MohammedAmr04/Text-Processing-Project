
# Definitions for Text Processing Project

## Tokenization
- **Law**: The Tokenization Law states that a string can be broken down into smaller components (tokens) based on specific delimiters (like spaces and punctuation).

## Stemming
- **Law**: The Stemming Law applies reduction techniques to convert words into their root forms to improve matching efficiency.
- **Algorithms Used**:
  - **Porter Stemmer**: Applies a series of rules to remove common morphological and inflexional endings.
  - **Snowball Stemmer**: A more refined version of the Porter algorithm for multiple languages.

## Positional Indexing
- **Law**: The Indexing Law states that creating an index can improve search efficiency by providing direct access to document locations of terms.
- **Implementation**: Maps each term to its positions in each document.

## Term Frequency (TF)
- **Formula**:
  \[
  TF(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of terms in document } d}
  \]
- **Law**: The Term Frequency Law suggests that the frequency of a term within a document is a strong indicator of its relevance.

## Inverse Document Frequency (IDF)
- **Formula**:
  \[
  IDF(t) = \log\left(\frac{N}{df(t)}\right)
  \]
  where \( N \) is the total number of documents and \( df(t) \) is the number of documents containing the term \( t \).
