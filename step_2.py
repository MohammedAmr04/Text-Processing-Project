from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import step_1  # Import your custom module

# Create positional index from stemmed documents
def pos_index(stem_document):  # Input: stemmed documents from `stemming` function in step_1
    pos_index = {}  # Initialize the positional index dictionary
    for doc_num, doc in enumerate(stem_document):
        for position, term in enumerate(doc):
            if term in pos_index:
                # Update term frequency and position
                pos_index[term][0] += 1
                if doc_num in pos_index[term][1]:
                    pos_index[term][1][doc_num].append(position)
                else:
                    pos_index[term][1][doc_num] = [position]
            else:
                # Initialize term in index with frequency and positions
                pos_index[term] = [1, {doc_num: [position]}]
    return pos_index

# Search for positional matches in the index
def queryp(query, pos_index):
    # Tokenize and stem the query
    ps = PorterStemmer()
    tokenized_query = word_tokenize(query)
    stemmed_query = [ps.stem(word) for word in tokenized_query]

    # Prepare list to track matched positions across documents
    match_positions = [[] for _ in range(10)]  # Assuming 10 documents max

    # Search positional index for each stemmed query term
    for word in stemmed_query:
        if word in pos_index:
            for doc_num, positions in pos_index[word][1].items():
                if match_positions[doc_num]:
                    if match_positions[doc_num][-1] == positions[0] - 1:
                        match_positions[doc_num].append(positions[0])
                else:
                    match_positions[doc_num].append(positions[0])

    # Output documents with full positional matches for all query terms
    for doc_id, pos_list in enumerate(match_positions):
        if len(pos_list) == len(stemmed_query):
            print(f"Query match found in document {doc_id + 1}")

# Example usage:
# documents = step_1.Tokenization('files')
# stemmed_documents = step_1.stemming(documents)
# positional_index = pos_index(stemmed_documents)

# Example query:
# query = 'brutus caeser'
# queryp(query, positional_index)
