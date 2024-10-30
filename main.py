import step_1
import step_2
import step_3 

# Path to files directory
path = 'files'

# Tokenization Step
print("\n" + "-" * 50 + " Tokenization " + "-" * 50 + "\n")
tokenization = step_1.Tokenization(path)
print(tokenization)

# Stemming Step
print("\n" + "-" * 50 + " Stemming " + "-" * 50 + "\n")
stemming_doc = step_1.stemming(tokenization)
print(stemming_doc)

# Positional Index Step
print("\n" + "-" * 50 + " Positional Index " + "-" * 50 + "\n")
positional_index = step_2.pos_index(stemming_doc)
print(positional_index)

# Term Frequency (TF) Calculation
print("\n" + "-" * 50 + " Term Frequency (TF) " + "-" * 50 + "\n")
term_frequency, weighted_tf = step_3.term_frequence()
print(term_frequency)

# Weighted Term Frequency Calculation
print("\n" + "-" * 50 + " Weighted TF (1 + log(tf)) " + "-" * 50 + "\n")
print(weighted_tf)

# Document Frequency (DF) & Inverse Document Frequency (IDF)
print("\n" + "-" * 50 + " DF & IDF " + "-" * 50 + "\n")
term_frequency_document, term_frequency_idf = step_3.term_frequence_doc()
print(term_frequency_document)

# TF * IDF Calculation
print("\n" + "-" * 50 + " TF * IDF " + "-" * 50 + "\n")
print(term_frequency_idf)

# Document Length Calculation
print("\n" + "-" * 50 + " Document Length " + "-" * 50 + "\n")
document_length, normalized_terms = step_3.normalization()
print(document_length)

# Normalization Step
print("\n" + "-" * 50 + " Normalization " + "-" * 50 + "\n")
print(normalized_terms)

# Query Phase - Positional Index Query
print("\n" + "-" * 50 + " Query Phase (Positional Index) " + "-" * 50 + "\n")
while True:
    query = input("Please Enter your Query (or type 'no' to exit):\n")
    if query.lower() == "no":
        break
    else:
        query_result = step_2.queryp(query, positional_index)
        print("Query Result:", query_result)

# Similarity Search Phase
print("\n" + "-" * 50 + " Similarity Search " + "-" * 50 + "\n")
qsearch = input("What do you want to search?   ")
scores, query_vec, product, product_sum, query_len = step_3.samliorty(qsearch, normalized_terms, term_frequency_document)

# Display results
print("\nQuery Vector:\n", query_vec)
print("\nProduct Matrix:\n", product)
print("\nQuery Length:\n", query_len)
print("\nSimilarity Scores:\n", product_sum)

# Final Scores
final_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
print("\nDocuments sorted by similarity:\n")
for doc in final_score:
    print(doc[0], end=' ')
