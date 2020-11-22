import nltk
import pickle
import numpy as np

INDEX = "./pickles/"

# Main code

# Load the data previously stored using Pickle
norm_fname = 'Normalized_tdf'
norm_tfwt = open(INDEX+norm_fname, 'rb') # normalized log tf wt for query term in document
norm = pickle.load(norm_tfwt)

idf_fname = 'term_idf' 
term_idfs = open(INDEX+idf_fname, 'rb')  # IDF wts of all terms in vocab
idf = pickle.load(term_idfs)

docs_dict_fname = 'docs_dict'
doc_dictionary = open(INDEX+docs_dict_fname, 'rb') # Documents IDs and Title
docs_dict = pickle.load(doc_dictionary)

vocabulary_fname = 'vocabulary'
vocab = open(INDEX+vocabulary_fname, 'rb') # Terms present in the corpus
vocabulary = pickle.load(vocab)

  
N = len(docs_dict)  # number of documents
K = 10  # top K documents to be returned

# defining punctuations and removing them
punctuations = '''!()-[]{\};:'",<>./?@#$%^&*_~'''
query = ""
input_q = input("Enter query: ")
for char in input_q:
  if char in punctuations:
       char = ' '
  query = query + char
 

query_tokens = nltk.word_tokenize(query)  # Tokenizing query words using nltk
q_p = []  # List containing query terms present in vocab

freq = {}  # dictionary with query term as key and logarithmic tf as value
tf_idf = {}  # dictionary with query term as key and tf_idf wt as value
prod = {}  # dictionary with doc_id as key and product of query wt and doc wt as value
sos = 0  # Variable to store sum of squares to be used in normalization


# Checking if term present in vocab and storing it in a new list
for token in query_tokens:
    if(token in vocabulary):
        q_p.append(token)

# Storing term freq for query for terms present in vocab and 0 if not present
# freq is a dict variable with term as key and frequency as value
for token in query_tokens:
    if(token in q_p):
        if (token in freq):
            freq[token] = freq[token]+1
        else:
            freq[token] = 1
    else:
        freq[token] = 0
        tf_idf[token] = 0


# Converting natural term frequency to its logarithmic equivalent --> 1 + log(tf)
# calculating tf_idf of query vector and sum of squares for cosine normalization
for q in set(q_p):
  freq[q] = (1+np.log10(freq[q]))
  tf_idf[q] = freq[q]*idf[q]
  sos = sos + tf_idf[q]**2

sqrt_sos = np.sqrt(sos) # To normalize the weights

for q in set(q_p):
  tf_idf[q] = tf_idf[q]/sqrt_sos # Final normalized tf_idf wt of query terms


#Calculating relevance of document to given query using vector space model calculations
for (d_id, d_name) in docs_dict.items():
  prod[d_id] = 0
  for q in query_tokens:  
    if q in norm and d_id in norm[q]:
      prod[d_id] = prod[d_id] + tf_idf[q]*norm[q][d_id]


rel_docs = sorted(prod.items(), key=lambda x: (x[1], x[0]), reverse=True)

# Retrieving the top K docs out of all the ranked documents
topK = rel_docs[:K]
i = 1
for (key, value) in topK:
    title = docs_dict[key]
    print("Document", i)
    i = i+1
    print(key, title, value)
    print("")





