import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.util import ngrams
import pickle
import numpy as np

INDEX = "./pickles_imp2/"

# Main code

# Load the data previously stored using Pickle
norm_fname = 'Normalized_tdf'
norm_tfwt = open(INDEX+norm_fname, 'rb') # normalized log tf wt for query term in document
norm = pickle.load(norm_tfwt)

bigram_tdf_fname = 'bigram_tdf'
bi_tdf = open(INDEX+bigram_tdf_fname, 'rb') # bigram tdfs in the documents
bigram_tdf = pickle.load(bi_tdf)

bigram_norm_fname = 'bigram_normalized_tdf'
bigram_tfwt = open(INDEX+bigram_norm_fname, 'rb') # normalized log tf wt for bigrams in document
bigram_norm = pickle.load(bigram_tfwt)

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

query = ""
input_q = input("Enter query: ")
q_p = []  # List containing query terms present in vocab

# defining punctuations and removing them
punctuations = '''!()-[]{\};:'",<>./?@#$%^&*_~'''

for char in input_q:
  if char in punctuations:
       char = ' '
  query = query + char
 
q_tokens = nltk.word_tokenize(query)  # Tokenizing query words using nltk
query_tokens = []

# defining stop words and removing them
stop_words = set(stopwords.words('english')) 
for token in q_tokens: 
    if token not in stop_words: 
        query_tokens.append(token)


freq = {}  # dictionary with query term as key and logarithmic tf as value
tf_idf = {}  # dictionary with query term as key and tf_idf wt as value
prod = {}  # dictionary with doc_id as key and product of query wt and doc wt as value
prod_bi = {} # dictionary with doc_id as key and product of query wt and doc wt as value for bigrams
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
# Calculating tf_idf of query vector and sum of squares for cosine normalization
for q in set(q_p):
  freq[q] = (1+np.log10(freq[q]))
  tf_idf[q] = freq[q]*idf[q]
  sos = sos + tf_idf[q]**2

sqrt_sos = np.sqrt(sos) # To normalize the weights

for q in set(q_p):
  tf_idf[q] = tf_idf[q]/sqrt_sos # Final normalized tf_idf wt of query terms



# Calculating tf.idf score for bigrams in query(only for bigrams in top 1000)
freq_bi = {}
idf_bi = {}
tfidf_bi = {}
sos_bi = 0
query_bigrams = list(ngrams(query_tokens,2))

#initializing freq to 0 and getting the idf
for bigram in bigram_tdf:
  freq_bi[bigram] = 0
  if len(bigram_tdf[bigram]) == 0:
    idf_bi[bigram] = 0 
  else:
    idf_bi[bigram] = np.log10(N/len(bigram_tdf[bigram])) 

#Finding the bigram frequencies in the query
for q_bigram in query_bigrams:
  if q_bigram in bigram_tdf:
    freq_bi[q_bigram] = freq_bi[q_bigram] + 1 

#Finding the tf wt for bigrams and getting the tf-idf wt for bigrams
for q in set(query_bigrams):
  if q in bigram_tdf:
    freq_bi[q] = 1 + np.log10(freq_bi[q])
    tfidf_bi[q] = freq_bi[q] * idf_bi[q]
    sos_bi = sos_bi + tfidf_bi[q]**2

sqrt_sos_bi = np.sqrt(sos_bi)

#Normalizing the tf-idf wt of the bigrams
for q in query_bigrams:
  if q in bigram_tdf:
    tfidf_bi[q] = tfidf_bi[q]/sqrt_sos_bi
                   

#Calculating relevance of document to given query using unigrams
for (d_id, d_name) in docs_dict.items():
  prod[d_id] = 0
  for q in query_tokens:  
    if q in norm and d_id in norm[q]:
      prod[d_id] = prod[d_id] + tf_idf[q]*norm[q][d_id]

#Calculating relevance of document to given query using bigrams
for (d_id, d_name) in docs_dict.items():
  prod_bi[d_id] = 0
  for q in query_bigrams:  
    if q in bigram_norm and d_id in bigram_norm[q]:
      prod_bi[d_id] = prod_bi[d_id] + tfidf_bi[q]*bigram_norm[q][d_id]

f_score = {} #Final Score
for (d_id, d_name) in docs_dict.items():
  f_score[d_id] = 0.6*prod[d_id] + 0.4*prod_bi[d_id]

rel_docs = sorted(f_score.items(), key=lambda x: (x[1], x[0]), reverse=True)

# Retrieving the top K docs out of all the ranked documents

topK = rel_docs[:K]
i = 1
for (key, value) in topK:
    title = docs_dict[key]
    print("Document", i)
    i = i+1
    print(key, title, value)
    print("")





