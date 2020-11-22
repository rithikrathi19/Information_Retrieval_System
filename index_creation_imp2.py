from bs4 import BeautifulSoup
import nltk # importing the natural language toolkit
from nltk.util import ngrams # importing ngrams to create bigram indexes
from nltk.corpus import stopwords
import numpy as np
import pickle # importing pickle to convert objects to bytestream and use later
from collections import Counter

INDEX = "./pickles_imp2/" # creating directory for pickled objects

all_docs = []

for i in [15,19,25,30,45]:
    f = open("AC_wiki_" + str(i), encoding="utf8")
    filecontent = f.read()
    # Separating the different documents using a split on end doc tag
    docs = filecontent.split("</doc>")
    # Parsing the split output docs using bs and lxml leaving the last split as it would be empty
    docs = [BeautifulSoup(doc + "</doc>", "lxml") for doc in docs][:-1]
    all_docs.extend(docs)
    f.close()


# Creating the list of doc ids, doc titles and document text to zip them together
doc_id = [] # List that contains ids of all documents
doc_title = [] # List that contains title of all documents
doc_text = [] # List that contains content of all documents
docs_dict = {} # term_dictonary with id and title as key-value
for doc in all_docs:
    id = doc.find_all("doc")[0].get("id")
    title = doc.find_all("doc")[0].get("title")
    text = doc.get_text()
    doc_id.append(id)
    doc_title.append(title)
    doc_text.append(text)
    docs_dict[id] = title
# Zipping all the lists together
indexed_docs = list(zip(doc_id, doc_title, doc_text))

# Creation of vocabulary with all the content we have
tokens = []
for page in doc_text:
    # Using the nltk word tokenizer to get word tokens from a given doc content
    tokens.extend(nltk.word_tokenize(page))
vocabulary = sorted(set(tokens))

tdf = {}  # term frequency for all terms in the vocab
for term in vocabulary:
    tdf[term] = {}
for document in indexed_docs:
    d_id = document[0] # ID of the specific doc
    d_tokens = nltk.word_tokenize(document[2]) # Tokens for specific doc
    for term in d_tokens:
        if term in tdf: # Why is this required as all terms are there in vocab
            if d_id in tdf[term]:
                tdf[term][d_id] = tdf[term][d_id] + 1
            else:
                tdf[term][d_id] = 1

wt = {}  # for storing the tf-wt(logarithmic)
sos = {}  # sum of squares of wt for final normalization
for doc in doc_id:
    sos[doc] = 0
for term in vocabulary:
    term_dict = tdf[term]
    wt[term] = {}
    for key, value in term_dict.items():
        wt[term][key] = 1 + np.log10(value) # key is the doc id and value is tf for respective doc id
        sos[key] = sos[key] + wt[term][key] ** 2

norm = {}  # Normalized logarithmic term document frequencies
for term in vocabulary:
    term_dict = tdf[term]
    norm[term] = {}
    for key, value in term_dict.items():
        norm[term][key] = wt[term][key] / (np.sqrt(sos[key]))


#bigrams
bigrams = [] #list of all bigrams in corpus
bigram_freq = {} # dictionary which contains bigrams as key as frequency as value
first_word = {} # dictionary which contains first word 
second_word = {} # dictionary which contains second word 

stop_words = set(stopwords.words('english')) 
for content in doc_text:
    temp = list(ngrams(nltk.word_tokenize(content),2))
    bigrams.extend(temp)

unique_bigrams = list(set(bigrams)) # list of unique bigrams
num_bigrams = len(bigrams) # number of bigrams

#Creating the dictionary containing the bigram frequency
for val in bigrams:						
  if val in bigram_freq:
    bigram_freq[val] = bigram_freq[val] + 1
  else:
    bigram_freq[val]=1

#frequency of tokens
for token in tokens:
    first_word[token] = 0
    second_word[token] = 0
for token in tokens:
    first_word[token] = first_word[token] + 1
    second_word[token] = second_word[token] + 1

chi_square_scores = {}

#calculating chi-square scores for all the bigrams
for bigram in unique_bigrams:			
	word1 = bigram[0] #word1 of the bigram
	word2 = bigram[1] #word2 of the bigram
	o11 = bigram_freq[bigram] #freq of bigram in the corpus
	o21 = first_word[word1] - o11 
	o12 = second_word[word2] - o11
	o22 = num_bigrams - o11 - o21 - o12
	chi_score = num_bigrams*(((o11*o22-o21*o12)**2)/((o11+o21)*(o11+o12)*(o21+o22)*(o12+o22)))
	if(o21 + o12 > 10):
			chi_square_scores[bigram] = chi_score

#sort collocations in descending order of importance
collocations = sorted(chi_square_scores.items(), key = lambda x:(x[1], x[0]),reverse=True) 

frequent_collocations = [] #storing the top 1000 collocations

#Getting the 1000 most common bigrams
count = 0
for (bi,chi_score) in collocations:
    count = count + 1
    if count <= 1000:
          frequent_collocations.append(bi)
    else:
      break
   
bigram_tdf = {} #Dictionary containing tdf of the bigrams in the documents								
for term in frequent_collocations:
    bigram_tdf[term] = {}

#To create natural term document frequency of the frequent collocations
for doc in indexed_docs:              
    d_id = doc[0]
    doc_bigrams = ngrams(nltk.word_tokenize(doc[2]),2)
    for bigram in doc_bigrams:
        if bigram not in bigram_tdf:
            continue
        if d_id in bigram_tdf[bigram]:
            bigram_tdf[bigram][d_id] = bigram_tdf[bigram][d_id] + 1
        else:
            bigram_tdf[bigram][d_id] = 1

#Calculating the bigram normalized logarithmic tf for top 1000 collocations
bigram_wt = {}							
bigram_sos = {}
bigram_norm = {}

for doc in doc_id:
    bigram_sos[doc] = 0

#Getting the bigram normalised tdf
for bigram in bigram_tdf:
    bigram_dict = bigram_tdf[bigram]
    bigram_wt[bigram] = {}
    bigram_norm[bigram] = {}
    for key,value in bigram_dict.items():
        bigram_wt[bigram][key] = 1 + np.log10(value)
        bigram_sos[key] = bigram_sos[key] + bigram_wt[bigram][key]**2
    for key,value in bigram_dict.items():
        bigram_norm[bigram][key] = bigram_wt[bigram][key] / (np.sqrt(bigram_sos[key]))


#dictionary for inverse document frequency of all terms in vocab
idf = {}  
for term in vocabulary:
    if len(norm[term]) == 0:
        idf[term] = 0
    else:
        idf[term] = np.log10(len(all_docs) / len(norm[term]))

# Creation of the index and dumping as bytestream using pickle
# Using wb to write in binary mode.
norm_tfwt = open(INDEX + 'Normalized_tdf', 'wb')
pickle.dump(norm, norm_tfwt)
norm_tfwt.close()

term_idfs = open(INDEX + 'term_idf', 'wb')
pickle.dump(idf, term_idfs)
term_idfs.close()

doc_dictionary = open(INDEX + 'docs_dict', 'wb')
pickle.dump(docs_dict, doc_dictionary)
doc_dictionary.close()

vocab = open(INDEX + 'Vocabulary', 'wb')
pickle.dump(vocabulary, vocab)
vocab.close()

bigram_tdfs = open(INDEX+'bigram_tdf','wb')
pickle.dump(bigram_tdf, bigram_tdfs)
bigram_tdfs.close()

bigram_norm_tfwt = open(INDEX+'bigram_normalized_tdf','wb')
pickle.dump(bigram_norm, bigram_norm_tfwt)
bigram_norm_tfwt.close()
