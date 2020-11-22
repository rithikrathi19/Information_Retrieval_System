from bs4 import BeautifulSoup
import nltk # importing the natural language toolkit
# nltk.download('punkt')
import numpy as np
import pickle # importing pickle to convert objects to bytestream and use later

INDEX = "./pickles/" # creating directory for pickled objects

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

idf = {}  # dict for inverse document frequency of all terms in vocab
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
