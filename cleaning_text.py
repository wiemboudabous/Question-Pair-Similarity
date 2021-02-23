from cleaning import clean_text 
from cleaning import clean 
from cleaning import stemming 
from cleaning import remove_lemmatization 
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def apply_clean(text):
    cleaned=clean_text(text)
    cleaned=clean(cleaned)
    cleaned=stemming(cleaned)
    cleaned=remove_lemmatization(cleaned)
    return cleaned

def vectorize(text):
    tfidf = TfidfVectorizer() #  Convert a collection of raw documents to a matrix of TF-IDF features
    text=[text]
    tfidf.fit_transform(text)  # Converting out text to a matrix of TF-IDF features
    word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    nlp = spacy.load("C:\Python39\Lib\site-packages\en_core_web_sm\en_core_web_sm-2.3.1")
    doc1 = nlp(str(text))
    vecs1 = []
    mean_vec1 = np.zeros([len(doc1), len(doc1[0].vector)])
    for word1 in doc1:
        vec1 = word1.vector
        try:
          idf = word2tfidf[str(word1)]
        except:
          idf = 0
        mean_vec1 += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vecs1.append(mean_vec1)  
    return vecs1
def all_process(text):
    txt_field1=form['txt_field1']
    txt_field2=form['txt_field2']
    print(type(txt_field1))
    cleaned1=apply_clean(txt_field1).apply(str)
    cleaned2=apply_clean(txt_field2).apply(str)
    vec1=vectorize(cleaned1)
    vec2=vectorize(cleaned2)
    x=np.vstack((vec1,vec2))
    x_test=x.reshape(1,192)
    return x_test

      
        