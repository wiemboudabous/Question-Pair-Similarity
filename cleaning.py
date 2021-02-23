import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
def clean_text(text):
    text = re.sub("\'s", " ", text) 
    """ we have cases like "Sam is" or "Sam's" (i.e. his) these two cases aren't separable,
     I choose to compromise are kill "'s" directly"""
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)    
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    return text

nltk.download('punkt')
from string import punctuation
def clean (text):
   text = ''.join([c for c in text if c not in punctuation]).lower()
    # Return a list of words
   return text

def stemming(text):
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text

def remove_lemmatization(text):
  #is based on The Porter Stemming Algorithm
  stopword = stopwords.words('english')
  wordnet_lemmatizer = WordNetLemmatizer()
 
  word_tokens = nltk.word_tokenize(text)
  
  
  lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
  return lemmatized_word
