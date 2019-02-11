import string
import nltk
from nltk.stem import RSLPStemmer
#nltk.download('rslp')
#nltk.download('stopwords')

def tokenize(sentence):
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)
    sentenceFinal = []
    for word in sentence:
        if word not in string.punctuation + "\..." and word != '``' and word != '"' and word != "“":
            sentenceFinal.append(word)
    return sentenceFinal

def stemming(sentence):
    stemmer = RSLPStemmer()
    phrase = []
    for word in tokenize(sentence):
        phrase.append(stemmer.stem(word.lower()))
    return phrase

def removeStopWords(sentence):
    stopwords = nltk.corpus.stopwords.words('portuguese')

    phrase = []
    for word in tokenize(sentence):
        if word not in stopwords:
            phrase.append(word)
    return phrase

def convertText(text):
    finalText = ''
    for word in text:
        finalText = finalText + ' ' + word
    return finalText