import nltk
from nltk import word_tokenize
import MethodsNLP as mnlp

wordnet = []

def readWordnet():
    wn = open('layout-one.txt','r',encoding='utf-8',errors='ignore').read().split('\n')
    for line in wn:
        line = line.replace('[','').replace(']',';').replace('{','').replace('}','').replace(',',';').replace('<','').replace('>','').replace(' ','').replace('0','').replace('1','').replace('2','').replace('3','').replace('4','').replace('5','').replace('6','').replace('7','').replace('8','').replace('9','').split(';')
        if(line!=[]):
            category = line[0]
            words = line[1::]
            for w in words:
                if('<' in w):
                    words.remove(w)
            wordnet.append(words)

def num_synonyms( word):
    count = 0
    if(type(word)!=str):
        return 0
    for wordSet in wordnet:
        if(wordSet.__contains__(word)):
            count+=len(wordSet)-1
    return count

def get_synonyms( word):
    syns = []
    if(type(word)!=str):
        return 0
    for wordSet in wordnet:
        for words in wordSet:
            if words == word:
                for w in wordSet:
                    if w != word:
                        if w not in syns:
                            syns.append(w)
    return syns

def union(vector, text):
    for word in vector:
        text +=  ' ' + word + ' '
    return text

def addSynonyms(dataSet):
    dataSetSyns = []
    for i in range(len(dataSet)):
        text = ''
        synonyms = []
        for word in mnlp.tokenize(dataSet[i]):
            num = num_synonyms(word.lower())
            if num > 2:
                for x in range(2):
                    synTemp = get_synonyms(word.lower())
                    synonyms.append(synTemp[x])
        text = union(synonyms, dataSet[i])
        dataSetSyns.append(text)
    return dataSetSyns

readWordnet()
print(num_synonyms( "bonito"))
print(get_synonyms( "bonito"))