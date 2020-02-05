# coding: utf-8
import gensim
import numpy as np
from scipy import spatial
import string
import nltk
nltk.download('averaged_perceptron_tagger')

def startModel():
    print("reading model")
    model = gensim.models.KeyedVectors.load_word2vec_format("cbow_s50.txt")
    word_vectors = model.wv
    print("read complete")
    return word_vectors, model

def calculateSimilarity(word_vectors, model, words1, words2):

    sentence1 = converteStringToVector(words1)
    sentence2 = converteStringToVector(words2)

    if isinstance(sentence1, float) or isinstance(sentence2, float):
        return 0
    else:
        if(len(sentence1) > len(sentence2)):
            return tableSimilarity(word_vectors, model, sentence2, sentence1)
        else:
            return tableSimilarity(word_vectors, model, sentence1, sentence2)


def wordOrderSimilarity(word_vectors, model, sent1, sent2):

    similarity = 0
    cont = 0
    dif = 0

    #separa as sentenças em tokens simples
    s1 = nltk.word_tokenize(sent1.lower())
    s2 = nltk.word_tokenize(sent2.lower())

    #etiqueta todas as palavras com suas respectivas classes gramaticais
    tagged1= nltk.pos_tag(s1)
    tagged2= nltk.pos_tag(s2)

    #Seleciona apenas os substantivos e verbos de cada sentença
    vb1 = [item[0] for item in tagged1 if 'VB' in item[1]]
    nn1 = [item[0] for item in tagged1 if 'NN' in item[1]]

    vb2 = [item[0] for item in tagged2 if 'VB' in item[1]]
    nn2 = [item[0] for item in tagged2 if 'NN' in item[1]]

    #Verifica a diferença de posição entre de elementos similares em cada sentença
    for i in range(len(vb1)):
        sim = []
        for j in range(len(vb2)):
            sim.append(tableSimilarity(word_vectors, model, vb1[i], vb2[j]))

        if len(sim) > 0:
            position = np.argmax(np.asarray(sim))
            if sim[position] > 0:
                dif += s1.index(vb1[i]) - s2.index(vb2[position])
                cont += 1

    for i in range(len(nn1)):
        sim = []
        for j in range(len(nn2)):
            sim.append(tableSimilarity(word_vectors, model, nn1[i], nn2[j]))

        if len(sim) > 0:
            position = np.argmax(np.asarray(sim))
            if sim[position] > 0:
                dif += (s1.index(nn1[i]) - s2.index(nn2[position]))
                cont += 1

    if dif > 0:
        similarity = float(abs(dif)) / (cont**2)

    return similarity


def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def normalizedLevenshtein(sentence1, sentence2):

    sim = levenshtein(sentence1, sentence2)
    similarity = 0
    if (len(sentence1) > 0) and (len(sentence2)) > 0:
        if(len(sentence1) > len(sentence2)):
            similarity = sim / len(sentence1)
        else:
            similarity = sim / len(sentence2)

    return similarity


def embeddingsSimilarity(model, sentence1, sentence2):

    words1 = converteStringToVector(sentence1)
    words2 = converteStringToVector(sentence2)

    size = 0
    if isinstance(words1, float) or isinstance(words2, float):
        return 0
    else:
        for l in range(len(words1)):
            if (words1[l].lower() in model.vocab):
                size = len(model[words1[l].lower()])

        sum1 = np.zeros(size)
        sum2 = np.zeros(size)

        for l in range(len(words1)):
            if (words1[l].lower() in model.vocab):
                temp = model[words1[l].lower()]
                if len(temp) == len(sum1):
                    sum1 = sumVectors(sum1, temp)

        for l in range(len(words2)):
            if (words2[l].lower() in model.vocab):
                temp = model[words2[l].lower()]
                if len(temp) == len(sum2):
                    sum2 = sumVectors(sum2, temp)

        cosineDistance = spatial.distance.cosine(sum1,sum2)

        return 1 - cosineDistance


def sumVectors(vector1, vector2):

    vectorSum = []

    for i in range(len(vector2)):
        vectorSum.append(vector1[i] + vector2[i])

    return vectorSum


def tableSimilarity(word_vectors, model, words1, words2):

    #words1 = converteStringToVector(sentence1)
    #words2 = converteStringToVector(sentence2)

    results = np.zeros(len(words1))
    table = np.zeros((len(words1), len(words2)))

    #if the words are present in the vocabulary of the word2vec model, calculate the similarity using word2vec, if not, Levenshtein's similarity is calculated
    #if (words1[l] in word_vectors.vocab) and (words2[j] in word_vectors.vocab):
    for l in range(len(words1)):
        for j in range(len(words2)):
            if (words1[l] in model.vocab) and (words2[j] in model.vocab):
                sim = word_vectors.similarity(words1[l], words2[j])
            else:
                sim = normalizedLevenshtein(words1[l], words2[j])
            table[l][j] = sim

    #removes row and column from highest similarity value in table
    for k in range(len(words1)):

        vector = getPositionMaxValueTable(table)
        results[k] = table[vector[0]][vector[1]]

        for m in range(len(table)):
            for n in range(len(table[0])):
                if m == vector[0]:
                    table[vector[0]][n] = -1.0
                if n == vector[1]:
                    table[m][vector[1]] = -1.0

    #returns the average of highest values in the table
    return np.mean(results)

def converteStringToVector(text_string):
    vector = []
    for word in text_string.split(' '):
        newWord = word.replace('[','').replace(']','').replace(',',':').replace('<','').replace('>','').replace(' ','').replace('!','').replace('?','').replace('-','').replace('#','').replace('@','').replace('.','')
        vector.append(newWord)
    return vector

def binarySimilarity( sentence1, sentence2):

    words1 = converteStringToVector(sentence1)
    words2 = converteStringToVector(sentence2)

    tam1 = len(words1)
    tam2 = len(words2)

    results = np.zeros(tam1)
    table = np.zeros((tam1, tam2))

    for l in range(tam1):
        for j in range(tam2):
            if (words1[l] == words2[j]):
                sim = 1.0
            else:
                sim = 0.0
            table[l][j] = sim

    # removes row and column from highest similarity value in table
    for k in range(tam1):

        vector = getPositionMaxValueTable(table)

        if table[vector[0]][vector[1]] != 0. and table[vector[0]][vector[1]] != -1.:
            results[k] = table[vector[0]][vector[1]]
        else:
            results[k] = 0.


        for m in range(len(table)):
            for n in range(len(table[0])):
                if m == vector[0]:
                    table[vector[0]][n] = -1.0
                if n == vector[1]:
                    table[m][vector[1]] = -1.0

    #retorna a média dos maiores valores da tabela
    return abs(np.mean(results))

def getPositionMaxValueTable(table):
    pos = []
    higherRow = 0
    higherColumn = 0
    higher =0
    vector = []

    for i in range (len(table)):
        for j in range(len(table[0])):
            if higher < table[i][j]:
                higher = table[i][j]
                higherRow = i
                higherColumn = j
    vector.append(higherRow)
    vector.append(higherColumn)
    return vector



####Testing method

# words1 = ["it", "is", "awesome"]
# words2 = ["today", "is","a", "beautiful", "day"]
#
# word_vectors = startModel()
# similarity = calculateSimilarity(word_vectors, words1, words2)
#
# print("similarity =" , similarity)
