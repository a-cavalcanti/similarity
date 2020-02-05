import MethodsNLP as mnlp, ReadingXML as rxml, Tfidf, Tep
from scipy import spatial
import Word2Vec
import math
import pandas as pd
import re


def saveFile(features, file, cols):

    arq = open(file + ".csv", 'w')
    arq.write(str(cols) + "\n")
    for i in range(len(features)):
        for j in range(len(features[i])):
            arq.write(features[i][j])
            if i != len(features[i]) - 1:
                arq.write(",")
        arq.write("\n")
    arq.close()

def featuresExtraction(dataSet):

    #calculate feature 1 - TFIDF
    textDataSet = []

    for line in dataset:
        textDataSet.append(line[0])
        textDataSet.append(line[1])

    #add synonyms
    newDataSet = Tep.addSynonyms(textDataSet)
    finalDataSet = []

    #stemming
    for line in newDataSet:
        text = mnlp.stemming(line)
        finalDataSet.append(mnlp.convertText(text))


    #obtendo os vetores TF-IDF de cada frase
    vector = Tfidf.calculateTFIDF(finalDataSet)

    similarities = []

    """aqui calculamos a distância do cosseno entre a frase 1 e a frase 2, ou seja, entre os pares de frases
       esse vector vai ter os vetores tf-idf de cada frase, no caso, é como se na posição 0 estivesse a frase 1, 
       na posição 1 estivesse a frase 2, na posição 2 estivesse a frase 3... e assim por diante
       Então se queremos calcular a similaridade entre a frase 1 e a frase 2 do nosso banco, devemos calcular
       a distância do cosseno entre vector[0] e vector[1]
       Por isso o for abaixo intera de 2 em 2 --> range(0, len(vector), 2)
    """

    for i in range(0, len(vector), 2):
        distance = spatial.distance.cosine(vector[i], vector[i + 1])
        similarities.append(1 - distance)

    #calculando as outras features
    #iniciando o modelo do word2vec
    word_vectors, model = Word2Vec.startModel()
    features = []

    #para cada linha do meu csv, vou calcular a similaridade utilizando esses métodos a seguir:
    for x in range(len(dataSet)):

        featuresLine = []

        """calculando a feature 2 entre a coluna 0 e coluna 1 do meu csv
            esse método obtive de um trabalho da literatura, vou te passar o pdf dele também
        """
        feature2 = Word2Vec.wordOrderSimilarity(word_vectors, model, dataSet[x][0] , dataSet[x][1])

        """A feature 3 é distância do cosseno entre os vetores de cada frase, ou seja, 
        o vetor de cada frase é a soma dos vetores de embeddings de cada palavra"""

        sim2 = Word2Vec.embeddingsSimilarity(model, dataSet[x][0], dataSet[x][1])
        if math.isnan(sim2):
            feature3 = 1.0
        else:
            feature3 = sim2

        """A feature 4 utiliza uma matriz de similaridades com tamanho: 
        numero de palavras da frase 1 X numero de palavras da frase 2 
        Esse é o método que utilizei na minha dissertação. O word2vec aqui foi utilizado para calcular a similaridade
        entre as palavras. E a similaridade entre as frases é obtida utilizando esse método da matriz"""

        sim3 = Word2Vec.calculateSimilarity(word_vectors, model, dataSet[x][0], dataSet[x][1])
        if math.isnan(sim3):
            feature4 = 1.0
        else:
            feature4 = sim3

        """Esse feature utiliza a mesma matriz da feature acima, só que no lugar de calcular a similaridade entre
        as palavras usando word2vec, nós usamos uma abordagem binária. Se as palavras forem iguais, a similaridade entre
        elas será 1, se forem diferentes a similaridade entre elas será 0"""

        feature5 = Word2Vec.binarySimilarity(dataSet[x][0], dataSet[x][1])

        """A feature 6 será o tamanho da frase menor dividido pelo tamanho da frase maior"""
        size1 = len(mnlp.tokenize(dataset[x][0]))
        size2 = len(mnlp.tokenize(dataset[x][1]))

        if( size1 > size2):
            feature6 = size2/size1
        else:
            feature6 = size1/size2



        #salvo um aquivo com as features extraídas e a classe a qual elas pertencem
        featuresLine.append(similarities[x])
        featuresLine.append(feature2)
        featuresLine.append(feature3)
        featuresLine.append(feature4)
        featuresLine.append(feature5)
        featuresLine.append(feature6)
        featuresLine.append(dataSet[x][2]) #similarity class

        #print(featuresLine)
        features.append(featuresLine)

        #imprimindo o valor de similaridade obtido, combinando as features
        similaridade =  (0.3*similarities[x]) + (0.1*feature2) + (0.2 * feature3) + (0.2 * feature4) + (0.1 * feature5) + (0.1 * feature6)
        print(similaridade)

#lendo o banco com os textos
#dataset = rxml.readDataSet("assin\\assin-ptbr-train2.xml")

dados = open('DadosProcessados.csv', 'r', encoding='utf-8', errors='ignore').read().split('\n')

dataset = []

for line in dados:
    wordsBase = line.split(',')
    if (wordsBase != []):
        dataset.append(wordsBase)

featuresExtraction(dataset)