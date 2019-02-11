import MethodsNLP as mnlp, ReadingXML as rxml, Tfidf, Tep
from scipy import spatial
import Word2Vec
import math
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
        textDataSet.append(line[3])
        textDataSet.append(line[4])

    #add synonyms
    newDataSet = Tep.addSynonyms(textDataSet)
    finalDataSet = []

    #stemming
    for line in newDataSet:
        text = mnlp.stemming(line)
        finalDataSet.append(mnlp.convertText(text))

    vector = Tfidf.calculateTFIDF(finalDataSet)

    similarities = []

    for i in range(0, len(vector), 2):
        distance = spatial.distance.cosine(vector[i], vector[i + 1])
        similarities.append(1 - distance)

    #calculate others features
    word_vectors, model = Word2Vec.startModel()
    features = []

    for x in range(len(dataSet)):

        featuresLine = []

        #calculate feature 2
        feature2 = Word2Vec.wordOrderSimilarity(word_vectors, dataSet[x][3] , dataSet[x][4])

        #calculate feature 3
        sim2 = Word2Vec.embeddingsSimilarity(model, dataSet[x][3], dataSet[x][4])
        if math.isnan(sim2):
            feature3 = 1.0
        else:
            feature3 = sim2

        #calculate feature 4
        sim3 = Word2Vec.calculateSimilarity(word_vectors, dataSet[x][3], dataSet[x][4])
        if math.isnan(sim3):
            feature4 = 1.0
        else:
            feature4 = sim3

        # calculate feature 5
        feature5 = Word2Vec.binarySimilarity(dataSet[x][3], dataSet[x][4])

        # calculate feature 6
        size1 = len(mnlp.tokenize(dataset[x][3]))
        size2 = len(mnlp.tokenize(dataset[x][4]))

        if( size1 > size2):
            feature6 = size2/size1
        else:
            feature6 = size1/size2

        featuresLine.append(similarities[x])
        featuresLine.append(feature2)
        featuresLine.append(feature3)
        featuresLine.append(feature4)
        featuresLine.append(feature5)
        featuresLine.append(feature6)
        featuresLine.append(dataSet[x][2]) #similarity class

        print(featuresLine)
        features.append(featuresLine)



dataset = rxml.readDataSet("assin\\assin-ptbr-train2.xml")

featuresExtraction(dataset)