import Word2Vec

####Testing method

words1 = "Foi um ótimo dia!"
words2 = "Hoje está um dia lindo!"

#iniciando o modelo de embeddings
word_vectors, model = Word2Vec.startModel()

similaridadeMatrizWord2vec = Word2Vec.calculateSimilarity(word_vectors, model, words1, words2)
similaridadeVetoresEmbeddings = Word2Vec.embeddingsSimilarity( model, words1, words2)
similaridadeWordOrder = Word2Vec.wordOrderSimilarity(word_vectors, model, words1, words2)
similaridadeBinaria = Word2Vec.binarySimilarity( words1, words2)

print("similaridade Matriz Word2vec = " , similaridadeMatrizWord2vec)
print("similaridade Vetores Embeddings = " , similaridadeVetoresEmbeddings)
print("similaridade Word Order = ", similaridadeWordOrder)
print("similaridade binaria = ", similaridadeBinaria )


"""TESTANDO COM A BASE DE DADOS"""

dados = open('DadosProcessados.csv', 'r', encoding='utf-8', errors='ignore').read().split('\n')

dataset = []

#salvando os dados da base em dataset
for line in dados:
    wordsBase = line.split(',')
    if (wordsBase != []):
        dataset.append(wordsBase)

for x in range(len(dataset)):
    similaridadeMatrizWord2vec = Word2Vec.calculateSimilarity(word_vectors, model, dataset[x][0], dataset[x][1])
    similaridadeVetoresEmbeddings = Word2Vec.embeddingsSimilarity(model, dataset[x][0], dataset[x][1])
    similaridadeWordOrder = Word2Vec.wordOrderSimilarity(word_vectors, model, dataset[x][0], dataset[x][1])
    similaridadeBinaria = Word2Vec.binarySimilarity(dataset[x][0], dataset[x][1])

    print("sim_W2V linha "+ str(x) + " = ", similaridadeMatrizWord2vec)
    print("sim_EMB linha "+ str(x) + " = ", similaridadeVetoresEmbeddings)
    print("sim_WOR linha "+ str(x) + " = ", similaridadeWordOrder)
    print("sim_BIN linha "+ str(x) + " = ", similaridadeBinaria)
    print("\n")


