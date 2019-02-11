import gensim
import numpy as np
from scipy import spatial
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import ReadingXML

# list of text documents
#  text = ["women women women ball",
# 		"women women call",
# 		"women women",
#         "ball ball"]

text = []
text.append("women women women ball")
text.append("women women call")
text.append("women women")
text.append("ball ball")

# create the transform
vectorizer = TfidfVectorizer()

# tokenize and build vocab
vectorizer.fit(text)

# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)

# encode document

print(vectorizer.transform([text[0]]).toarray())
print(vectorizer.transform([text[1]]).toarray())
print(vectorizer.transform([text[2]]).toarray())
print(vectorizer.transform([text[3]]).toarray())

response = vectorizer.fit_transform(text)
print(response.toarray())


dataset = ReadingXML.readDataSet("assin\\assin-ptbr-train2.xml")
#print(dataset)

textDataSet = []

# for line in dataset:
#     text1 = stemming(line[3])
#     text2 = stemming(line[4])
#     textDataSet.append(convertText(text1))
#     textDataSet.append(convertText(text2))
#
# print(textDataSet)