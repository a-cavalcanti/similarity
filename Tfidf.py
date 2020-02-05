from sklearn.feature_extraction.text import TfidfVectorizer

def calculateTFIDF(texts):

    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(texts)

    return response.toarray()

