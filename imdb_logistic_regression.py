from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from matplotlib import pyplot as pl

from core import load_imdb2pandas

if __name__ == '__main__':
    training, test = load_imdb2pandas(user="root",
                                      password="password",
                                      db="imdb",
                                      collection="reviews")

    vectorizer = TfidfVectorizer(stop_words="english",
                                 max_features=10000)
    training_features = vectorizer.fit_transform(training['text'])

    model = LogisticRegression()
    model.fit(training_features, training['sentiment'])

    test_pred = model.predict(vectorizer.transform(test['text']))
    accuracy = accuracy_score(test_pred, test['sentiment'])

    pca = TruncatedSVD(n_components=2)
    tf2plot = pca.fit_transform(training_features)

    # pl.plot(tf2plot[:, 0], tf2plot[:, 1], lw=0, ms=3, marker='.')
    # pl.show()

    print(f"Accuracy: {accuracy}")
