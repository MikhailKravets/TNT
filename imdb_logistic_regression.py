from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from core import load_imdb2pandas

if __name__ == '__main__':
    training, test = load_imdb2pandas(user="root",
                                      password="password",
                                      db="imdb",
                                      collection="reviews")

    vectorizer = CountVectorizer(stop_words="english",
                                 max_features=10000)
    training_features = vectorizer.fit_transform(training['text'])

    model = LogisticRegression()
    model.fit(training_features, training['sentiment'])

    test_pred = model.predict(vectorizer.transform(test['text']))
    accuracy = accuracy_score(test_pred, test['sentiment'])
    print(f"Accuracy: {accuracy}")
