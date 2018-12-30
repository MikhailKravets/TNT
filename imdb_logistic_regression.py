from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from core import load_imdb2pandas

if __name__ == '__main__':
    df = load_imdb2pandas(user="root",
                          password="password",
                          db="imdb",
                          collection="reviews")

    vectorizer = CountVectorizer(stop_words="english",
                                 max_features=10000)
    training_features = vectorizer.fit_transform(df['text'])

    model = LogisticRegression()
    model.fit(training_features, df['sentiment'])

