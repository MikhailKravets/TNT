from core import load_imdb2pandas

if __name__ == '__main__':
    df = load_imdb2pandas(user="root",
                          password="password",
                          db="imdb",
                          collection="reviews")
