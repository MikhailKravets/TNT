from core import load_imdb2mongo


if __name__ == '__main__':
    load_imdb2mongo(".data/aclImdb",
                    mongo_user="root",
                    mongo_password="password",
                    mongo_db="imdb")
