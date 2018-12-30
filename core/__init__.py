import os
import pandas as pd
import numpy as np

import tqdm
from pymongo import MongoClient
from pymongo.collection import Collection


def load_imdb2mongo(data_dir, mongo_user, mongo_password, mongo_db):
    """Loads the IMDB train/test datasets from a folder path
    to the mongo database
    """
    client = MongoClient('localhost', 27017,
                         username=mongo_user,
                         password=mongo_password)
    db = client.imdb
    collection: Collection = db.reviews

    for split in "train", "test":
        for sentiment in "neg", "pos":
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in tqdm.tqdm(file_names):
                with open(os.path.join(path, f_name), "r") as f:
                    try:
                        review = f.read()
                        collection.insert_one({
                            'review': review,
                            'score': score
                        })

                    except UnicodeDecodeError:
                        print(f"{f_name} wasn't decoded")


def load_imdb2pandas(user, password, db, collection, test_part=.15):
    client = MongoClient('localhost', 27017,
                         username=user,
                         password=password)
    db = client[db]
    collection: Collection = db[collection]

    data = []
    for v in collection.find():
        data.append([v['review'], v['score']])

    count = len(data)
    test_count = int(test_part * count)
    np.random.shuffle(data)

    test, training = data[0:test_count], data[test_count:]
    training = pd.DataFrame(training, columns=['text', 'sentiment'])
    test = pd.DataFrame(test, columns=['text', 'sentiment'])
    return training, test
