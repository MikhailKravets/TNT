import pandas
from keras_preprocessing.text import Tokenizer


def convert_data():
    with open('.data/fra.txt', 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        inp, target, _ = line.split('\t')
        data.append((inp, target))

    data = pandas.DataFrame(data, columns=['English', 'French'])
    data.to_csv('.data/fra_cleared.csv', index=False)
    return data


if __name__ == '__main__':
    data = pandas.read_csv('.data/fra_cleared.csv')

    eng_tokenizer, french_tokenizer = Tokenizer(), Tokenizer()
    eng_tokenizer.fit_on_texts(data['English'])
    french_tokenizer.fit_on_texts(data['French'])
