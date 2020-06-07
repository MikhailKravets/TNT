import math
import random

import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(out):
    return out * (1 - out)


def bin2dec(b):
    out = 0
    for i, x in enumerate(b[::-1]):
        out += x * pow(2, i)

    return out


if __name__ == '__main__':
    BIN_DIM = 20
    INPUT_DIM = 1
    HIDDEN_DIM = 50
    OUTPUT_DIM = 1

    LEN = 200

    ALPHA = 0.0001
    ITER_NUM = 50000
    LOG_ITER = 100
    PLOT_ITER = ITER_NUM // 200

    space = np.linspace(0, 20, LEN)
    sin_data = np.sin(space)

    w0 = np.random.normal(0, 1, (INPUT_DIM, HIDDEN_DIM))
    wh = np.random.normal(0, 2, (HIDDEN_DIM, HIDDEN_DIM))
    w1 = np.random.normal(0, 1, (HIDDEN_DIM, OUTPUT_DIM))

    # delta values
    d0 = np.zeros_like(w0)
    d1 = np.zeros_like(w1)
    dh = np.zeros_like(wh)

    errs = []
    accs = []

    error = 0
    accuracy = 0

    for i in range(ITER_NUM + 1):
        # a + b = c
        rand_i = random.choice([0, LEN - BIN_DIM - 2])
        inp = sin_data[rand_i:rand_i + BIN_DIM + 1]
        true = sin_data[rand_i + BIN_DIM + 1:rand_i + BIN_DIM + 2]
        pred = sin_data[rand_i + BIN_DIM + 1:rand_i + BIN_DIM + 2]

        overall_err = 0  # total error in the whole calculation process.

        output_deltas = []
        hidden_values = [np.zeros(HIDDEN_DIM)]

        future_delta = np.zeros(HIDDEN_DIM)

        # forward propagation
        for pos in range(BIN_DIM):
            X = np.array([inp[pos]])  # shape=(1, 2)
            Y = np.array([inp[pos + 1]])  # shape=(1, 1)

            hidden = sigmoid(np.dot(X, w0) + np.dot(hidden_values[-1], wh))
            output = sigmoid(np.dot(hidden, w1))

            # pred[pos] = output[0]

            # squared mean error
            hidden_values.append(hidden)

        output_err = true - output
        output_deltas.append(output_err * deriv_sigmoid(output))
        overall_err += np.abs(output_err[0])

        # backpropagation through time
        for pos in range(BIN_DIM - 1, -1, -1):
            X = np.array([inp[pos]])

            hidden = hidden_values[pos]
            prev_hidden = hidden_values[pos + 1]

            output_delta = output_deltas
            hidden_delta = (np.dot(future_delta, wh.T) + np.dot(output_delta, w1.T)) * deriv_sigmoid(hidden)

            d1 += np.dot(np.atleast_2d(hidden).T, np.atleast_2d(output_delta))
            dh += np.dot(np.atleast_2d(prev_hidden).T, np.atleast_2d(hidden_delta))
            d0 += np.dot(X.T, np.atleast_2d(hidden_delta))

            future_delta = hidden_delta

        w1 += ALPHA * d1
        w0 += ALPHA * d0
        wh += ALPHA * dh

        d1 *= 0
        d0 *= 0
        dh *= 0

        error += overall_err
        # if bin2dec(pred) == c_dec:
        #     accuracy += 1
        #
        # if i % PLOT_ITER == 0:
        #     errs.append(error / PLOT_ITER)
        #     accs.append(accuracy / PLOT_ITER)
        #
        #     error = 0
        #     accuracy = 0

        if i % LOG_ITER == 0:
            print('Iter', i)
            print("Error :", overall_err)
            print("Pred :", pred)
            print("True :", true)
            print('----------')
