#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils

plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')

def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features)) #10 lines (classes) by 785 columns (features)

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):  #x_i is a line of 785 features
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
    
        y_hat = self.predict(x_i)
        
        if y_hat != y_i:

            self.W[y_i] += x_i
            self.W[y_hat] -= x_i

        # Q1.1a


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """

        def ey_C(i): #return one-hot encoded vector
            ret = np.zeros(self.W.shape[0])
            ret[i] = 1
            return ret
        
        scores = self.W @ x_i
        scores -= np.max(scores)

        probs = np.exp(scores)
        Z = np.sum(probs)
        probs = probs / Z

        gradient = np.outer(probs - ey_C(y_i), x_i)

        self.W = self.W - learning_rate * gradient

        #Q1.1b


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.W1 = np.random.normal(loc = 0.1, scale = 0.1, size = (hidden_size, n_features))
        self.W2 = np.random.normal(loc = 0.1, scale = 0.1, size = (n_classes, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(n_classes)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        z_1 = self.W1 @ X.T + self.b1.reshape((self.b1.shape[0], 1))
        h_1 = z_1 * (z_1 > 0)

        z_2 = self.W2 @ h_1 + self.b2.reshape((self.b2.shape[0], 1))
        z_2 -= np.max(z_2, axis = 0)
        probs = np.exp(z_2)
        f = probs / np.sum(probs)
        return np.argmax(f, axis = 0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):

        for i in range(X.shape[0]):

            #forward propagation
            x = X[i]
            z_1 = self.W1 @ x + self.b1
            h_1 = z_1 * (z_1 > 0)

            z_2 = self.W2 @ h_1 + self.b2
            z_2 -= np.max(z_2)
            probs = np.exp(z_2)
            f = probs / np.sum(probs)

            #backwards propagation
            output = np.zeros(self.W2.shape[0]) #one-hot encoding
            output[y[i]] = 1

            grad_z_2 = - (output - f) 
            grad_W_2 = np.outer(grad_z_2, h_1)
            grad_b_2 = grad_z_2  #gradient for the bias weights

            grad_h_1 = self.W2.T @ grad_z_2
            grad_z_1 = grad_h_1 * (z_1 > 0)
            
            grad_W_1 = np.outer(grad_z_1, x)
            grad_b_1 = grad_z_1

            self.W2 -= learning_rate * grad_W_2
            self.b2 -= learning_rate * grad_b_2

            self.W1 -= learning_rate * grad_W_1
            self.b1 -= learning_rate * grad_b_1


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)
    print(f"Final validation accuracy: {valid_accs[-1]}")
    print(f"Final test accuracy: {test_accs[-1]}")


if __name__ == '__main__':
    main()
