from __future__ import division
from __future__ import print_function
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt


# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):
    # DEFINE __init function
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(1, output_dim)
        self.params = {'W': self.W, 'b': self.b}

    def forward(self, x):
        return x @ self.W + b

    def backward(self, x):
        return x, np.ones((x.shape[0], 1)), self.W  # dw, db, dx


# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def forward(self, x):
        x[x < 0] = 0
        return x

    def backward(self, x):
        x[x > 0] = 1
        return x


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def __init__(self):
        self.factor = 1e-3

    def sigmoid(self, x):
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, x, labels):
        sigmoid = self.sigmoid(x)
        sigmoid[sigmoid <= self.factor] += self.factor
        sigmoid[sigmoid >= 1 - self.factor] -= self.factor

        y = -(labels * np.log(sigmoid) + (1 - labels) * np.log(1 - sigmoid))
        y_ = 1 / labels.shape[0] * y.sum()  # average loss
        return sigmoid, y_

    def backward(self, x, labels):
        return 1 / labels.shape[0] * (labels - self.sigmoid(x))


# This is a class for the Multilayer perceptron
class MLP(object):
    def __init__(self, input_dims, hidden_units):
        self.linear1 = LinearTransform(input_dims, hidden_units)
        self.relu1 = ReLU()
        self.linear2 = LinearTransform(hidden_units, 1)
        self.sigmoid_cross_entropy = SigmoidCrossEntropy()
        self.dict = dict()
        self.params = {'dw1': self.linear1.params['W'],
                       'db1': self.linear1.params['b'],
                       'dw2': self.linear2.params['W'],
                       'db2': self.linear2.params['b']}
        self.d_params = dict()

    def forward(self, x_batch, y_batch):
        z1 = self.linear1.forward(x_batch)
        r2 = self.relu1.forward(z1)
        z2 = self.linear2.forward(r2)
        out, loss = self.sigmoid_cross_entropy.forward(z2, y_batch)

        self.dict['z1'] = z1
        self.dict['r2'] = r2
        self.dict['z2'] = z2
        self.dict['loss'] = loss
        self.dict['labels'] = y_batch
        self.dict['x'] = x_batch
        return out, loss

    def backward(self):
        dz2 = self.sigmoid_cross_entropy.backward(self.dict['z2'], self.dict['labels'])
        dz2_dw2, dz2_db2, dz2_dr2 = self.linear2.backward(self.dict['r2'])
        dr2_dz1 = self.relu1.backward(self.dict['z1'])
        dz1_dw1, dz1_db1, _ = self.linear1.backward(self.dict['x'])

        dloss_dz2 = self.dict['loss'] * dz2
        dloss_dr2 = dloss_dz2 @ dz2_dr2.T
        dloss_dz1 = dloss_dr2 * dr2_dz1
        dloss_dw2 = dz2_dw2.T @ dloss_dz2
        dloss_db2 = dloss_dz2.T @ dz2_db2
        dloss_dw1 = dz1_dw1.T @ dloss_dz1
        dloss_db1 = dz1_db1.T @ dloss_dz1

        self.d_params['dw2'] = dloss_dw2
        self.d_params['dw1'] = dloss_dw1
        self.d_params['db2'] = dloss_db2
        self.d_params['db1'] = dloss_db1


def batch_evaluate(out, y):
    out[out >= 0.5] = 1
    out[out < 0.5] = 0
    correct = int(sum(out == y))
    return correct


class Optimizer(object):
    def __init__(self, params, d_params, learning_rate=1e-2, momentum=None, l2_penalty=0):
        self.params = params
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_penalty = l2_penalty
        self.d_params = d_params
        if momentum:
            self.D = dict()

    def zero_grad(self):
        for k, v in self.d_params.items():
            self.d_params[k] = np.zeros_like(v)

    def step(self):
        for var in self.params.keys():
            if momentum:
                if var not in self.D:
                    self.D[var] = self.learning_rate * self.d_params[var]
                else:
                    self.D[var] = self.momentum * self.D[var] + self.learning_rate * self.d_params[var]

                self.params[var] += self.D[var]
            else:
                self.params[var] += learning_rate * self.d_params[var]


if __name__ == '__main__':
    with open('cifar_2class_py2.p', 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    train_x = data['train_data'].astype('float64')
    train_y = data['train_labels']
    test_x = data['test_data'].astype('float64')
    test_y = data['test_labels']

    mean = train_x.mean()
    train_x -= mean
    std = train_x.std()
    train_x /= std

    mean = test_x.mean()
    test_x -= mean
    std = test_x.std()
    test_x /= std

    num_examples, input_dims = train_x.shape
    test_num_examples, test_input_dims = test_x.shape

    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    batch_size = 10  # 400
    hidden_units = 10  # 200
    learning_rate = 1e-2
    momentum = .8
    l2_penalty = 0

    mlp = MLP(input_dims, hidden_units)
    sgd = Optimizer(mlp.params, mlp.d_params, learning_rate, momentum, l2_penalty)

    # batch_size_list = list([10, 100, 300, 400, 500])
    # learning_rate_list = list([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    # hidden_units_list = list([50, 100, 150, 200, 250])

    # m, n, p = len(batch_size_list), len(learning_rate_list), len(hidden_units_list)

    # plt.figure()
    # plt.title('Test accuracy with different number of batch size\n')
    # plt.xlabel('Batch_Size')
    # plt.ylabel('Accuracy (%)')

    # test_accuracies = np.zeros((m, n, p))

    # history_test_accuracy_for_batch = list()

    # for h, batch_size in enumerate(batch_size_list):
    # for j, learning_rate in enumerate(learning_rate_list):
    # for k, hidden_units in enumerate(hidden_units_list):

    # history_test_accuracy = list()
    # location_epochs = list()

    for epoch in range(num_epochs):

        train_loss = 0
        train_correct = 0
        test_loss = 0
        test_correct = 0
        # train
        # INSERT YOUR CODE FOR EACH EPOCH HERE
        for b, i in enumerate(range(0, num_examples, batch_size)):
            x = train_x[i:i + batch_size].astype('float128')
            y = train_y[i:i + batch_size]

            out, loss = mlp.forward(x, y)

            sgd.zero_grad()
            mlp.backward()
            sgd.step()
            correct = batch_evaluate(out, y)

            train_correct += correct
            train_loss += loss

            # print('{} / {}: current loss: {:.3f}, current accuracy: {:.2f}'.format(i, num_examples,
            #                                                                        train_loss / (
            #                                                                        (b + 1) * batch_size),
            #                                                                        train_correct / (
            #                                                                        (b + 1) * batch_size)))

            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            # MAKE SURE TO UPDATE total_loss

        train_accuracy = train_correct / num_examples
        train_loss = train_loss / num_examples

        print('\r[Epoch {}] Train_avg_loss={:.3f}, Train_accuracy={:.2f}]'.format(epoch + 1, train_loss,
                                                                                  train_accuracy * 100))

        # test
        # INSERT YOUR CODE FOR EACH EPOCH HERE
        for b, i in enumerate(range(0, test_num_examples, batch_size)):
            test_x_temp = test_x[i:i + batch_size].astype('float128')
            test_y_temp = test_y[i:i + batch_size]

            test_out, test_loss_temp = mlp.forward(test_x_temp, test_y_temp)
            test_correct_temp = batch_evaluate(test_out, test_y_temp)

            test_correct += test_correct_temp
            test_loss += test_loss_temp

        test_accuracy = test_correct / test_num_examples
        test_loss = test_loss / test_num_examples

        print('\r[Epoch {}] Test_avg_loss={:.3f}, Test_accuracy={:.2f}]'.format(epoch + 1, test_loss,
                                                                                test_accuracy * 100))

        # history_test_accuracy.append(test_accuracy * 100)
        # location_epochs.append(epoch)

        # history_test_accuracy_for_batch.append(test_accuracy * 100)

    #     test_accuracies[h, j, k] = test_accuracy
    #     print('test_accuracies', test_accuracies)
    #
    # with open('result.pkl', 'wb') as f:
    #     pickle.dump(test_accuracies, f, pickle.HIGHEST_PROTOCOL)

    # plt.plot(batch_size_list, history_test_accuracy_for_batch)
    # plt.show()
