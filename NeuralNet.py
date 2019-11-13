#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class NeuralNet:
    def __init__(self, train, iscsv ,header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers
        if iscsv == 'local_csv':
            raw_input = pd.read_csv(train)
        else:
            raw_input=train
        # print(raw_input.isnull().sum())
        # print(raw_input.head(5))
        # TODO: Remember to implement the preprocess method
        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        elif activation == "tanh":
            self.__tanh(self, x)
        elif activation == "relu":
            self.__relu(self, x)
    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        elif activation == "tanh":
            self.__tanh_derivative(self, x)
        elif activation == "relu":
            self.__relu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __relu(self, x):
        return np.maximum(x,0)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return (1 + x) * (1 - x)

    def __relu_derivative(self, x):
        return (x > 0).astype(int)

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        X=X.dropna()
        labelencoder = LabelEncoder()
        for column in X:
            if X[column].dtype == 'object':
                X[column] = labelencoder.fit_transform(X[column])

        scaled_features = StandardScaler().fit_transform(X.values)
        scaled_features_df = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)
        return scaled_features_df

    # Below is the training function

    def train(self, activation="sigmoid",max_iterations = 1000, learning_rate = 0.05):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        # print("The final weight vectors are (starting from input to output layers)")
        # print(self.w01)
        # print(self.w12)
        # print(self.w23)

    def forward_pass(self,activation="sigmoid"):
        # pass our inputs through our neural network
        if activation=="sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        elif activation=="tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        elif activation=="relu":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
        return out



    def backward_pass(self, out, activation="sigmoid"):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        elif activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        elif activation == "relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__relu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation="sigmoid"):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        elif activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        elif activation == "relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__relu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test,iscsv,activation="sigmoid", header = True):
        if iscsv == 'local_csv':
            test_dataset1 = pd.read_csv(test)
        else:
            test_dataset1 = test
        test_dataset = self.preprocess(test_dataset1);
        #test_dataset = pd.DataFrame(test, index=test.index, columns=test.columns)
        nc = len(test_dataset.columns)
        nr = len(test_dataset.index)
        self.X = test_dataset.iloc[:, 0:(nc - 1)].values.reshape(nr, nc - 1)
        self.Y = test_dataset.iloc[:, (nc - 1)].values.reshape(nr, 1)
        pred = self.forward_pass(activation)
        error = 0.5 * np.power((pred - self.Y), 2)
        print("Testing error : " + str(np.sum(error)))
        return np.sum(error)

if __name__ == "__main__":

    list_activation = ['sigmoid','tanh','relu']

    for l in list_activation:
        print(l)
        print("-------------------------------------------------\n")
        neural_network = NeuralNet("train.csv","local_csv")
        neural_network.train(l)
        testError = neural_network.predict("test.csv","local_csv",l)
        print("-------------------------------------------------\n")
    # dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/bridges/bridges.data.version1")
    # dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/bridges/bridges.data.version1" )
    # dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
    # training = dataset.sample(frac=0.75,random_state=0)
    # testing = dataset.drop(training.index)
    # msk = np.random.rand(len(dataset)) < 0.8
    # train = dataset[msk]
    # test = dataset[~msk]
    # print(len(train))
    # print(len(test))
    for l in list_activation:
        dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
        training = dataset.sample(frac=0.75, random_state=0)
        testing = dataset.drop(training.index)
        print(l)
        print("-------------------------------------------------\n")
        neural_network = NeuralNet(training,"not_local_csv")
        neural_network.train(l)
        testError = neural_network.predict(testing,"not_local_csv", l)
        print("-------------------------------------------------\n")
    # print(testError)