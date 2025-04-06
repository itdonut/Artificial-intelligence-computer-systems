import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.use_linear_activation: bool = False
        
        self.input_size = 3
        self.hidden_size = 4
        self.output_size = 1

        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.random.rand(self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.random.rand(self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return np.ones_like(x)

    def forward(self, inputs):
        self.hidden_layer_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        if self.use_linear_activation:
            self.output_layer_output = self.output_layer_input
        else:
            self.output_layer_output = self.sigmoid(self.output_layer_input)

        return self.output_layer_output

    def backward(self, inputs, expected_output, learning_rate):
        # Calculate error
        output_error = expected_output - self.output_layer_output
        if self.use_linear_activation:
            output_delta = output_error
        else:
            output_delta = output_error * self.sigmoid_derivative(self.output_layer_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate
    
    def train(self, inputs, expected_output, learning_rate, epochs):
        for _ in range(epochs):
            self.forward(inputs)
            self.backward(inputs, expected_output, learning_rate)