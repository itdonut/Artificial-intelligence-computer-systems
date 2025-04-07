import numpy as np
from neural_network import NeuralNetwork
from sklearn.neural_network import MLPRegressor

class Perceptron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def activate(self, x):
        return 1 if x > 0 else 0

    def predict(self, inputs):
        total_input = np.dot(self.weights, inputs) + self.bias
        return self.activate(total_input)

def xor_modeling(x1, x2):
    not_x1 = not_perceptron.predict([x1, 0])
    not_x2 = not_perceptron.predict([x2, 0])
    and_result1 = and_perceptron.predict([x1, not_x2])
    and_result2 = and_perceptron.predict([not_x1, x2])
    or_result = or_perceptron.predict([and_result1, and_result2])
    return or_result

def xor_model(x1, x2):
    weights_1 = [1, -1]
    weights_2 = [-1, 1]
    weights_3 = [1,  1]
    bias = -0.5

    perceptron = Perceptron(weights_1, bias)
    perceptron_2 = Perceptron(weights_2, bias)
    perceptron_3 = Perceptron(weights_3, bias)

    y1 = perceptron.predict([x1, x2])
    y2 = perceptron_2.predict([x1, x2])
    return perceptron_3.predict([y1, y2])

# Define weights and bias for AND function
and_weights = [1, 1]
and_bias = -1.5

# Define weights and bias for OR function
or_weights = [1, 1]
or_bias = -0.5

# Define weights and bias for NOT function
not_weights = [-1.5, 0]
not_bias = 1

and_perceptron = Perceptron(and_weights, and_bias)
or_perceptron  = Perceptron(or_weights, or_bias)
not_perceptron = Perceptron(not_weights, not_bias)

test_inputs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

time_row_train_inputs = np.array([
    [0.87, 4.12, 0.93],
    [4.12, 0.93, 4.62],
    [0.93, 4.62, 1.51],
    [4.62, 1.51, 5.76],
    [1.51, 5.76, 0.50],
    [5.76, 0.50, 5.48],
    [0.50, 5.48, 0.95],
    [5.48, 0.95, 4.03],
    [0.95, 4.03, 0.92],
    [4.03, 0.92, 5.15]
])

time_row_inputs = np.array([
    [0.92, 5.15, 1.66],
    [5.15, 1.66, 5.01]
])

# Define the expected outputs
time_row_train_expected_output = np.array([[4.62], [1.51], [5.76], [0.50], [5.48], [0.95], [4.03], [0.92], [5.15], [1.66]])
time_row_expected_output = np.array([[5.01], [0.40]])

# Define the truth table inputs
truth_table_inputs = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

# Define the expected outputs
expected_output = np.array([[1], [1], [0], [1]])

nn = NeuralNetwork()

def print_title(title):
    print()
    print("=" * 100)
    print(" " * 8 + title)
    print("=" * 100)

print_title("AND Function")
print("        x1 x2")
for inputs in test_inputs:
    output = and_perceptron.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")

print_title("OR Function")
print("        x1 x2")
for inputs in test_inputs:
    output = or_perceptron.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")

print_title("NOT Function")
print("        x1 x2")
for inputs in test_inputs:
    output = not_perceptron.predict(inputs)
    print(f"Input: {inputs}, Output: {output}")

print_title("XOR Function")
print("        x1 x2")
for inputs in test_inputs:
    output = xor_modeling(inputs[0], inputs[1])
    print(f"Input: {inputs}, Output: {output}")

print_title("XOR Function")
print("        x1 x2")
for inputs in test_inputs:
    output = xor_model(inputs[0], inputs[1])
    print(f"Input: {inputs}, Output: {output}")

learning_rate=0.01 # the best learning rate is 0.01
epochs=1000000

nn.train(truth_table_inputs, expected_output, learning_rate=learning_rate, epochs=epochs)
print_title(f"Neural Network <lr: {learning_rate}, epochs: {epochs}>")
for input_data in truth_table_inputs:
    output = nn.forward(input_data)
    print(f"Input: {input_data}, Output: [{output[0]:.8f}]")

nn.use_linear_activation = True
nn.train(time_row_train_inputs, time_row_train_expected_output, learning_rate=learning_rate, epochs=epochs) 
print_title(f"Neural Network (Time row) <lr: {learning_rate}, epochs: {epochs}>")
for input_data, expected in zip(time_row_inputs, time_row_expected_output):
    output = nn.forward(input_data)
    print(f"Input: {input_data}, Output: [{np.abs(output[0]):.8f}], Expected: [{expected[0]:.2f}], Error: [{(np.abs(expected[0] - output[0]) / expected[0] * 100):012.8f}]%")

nn = MLPRegressor(hidden_layer_sizes=(100), activation='logistic', solver='lbfgs', max_iter=epochs)
nn.fit(time_row_train_inputs, time_row_train_expected_output[:, 0])
predicted_output = nn.predict(time_row_inputs)
print_title(f"MLPRegressor Neural Network (Time row) <lr: 0.0001, epochs: {epochs}>")
for input_data, expected, predicted in zip(time_row_inputs, time_row_expected_output[:, 0], predicted_output):
    print(f"Input: {input_data}, Expected: [{expected:.2f}], Predicted: [{predicted:.8f}], Error: [{abs(expected - predicted) / expected * 100:012.8f}]%")