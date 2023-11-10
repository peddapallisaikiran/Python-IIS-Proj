import numpy as np

def calculate_probabilities(parameters, data, utilities):
    # Extracting independent variables from data
    X = np.array([data[key] for key in data.keys()])

    # Calculating deterministic utilities for each alternative
    utilities_values = np.array([utility(parameters, X) for utility in utilities])

    # Calculating probabilities using the multinomial logit model
    exp_utilities = np.exp(utilities_values)
    probabilities = exp_utilities / np.sum(exp_utilities, axis=0)

    # Creating a dictionary to store results
    result = {f'P{i+1}': probabilities[i].tolist() for i in range(len(probabilities))}

    return result

# Sample data
data = {
    'X1': [2, 3, 5, 7, 1, 8, 4, 5, 6, 7],
    'X2': [1, 5, 3, 8, 2, 7, 5, 9, 4, 2],
    'Sero': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

# Parameters
parameters = {'beta01': 0.1, 'beta1': 0.5, 'beta2': 0.5, 'beta02': 1, 'beta03': 0}

# Utilities functions
def utility1(parameters, X):
    return parameters['beta01'] + parameters['beta1']*X[0] + parameters['beta2']*X[1]

def utility2(parameters, X):
    return parameters['beta02'] + parameters['beta1']*X[0] + parameters['beta2']*X[1]

def utility3(parameters, X):
    return parameters['beta03'] + parameters['beta1']*X[2] + parameters['beta2']*X[2]

# List of utility functions
utilities_list = [utility1, utility2, utility3]

# Calculating probabilities
probabilities_result = calculate_probabilities(parameters, data, utilities_list)

# Save the result to a text file
with open('output.txt', 'w') as file:
    file.write(str(probabilities_result))