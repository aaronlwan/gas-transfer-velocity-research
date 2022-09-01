from mlfg_final import test_mlfg, load_data
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sys import argv
import math
import numpy as np
import sympy


def relative_error(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    sqrt = np.sqrt(mse)
    return sqrt / np.mean(actual)


def complexity(equation):
    l = sympy.count_ops(equation)
    return l


def create_and_evaluate_equation(dataset, n_features=4, n_nodes=1, evaluate=True):
    funcs = ['id', 'sin', 'cos', 'mult-a', 'mult-b']
    model = test_mlfg(datasets=dataset, n_epochs=5000, verbose=False, n_layer=1, n_per_base=1,
                      batch_size=16, L1_reg=0.01, L2_reg=0.01,
                      learning_rate=0.00075, reg_end=4000, gradient='adam')
    best_class = model['best_classifier']

    if evaluate:
        results = []
        x_train, y_train, x_test, y_test = dataset[0][0], dataset[0][1], dataset[1][0], dataset[1][1]
        trainpreds = best_class.evaluate(x_train)
        try:
            results.append(r2_score(y_train, trainpreds))
        except:
            results.append('-')
        try:
            results.append(mean_absolute_percentage_error(y_train, trainpreds))
        except:
            results.append('-')
        try:
            results.append(relative_error(y_train, trainpreds))
        except:
            results.append('-')

        testpreds = best_class.evaluate(x_test)
        try:
            results.append(r2_score(y_test, testpreds))
        except:
            results.append('-')
        try:
            results.append(mean_absolute_percentage_error(y_test, testpreds))
        except:
            results.append('-')
        try:
            results.append(relative_error(y_test, testpreds))
        except:
            results.append('-')

    for i in range(len(best_class.hidden_layers) + 1):
        if i == 0:

            layer_funcs = {}
            params = best_class.hidden_layers[i].get_params()
            weights = params[0]
            biases = params[1]

            # Weights for each feature
            for j in range(n_features):
                feature_weights = weights[j]
                # 5 * n_nodes total weights for each feature, switch functions every n_nodes
                for k in range(len(feature_weights)):
                    func = funcs[math.floor(k/n_nodes)] + str((k % n_nodes) + 1)
                    # Multiply each feature weight w/ the feature
                    if func in layer_funcs:
                        layer_funcs[func].append(str(feature_weights[k]) + '*x_' + str(j + 1))
                    else:
                        layer_funcs[func] = [str(feature_weights[k]) + '*x_' + str(j + 1)]

            # Add everything within each func
            for func in layer_funcs:
                sum = ''
                for var in layer_funcs[func]:
                    sum += var + ' + '
                layer_funcs[func] = sum
            # Add biases
            j = 0
            for func in layer_funcs:
                layer_funcs[func] += str(biases[j])
                j += 1

            # Apply functions
            new_matrix = []
            for func in layer_funcs:
                if func.find('id') != -1:
                    new_matrix.append(layer_funcs[func])
                elif func.find('sin') != -1:
                    new_matrix.append('sin(' + layer_funcs[func] + ')')
                elif func.find('cos') != -1:
                    new_matrix.append('cos(' + layer_funcs[func] + ')')
                elif func.find('mult-a') != -1:
                    left = layer_funcs[func]
                    # Multiply w/ corresponding b value
                    id = func[6:]
                    right = layer_funcs['mult-b' + str(id)]
                    new_matrix.append('(' + left + ')*' + '(' + right + ')')

        elif i < len(best_class.hidden_layers):
            # only working with one hidden layer
            raise Exception('Only working with single-layer model')

        else:
            out_matrix = []
            # Last Layer
            params = best_class.output_layer.get_params()
            weights = params[0]
            biases = params[1]

            # Matrix multiplication: wT and new_matrix
            weights = np.transpose(weights)
            bias_index = 0
            for row in weights:
                expression = ''
                for j in range(len(row)):
                    if j < len(row) - 1:
                        add = '(' + new_matrix[j] + ')' + '+'
                    else:
                        add = '(' + new_matrix[j] + ')'
                    expression += str(row[j]) + '*' + add
                expression += '+' + str(biases[bias_index])
                out_matrix.append(expression)
                bias_index += 1

            numerator = sympy.simplify(out_matrix[0])
            denominator = sympy.simplify(out_matrix[1])
            equation = sympy.simplify(numerator / denominator)

            if evaluate:
                results.append(complexity(equation))
                return str(equation), results
            else:
                return str(equation)


if __name__ == "__main__":
    if len(argv) == 2:
        dataset = load_data(argv[1])
        equation, results = create_and_evaluate_equation(dataset)
        print('Learned Equation:', equation)
        print('Results:', results)
    else:
        print('Invalid number of arguments')
