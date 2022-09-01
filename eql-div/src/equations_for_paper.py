from equation_from_eql import create_equation
from mlfg_final import load_data
import numpy as np


def combinations(l):
    if l:
        result = combinations(l[:-1])
        return result + [c + [l[-1]] for c in result]
    else:
        return [[]]


def run():
    # Store everything in here
    learned_equations = []

    # get all possible combos of features
    features = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    combos = combinations(features)

    # load data
    dataset = load_data('datasets/state_one_all_features_observed')

    tracker = 0
    for combo in combos:
        print(tracker)
        if len(combo) > 0:

            label = 'Feature(s):'
            for feature in combo:
                label += ' ' + str(feature)
            print(label)
            new_row = [label]

            x_train = dataset[0][0][:, np.array(combo) - 1]
            x_test = dataset[1][0][:, np.array(combo) - 1]
            data = [[x_train, dataset[0][1]], [x_test, dataset[1][1]]]
            results = create_equation(data, n_features=len(combo), n_nodes=1, evaluate=True)

            new_row.append(results[0])
            new_row += results[1]

            learned_equations.append(new_row)

            np.savetxt("learned_equations-div.csv", np.array(learned_equations),
                       delimiter=",", fmt='%s')

        tracker += 1


if __name__ == "__main__":
    run()

