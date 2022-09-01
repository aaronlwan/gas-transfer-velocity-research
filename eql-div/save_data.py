import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import gzip
import pickle

observed_data = np.array(pd.read_csv('datasets/observed.csv'))

X_observed = observed_data[:, [0, 1, 2, 3]]
Y_observed = observed_data[:, 4]

# Flow Velocity
flow_velocity_observed = np.transpose([np.power(X_observed[:, 2], 4)/(9.806*np.power(X_observed[:, 0], 2))])

# Water Depth
water_depth_observed = np.transpose([np.power(X_observed[:, 1], 3)/np.power(X_observed[:, 3], 4)])

# Fr
fr_observed = flow_velocity_observed/(np.sqrt(9.806*water_depth_observed))

# Re
re_observed = np.multiply(flow_velocity_observed, water_depth_observed)/0.000001

# Re*
re_shear_observed = np.multiply(np.transpose([X_observed[:, 1]]), water_depth_observed)/0.000001

X_observed = np.concatenate((X_observed, flow_velocity_observed, water_depth_observed, fr_observed, re_observed, re_shear_observed), axis=1)


# Split Data (state = 1 for consistency)
x_train, x_test, y_train, y_test = train_test_split(X_observed, Y_observed, random_state=1, train_size=0.8)


with gzip.open('datasets/state_one_all_features_observed', "wb") as f:
    pickle.dump([[x_train, y_train.reshape(-1, 1)], [x_test, y_test.reshape(-1, 1)]], f)
