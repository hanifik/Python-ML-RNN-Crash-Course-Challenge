# Importing libraries, dependencies, etc.
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
import matplotlib.pyplot as plt


###                                   ###
#   PREPARING THE BITCOIN PRICE DATA    #
###                                   ###

#   Reading the Bitcoin price data downloaded from the remote GitHub repository created for this project
url_for_raw_data = (r'https://raw.githubusercontent.com/hanifik/stat613/main/btc2014to2021daily.csv')
data_raw = pd.read_csv(url_for_raw_data)

#   I'll use the 'Close' price; Check for rows with null value and removing them, if any
data_raw['Close'].isnull().values.any()
data_raw = data_raw[data_raw['Close'].notna()]

#   Dropping all the columns but 'Close'; Then reshaping it into an array to further preprocess
data_whole = data_raw.Close.values.reshape(-1,1)

#   Splitting the 'Close' price data, i.e. data_whole, into training and set data
split_ratio = 0.8
data_training_noscale = data_whole[:int(len(data_whole) * split_ratio)]
data_test_noscale = data_whole[int(len(data_whole) * split_ratio):]

#   Normalizing/scaling the training data, i.e. data_training_noscale
#   First, import the MinMaxScaler from sklearn.preprocessing library
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler(feature_range=(0,1))
#   Second, fit and transform data_training_noscale
minmax_scaler.fit(data_training_noscale)
data_training = minmax_scaler.transform(data_training_noscale)
#print(data_training)


###                                                                            ###
#   TRANSFORMING THE BITCOIN PRICE DATA INTO THE SPECIFIC FORMAT FOR RNN MODEL   #
###                                                                            ###

#   From the training data, create samples for training the model via using previous #(input_lag) prices to predict the next one
input_lag = 20  # Number of time steps, i.e., how many periods of lagged prices the model takes as input to learn the next price
data_training_x = []
data_training_y = []
for k in range(len(data_training)-input_lag):
    data_training_x.append(data_training[k:k+input_lag, 0])
    data_training_y.append(data_training[k+input_lag:k+input_lag+1, 0])

#   Check: Last element of an arbitrary array of data_training_x is the same as the (arbitrary array-1)th element of data_training_y
print(data_training_x[1][input_lag-1])  # The arbitrary array here is chosen to be the 1st array
print(data_training_y[0])

#   Due to its recurrent nature, an RNN algorithm requires a particular specification of the input and output vectors
#   First, create the input and output vectors as collections of arrays
data_training_y = np.array(data_training_y)
#print(data_training_y)
data_training_x = np.array(data_training_x)
#print(data_training_x)

#   Second, transform the collection of output arrays into 3-D input vectors as required by RNN architecture
data_training_x = np.reshape(data_training_x, (data_training_x.shape[0],data_training_x.shape[1],1))
#print(data_training_x)

###
#   Normalizing/scaling the test data based on the fitting applied on the training data above, instantiated via "minmax_scaler"
data_test = minmax_scaler.transform(data_test_noscale)
#print(data_test)

#   Applying the exact steps applied above for transforming the training data in order to transform the test data this time
data_test_x = []
data_test_y = []
for k in range(len(data_test)-input_lag):
    data_test_x.append(data_test[k:k+input_lag, 0])
    data_test_y.append(data_test_noscale[k+input_lag:k+input_lag+1, 0])

#
data_test_y = np.array(data_test_y)
# print(data_test_y)
data_test_x = np.array(data_test_x)
# print(data_test_x)

#
data_test_x = np.reshape(data_test_x, (data_test_x.shape[0],data_test_x.shape[1],1))
# print(data_test_x)
###


###                                   ###
#   CONSTRUCTING THE MODEL (RNN)        #
###                                   ###

#   Specifying model-specific variables
regressor = Sequential()    # RNN architecture requires sequential modelling
regressor.add(LSTM(units = 64, return_sequences=True, input_shape = (data_training_x.shape[1], 1), activation='tanh')) # Adding a hidden LSTM unit with 64 neurons
regressor.add(LSTM(units = 64, input_shape = (data_training_x.shape[1], 1), activation='tanh')) # Adding another hidden LSTM unit with 64 neurons
regressor.add(Dropout(0.2)) # Dropping %20 of the neurons that are randomly chosen to avoid overfitting
regressor.add(Dense(units = 1)) # The dimension of the output vector, which is one in our case since the price of the next day is being predicted/learnt
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')  # Using adam as the optimizer since it is the state of the art optimizer used in ML
regressor.fit(data_training_x, data_training_y, epochs = 120, batch_size = 100) # Fitting the model; i.e., learning with 120 epochs each of which has 100 batches

# Retrieving the predicted prices from the LSTM RNN model based on the (normalized/scaled) input data_test_x
data_predicted_y = regressor.predict(data_test_x)
print(data_predicted_y)
data_predicted_y = minmax_scaler.inverse_transform(data_predicted_y)
print(data_predicted_y)
print(data_test_y)

# Plotting the predictions (after inversely transformed) vs. the actual values
plt.plot(data_predicted_y, color = 'blue', label = " Predicted Bitcoin Prices")
plt.plot(data_test_y, color = 'red', label = "Actual Bitcoin Prices")
plt.show()
