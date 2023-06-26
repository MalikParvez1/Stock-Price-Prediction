from MergeOnlyETHandBTCPrices import mergeDataframesPrices
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import matplotlib.pyplot as plt

df = mergeDataframesPrices()

df.drop('Date', axis=1, inplace=True)
df.drop('Timestamp', axis=1, inplace=True)

# Adding technical indicators
df['RSI'] = ta.rsi(df.Close, length=15)
df['EMAF'] = ta.ema(df.Close, length=20)
df['EMAM'] = ta.ema(df.Close, length=100)
df['EMAS'] = ta.ema(df.Close, length=150)

df['target'] = df['Close'].shift(-1)
print(df['target'], df['Close'])

sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(df)

# Assuming the target variable is the last column
X = data_set_scaled[:, :-1]
y = data_set_scaled[:, -1]

# Splitting the data into training and testing sets (adjust the ratio as needed)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))

# Plot loss values
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()