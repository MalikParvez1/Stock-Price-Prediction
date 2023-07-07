
from keras.layers import Dense, Dropout, LSTM , Input, Activation, concatenate
import numpy as np

from MergeOnlyETHandBTCPrices import mergeDataframesPrices
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

df = mergeDataframesPrices()

# Adding technical indicatorsa
df['RSI'] = ta.rsi(df.Close, length=15)
df['EMAF'] = ta.ema(df.Close, length=20)
df['EMAM'] = ta.ema(df.Close, length=100)
df['EMAS'] = ta.ema(df.Close, length=150)

df['target'] = df['Close'].shift(-1)
print(df['target'], df['Close'])

# Drop string data and 'Timestamp' column
df.drop(['Date', 'Trades', 'Timestamp'], axis=1, inplace=True)

# Split the data into features and target variables
features = df.drop(['target'], axis=1)
target = df['target']

# Scaling the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Split the data into training and test sets
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train_features = scaled_features[:train_size]
train_target = target[:train_size]
test_features = scaled_features[train_size:]
test_target = target[train_size:]

# Reshape the training data into the required LSTM input format
x_train = np.reshape(train_features, (train_features.shape[0], train_features.shape[1], 1))
y_train = np.array(train_target)

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile and train the LSTM model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=1, batch_size=32)

# Make predictions with the LSTM model
x_test = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predicted_prices = model.predict(x_test)

# Reshape the predicted prices to match the original shape
predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], 1)

# Inverse scaling of the predictions
predicted_prices = scaler.inverse_transform(predicted_prices)

# Output the predictions
print(predicted_prices)
