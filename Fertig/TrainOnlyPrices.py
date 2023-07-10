from keras.layers import Dense, Dropout, LSTM, Input, Activation, BatchNormalization
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt


from MergeOnlyETHandBTCPrices import mergeDataframesPrices
from keras import optimizers
from keras.models import Model
from sklearn.metrics import r2_score


df = mergeDataframesPrices()

# Adding technical indicators
df['RSI'] = ta.rsi(df.Close, length=15)
df['EMAF'] = ta.ema(df.Close, length=20)
df['EMAM'] = ta.ema(df.Close, length=100)
df['EMAS'] = ta.ema(df.Close, length=150)

#Removing NaN values from the df data frame
#Creating the "target" column by shifting the closing prices up by one row
df = df.dropna()
df['target'] = df['Close'].shift(-1)
print(df['target'], df['Close'])

#Removing NaN values from the df data frame after adding the technical indicators
#Resetting data frame index
df.dropna(inplace=True)
df.reset_index(inplace=True)

#Drop unnessasary colums
df.drop(['Date', 'Trades', 'Volume', 'Volume_(BTC)', 'Volume_(Currency)', 'Timestamp', 'index', 'Close', 'Weighted_Price' ], axis=1, inplace=True)

data_set = df.iloc[:, 0:12]#.values
pd.set_option('display.max_columns', None)

data_set.head(-1)

#Scaling data to the range of 0 to 1
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data_set)
print(data_set_scaled)

# multiple feature from data provided to the model
X = []

backcandles = 10
print(data_set_scaled.shape[0])
for j in range(11):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):#backcandles+2
        X[j].append(data_set_scaled[i-backcandles:i, j])

#move axis from 0 to position 2
X=np.moveaxis(X, [0], [2])

# Choose -1 for last column, classification else -2...
X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-1])
y=np.reshape(yi,(len(yi),1))

print(X)
print(X.shape)
print(y)
print(y.shape)

# split data into train test sets
splitlimit = int(len(X)*0.8)
print(splitlimit)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(y_train)

np.random.seed(10)

#Creating LSTM model
lstm_input = Input(shape=(backcandles, 11), name='lstm_input')
inputs = LSTM(150, name='first_layer', return_sequences=True)(lstm_input)
inputs = Dropout(0.2)(inputs)  
inputs = BatchNormalization()(inputs)  
inputs = LSTM(150, name='second_layer')(inputs)
inputs = Dropout(0.2)(inputs)
inputs = BatchNormalization()(inputs)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
model.fit(x=X_train, y=y_train, batch_size=3000, epochs=10, shuffle=True, validation_split = 0.1)

#Predict with the model
y_pred = model.predict(X_test)
for i in range(10):
    print(y_pred[i], y_test[i])

#Calculate R2 score for accuracy
r2 = r2_score(y_test, y_pred)
print('R2 für LSTM:', r2)

#Ploting the actual and predicted prices
plt.figure(figsize=(16,8))
plt.plot(y_test, color = 'black', label = 'Test')
plt.plot(y_pred, color = 'red', label = 'pred')
plt.legend()
plt.show()