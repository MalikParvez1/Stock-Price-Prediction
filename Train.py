import numpy as np
import pandas as pd

from MergeTweetsAndPrices import mergeDataframes
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

df = mergeDataframes()

#Adding technical indicators
df['RSI'] = ta.rsi(df.Close, length=15)
df['EMAF'] = ta.ema(df.Close, length=20)
df['EMAM'] = ta.ema(df.Close, length=100)
df['EMAS'] = ta.ema(df.Close, length=150)

df['target'] = df['Close'].shift(-1)
print(df['target'], df['Close'])

sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(df)

pd.to_numeric(df['views'])

df.drop('author', axis=1, inplace=True)
df.drop('Date', axis=1, inplace=True)
df.drop('created_at', axis=1, inplace=True)
df.drop('views', axis=1, inplace=True)


X = []

total_columns = df.shape[1]

#wie viele Minuten schauen wir uns an für eine prediction
backcandles = 6
print(data_set_scaled.shape[0])
for j in range(total_columns):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]): #backcandles + 2
        X[j].append(data_set_scaled[i-backcandles:i,j])

#mov axis from 0 to poistion 2
X=np.moveaxis(X, [0], [2])

target_column_index = df.columns.get_loc('target')
print(target_column_index)

X,yi = np.array(X), np.array(data_set_scaled[backcandles:,target_column_index])
y= np.reshape(yi,(len(yi), 1))

print(X.shape)
print(y.shape)

#split data into train and test
splitlimit = int(len(X)*0.8)
print(splitlimit)sploi
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

