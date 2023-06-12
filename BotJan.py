import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Lade die historischen Preise aus der CSV-Datei
df = pd.read_csv('../Api/newETHUSD2.csv')
second_column = df.iloc[:, 1]

# Entferne das Komma und konvertiere in Float-Werte
second_column = second_column.str.replace(',', '').astype(float)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(second_column.values.reshape(-1, 1))

# Teile die Daten in Trainings- und Testdaten auf
train_data = scaled_data[:int(0.8 * len(df))]
test_data = scaled_data[int(0.8 * len(df)):]

# Funktion zur Erstellung der Trainingsdaten
def create_train_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

# Definiere die Anzahl der vergangenen Zeitpunkte, die als Eingabe verwendet werden sollen
time_steps = 30

# Erstelle die Trainingsdaten
X_train, y_train = create_train_dataset(train_data, time_steps)

print(X_train.shape)  # Sollte (Anzahl der Beispiele, Zeitpunkte) sein
print(y_train.shape)  # Sollte (Anzahl der Beispiele,) sein

# Passe die Form von X_train an
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Passe die Eingabeform des Modells an
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))

# Erstelle das LSTM-Modell
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Kompiliere und trainiere das Modell
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Erstelle die Testdaten
inputs = df['Price'][len(df) - len(test_data) - time_steps:].values.reshape(-1, 1)
inputs = scaler.transform(inputs)
X_test, y_test = create_train_dataset(inputs, time_steps)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Vorhersage des Testdatensatzes
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Ausgabe der Vorhersagen
for i in range(len(predicted_prices)):
    print(f"Vorhersage für Tag {i+1}: {predicted_prices[i][0]}")


