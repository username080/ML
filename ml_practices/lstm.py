import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Örnek veri: 100 örnek, 10 zaman adımı, 8 özellik
X = np.random.random((100, 10, 8))
y = np.random.random((100, 1))

model = Sequential()
model.add(LSTM(32, input_shape=(10, 8)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5)

print("Eğitim tamamlandı.")
