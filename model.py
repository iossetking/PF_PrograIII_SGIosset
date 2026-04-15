from fcntl import FASYNC

import tensorflow as tf
import numpy as np

def train_model():
  celsius = np.array([30, 31, 32, 8, 15, 22, 38], dtype=np.float32)
  fahrenheit = celsius * 1.8 + 32

  layer = tf.keras.layers.Dense(units=1, input_shape=[1])
  model = tf.keras.Sequential([layer])

  model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error')

  history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

  return model, history
