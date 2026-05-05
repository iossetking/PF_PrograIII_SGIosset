import os
import tensorflow as tf
import numpy as np
import pandas as pd
import fastf1

if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

def load_dataset(year, number):
    print("Loading data from FastF1...")

    session = fastf1.get_session(year, number, 'R')
    session.load(telemetry=True)

    print("Extracting telemetry...")
    laps = session.laps.pick_driver('VER')
    telemetry = laps.get_telemetry()

    telemetry = telemetry.dropna(subset=['Speed', 'RPM', 'Throttle', 'Brake', 'nGear'])

    X = telemetry[['Speed', 'RPM', 'Throttle', 'Brake']].astype(np.float32).to_numpy()

    y = telemetry['nGear'].astype(np.int32).to_numpy()

    print(f"Dataset loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

def create_model(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            # Softmax will output probabilities for each gear
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def train_model(model, X, y, epochs=10, batch_size=128):
    print("Starting training...")
    # Telemetry has thousands of rows, so a larger batch_size (128) speeds things up
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    print("Model trained successfully")
    return history

def save_model(model, filepath="model_f1_gears.keras"):
    model.save(filepath)
    print(f"Model saved successfully in: {filepath}")

def make_prediction(model, new_data):
    new_data = np.array(new_data, dtype=np.float32)
    predictions = model.predict(new_data)
    predicted_classes = np.argmax(predictions, axis=1)
    return predicted_classes

X, y = load_dataset(2025, 1)

# Calculate num_classes based on the highest gear.
# F1 cars have 8 gears + neutral (0), so usually 9 classes total.
num_classes = int(np.max(y)) + 1
print(f"Class num: {num_classes}")

# 2. Build model
model = create_model(num_classes)

# 3. Train model
train_model(model, X, y, epochs=10)

# 4. Save model
save_model(model)

# 5. Make a prediction with dummy telemetry
# [Speed (km/h), RPM, Throttle (%), Brake (%)]
new_data = [
    [80.0, 6000.0, 20.0, 80.0],   # Curva lenta, frenando (probablemente Marcha 2 o 3)
    [310.0, 11500.0, 100.0, 0.0]  # Recta principal, acelerador a fondo (probablemente Marcha 8)
]

results = make_prediction(model, new_data)
print(f"Gear prediction for slow curve: {results[0]}")
print(f"Gear prediction for straight: {results[1]}")
