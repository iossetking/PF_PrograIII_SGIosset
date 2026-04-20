import model
import os
from tensorflow import keras

MODEL_ROUTE = "trained_model.keras"

def save_model(trained_model, route=MODEL_ROUTE):
  trained_model.save(route)
  print(f"✅ Model saved in: {os.path.abspath(route)}")

def load_model(route=MODEL_ROUTE):
  if os.path.exists(route):
    print(f"📦 Saving model from: {os.path.abspath(route)}")
    return keras.models.load_model(route)
  return None


def prediction(model, cel_temp):
  result = model.predict(cel_temp)
  print(f'Temp {cel_temp}C is equal to {result}F')


if __name__ == "__main__":
  trained_model = load_model()

  if trained_model is None:
    print("🚀 No trained model. Training...")
    trained_model, history = model.train_model()
    save_model(trained_model)

