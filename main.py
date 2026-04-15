import model

def prediction(model, cel_temp):
    result = model.predict(cel_temp)
    print(f'Temp {cel_temp}C is equal to {result}F')

if __name__ == '__main__':
    # Import and execute the model
    model, history = model.train_model()

    prediction(model, 100)


    # print(f"Model summary:")
    # model.summary()
