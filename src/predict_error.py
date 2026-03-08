import pickle

# load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_error(error_message):
    vec = vectorizer.transform([error_message])
    prediction = model.predict(vec)

    return prediction[0]


# test example
if __name__ == "__main__":
    error = "IndexError: list index out of range"
    result = predict_error(error)
    print("Predicted Error Type:", result)
