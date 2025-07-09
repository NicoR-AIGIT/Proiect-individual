from data_preprocessing import load_and_clean_data
from model_training import train_model
from predict_and_respond import classify_and_respond

if __name__ == "__main__":
    # Load and clean data
    X_train, X_test, y_train, y_test = load_and_clean_data("emails.csv")

    # Train model
    model = train_model(X_train, y_train, X_test, y_test)

    #print("Classes in y_test:", set(y_test))
    #print("Classes in model.classes_:", model.classes_)

    # Predict and respond to a new example email
    example_email = "Am probleme cu autentificarea in aplicatia bancii."
    classify_and_respond(example_email, model)