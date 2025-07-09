import json

def load_templates(path="response_templates.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def classify_and_respond(email_text, model):
    predicted_label = model.predict([email_text])[0]
    print(f"\nEmail clasificat ca: {predicted_label}")

    templates = load_templates()
    response = templates.get(predicted_label, "Nu avem un răspuns prestabilit pentru această categorie.")
    print("\nRăspuns sugerat:\n" + "-"*16)
    print(response)