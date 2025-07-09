from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import csv
import os
import seaborn as sns
from sklearn.model_selection import cross_val_score

def train_model(X_train, y_train, X_test, y_test):
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

 # Antrenarea modelului și obținerea performanței pe fiecare fold din cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.2f}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.2f}")

    #print("model.classes_:", model.classes_)
    #print("y_test unique labels:", set(y_test))

    common_labels = [label for label in model.classes_ if label in set(y_test)]
    if not common_labels:
        raise ValueError("Nicio etichetă comună între model.classes_ și y_test. Verifică datele de test.")
    
    cm = confusion_matrix(y_test, y_pred, labels=common_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, xticklabels=model.classes_, yticklabels=model.classes_, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

#     print("\nRezultate pentru fiecare email testat:")
#     for i in range(len(X_test)):
#         email_text = X_test.iloc[i]  # Email-ul curent
#         predicted_class = y_pred[i]  # Clasa prezisă
#         print(f"\nEmail testat: {email_text}")
#         print(f"Clasificat ca: {predicted_class}")

#         # Dictionary cu răspunsuri
#         responses = {
#             "autentificare": "Bună ziua! Vă rugăm să vă asigurați că aveți cea mai recentă versiune a aplicației. Dacă problema persistă, contactați departamentul tehnic la suport@banca.ro.",
#             "card_blocat": "Bună ziua! Pentru deblocarea cardului, vă rugăm să sunați la serviciul clienți la numărul 0800-123-456 sau să vizitați cea mai apropiată sucursală.",
#             "plati_refuzate": "Bună ziua! Verificați dacă aveți suficiente fonduri și dacă cardul este activ. Pentru mai multe informații, contactați-ne.",
#             "cont_nou": "Bună ziua! Puteți deschide un cont nou direct din aplicație sau vizitând una dintre sucursalele noastre.",
#             "general": "Vă mulțumim pentru mesaj. Un operator vă va contacta în cel mai scurt timp pentru a vă ajuta cu solicitarea dvs."
#         }

#         # Afișează răspunsul corespunzător
#         print("Răspuns sugerat:")
#         print("----------------")
#         print(responses.get(predicted_class, "Răspunsul nu este disponibil."))
    
# #Afisare a incadrarii eronate sau negategorisite si crearea un fisier csv cu erorile
#     misclassified = []
   
#     for i in range(len(y_test)):
#         if y_pred[i] != y_test[i]:
#             print("\n--- Email clasificare greșită ---")
#             print(f"Email: {X_test[i]}")
#             print(f"Etichetă reală: {y_test[i]}")
#             print(f"Etichetă prezisă: {y_pred[i]}")
#             misclassified.append([X_test[i], y_test[i], y_pred[i]])


 # Convertim în liste pentru a evita probleme de index
    X_test_list = list(X_test)
    y_test_list = list(y_test)

    print("\nRezultate pentru fiecare email testat:")
    responses = {
        "autentificare": "Bună ziua! Vă rugăm să vă asigurați că aveți cea mai recentă versiune a aplicației. Dacă problema persistă, contactați departamentul tehnic la suport@banca.ro.",
        "card_blocat": "Bună ziua! Pentru deblocarea cardului, vă rugăm să sunați la serviciul clienți la numărul 0800-123-456 sau să vizitați cea mai apropiată sucursală.",
        "plati_refuzate": "Bună ziua! Verificați dacă aveți suficiente fonduri și dacă cardul este activ. Pentru mai multe informații, contactați-ne.",
        "cont_nou": "Bună ziua! Puteți deschide un cont nou direct din aplicație sau vizitând una dintre sucursalele noastre.",
        "credite": "Bună ziua! Informații despre credite găsiți pe site-ul nostru sau contactând un consultant financiar.",
        "investitii": "Bună ziua! Pentru detalii despre investiții, vă recomandăm o întâlnire cu unul dintre experții noștri.",
        "general": "Vă mulțumim pentru mesaj. Un operator vă va contacta în cel mai scurt timp pentru a vă ajuta cu solicitarea dvs."
    }

    misclassified = []
    for i in range(len(X_test_list)):
        email_text = X_test_list[i]
        predicted_class = y_pred[i]
        real_class = y_test_list[i]

        print(f"\nEmail testat: {email_text}")
        print(f"Clasificat ca: {predicted_class}")
        print("Răspuns sugerat:")
        print("----------------")
        print(responses.get(predicted_class, "Răspunsul nu este disponibil."))

        if predicted_class != real_class:
            print("\n--- Email clasificare greșită ---")
            print(f"Etichetă reală: {real_class}")
            print(f"Etichetă prezisă: {predicted_class}")
            misclassified.append([email_text, real_class, predicted_class])
# Salvează într-un fișier CSV
    if misclassified:
        output_path = "erori_clasificare.csv"
        with open(output_path, mode="w", encoding="utf-8", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["email", "eticheta_reală", "eticheta_prezisă"])
            writer.writerows(misclassified)
        print(f"\nAu fost salvate {len(misclassified)} erori în fișierul '{output_path}'.")
    else:
        print("\n***************Nicio eroare de clasificare nu a fost găsită.*******************")


     # Plotarea unui grafic pentru scorurile de validare
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='b')
    plt.title('Performanța modelului pe fiecare fold (Cross-validation)')
    plt.xlabel('Fold')
    plt.ylabel('Acuratețe')
    plt.grid(True)
    plt.show()

    
    return model