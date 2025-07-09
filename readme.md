# 📬 Email Sorter AI - Proiect Final AI Engineering

Acest proiect utilizează un model de învățare automată pentru a clasifica automat emailurile primite într-un departament de suport bancar și pentru a genera un răspuns standardizat pe baza clasei detectate.

---

## 📁 Structura Proiectului

Email-Sorter-AI/
│
├── data/
│ └── emails.csv # Set de date cu emailuri etichetate
│
├── models/
│ └── email_classifier.pkl # Modelul antrenat salvat
│
├── visuals/
│ └── confusion_matrix.png # Matricea de confuzie salvată
│
├── templates/
│ └── response_templates.json # Răspunsuri predefinite pe categorie
│
├── src/
│ ├── data_preprocessing.py # Curățare și vectorizare text
│ ├── model_training.py # Antrenare model + metrici
│ └── predict_and_respond.py # Clasificare + generare răspuns
│
└── main.py # Script principal


---

## ⚙️ Funcționalități

- Curățare și preprocesare emailuri folosind `pandas`, `nltk` și `scikit-learn`
- Antrenare model Naive Bayes pentru clasificarea categoriilor de suport
- Generare automată de răspunsuri pe baza unui fișier JSON
- Vizualizare performanță cu `matplotlib`

---

## 🛠️ Cerințe

- Python 3.8+
- `pandas`, `nltk`, `scikit-learn`, `matplotlib`

Instalează pachetele necesare:
```bash
pip install -r requirements.txt