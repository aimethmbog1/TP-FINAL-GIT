import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def test_model_accuracy():
    """Charge le modèle et vérifie que sa précision dépasse 80%."""

    # 1. Charger le modèle généré par train.py
    model = joblib.load("model.pkl")

    # 2. Créer un jeu de données de test
    test_data = {
        "text": [
            "Je déteste ce produit, une vraie poubelle, je demande un remboursement.",
            "C'est le meilleur achat de ma vie, la qualité est incroyable !",
            "Le design est beau mais l'appareil est tombé en panne en deux jours.",
            "Super produit, livraison très rapide et emballage soigné.",
        ],
        "label": ["négatif", "positif", "négatif", "positif"],
    }
    df_test = pd.DataFrame(test_data)

    # 3. Demander au modèle de prédire les sentiments
    predictions = model.predict(df_test["text"])

    # 4. Calculer la précision (accuracy)
    accuracy = accuracy_score(df_test["label"], predictions)
    print(f"\nAccuracy du modèle sur le jeu de test : {accuracy * 100:.2f}%")

    # 5. L'affirmation (Assert) : le cœur du test unitaire
    assert accuracy >= 0.80, f"Échec : L'accuracy de {accuracy} est inférieure à 80%"
