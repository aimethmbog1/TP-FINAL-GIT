import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib


def load_data():
    """Génère un faux jeu de données d'avis sur des produits tech."""
    data = {
        "text": [
            "Ce téléphone est incroyable, la batterie dure deux jours !",
            "Très déçu par cet ordinateur, il surchauffe constamment.",
            "L'écran de cette tablette est magnifique et très fluide.",
            "Le casque est fragile, le plastique s'est cassé au bout d'un mois.",
            "Excellent rapport qualité-prix, je recommande cette montre connectée.",
            "Le logiciel de cet appareil photo est plein de bugs, inutilisable.",
            "Une vraie poubelle, je demande un remboursement immédiat.",
            "C'est le meilleur achat de ma vie, la qualité est incroyable !",
            "L'appareil est tombé en panne en deux jours, très mauvaise qualité.",
            "Super produit, livraison très rapide et emballage soigné.",
        ],
        "label": [
            "positif",
            "négatif",
            "positif",
            "négatif",
            "positif",
            "négatif",
            "négatif",
            "positif",
            "négatif",
            "positif",
        ],
    }
    return pd.DataFrame(data)


def train_model():
    """Entraîne le modèle et le sauvegarde."""
    print("Chargement des données...")
    df = load_data()

    print("Création du pipeline (TF-IDF + Régression Logistique)...")
    # Le pipeline transforme le texte en nombres, puis applique l'algorithme
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())

    print("Entraînement en cours...")
    model.fit(df["text"], df["label"])

    print("Sauvegarde du modèle dans 'model.pkl'...")
    # joblib crée un fichier .pkl que notre Git LFS va surveiller
    joblib.dump(model, "model.pkl")
    print("Entraînement terminé avec succès ! ✅")


if __name__ == "__main__":
    train_model()
