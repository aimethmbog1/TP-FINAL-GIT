import gradio as gr
import joblib

# 1. Charger le modèle pré-entraîné
model = joblib.load("model.pkl")


def predict_sentiment(text):
    """Fonction appelée à chaque fois que l'utilisateur clique sur le bouton."""
    prediction = model.predict([text])[0]

    if prediction == "positif":
        return "🟢 Sentiment Positif"
    else:
        return "🔴 Sentiment Négatif"


# 2. Créer l'interface graphique
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Écrivez un avis sur un produit tech ici (ex: Ce téléphone est génial !)...",
    ),
    outputs=gr.Text(label="Résultat de l'IA"),
    title="🤖 Analyseur d'Opinions Tech",
    description=(
        "Entrez un avis sur un ordinateur, smartphone ou accessoire, "
        "et l'IA devinera si le client est satisfait ou non."
    ),
)

# 3. Lancer le serveur Web
if __name__ == "__main__":
    interface.launch(share=True)
