#  Analyseur d'Opinions Tech - Projet MLOps

Ce projet implémente une infrastructure MLOps complète pour un modèle d'Intelligence Artificielle de classification de sentiments (Avis sur des produits Tech).

Il démontre l'utilisation des bonnes pratiques de l'industrie : gestion de versions avec **Git Flow**, stockage de modèles lourds avec **Git LFS**, analyse statique et sécurité avec **Pre-commit**, tests unitaires avec **Pytest**, et intégration continue via **GitHub Actions**.

##  Prérequis
- Python 3.10 ou supérieur
- Git et Git LFS installés sur votre machine

##  Installation rapide

**1. Cloner le dépôt et récupérer les fichiers LFS**
\`\`\`bash
git clone https://github.com/aimethmbog1/TP-FINAL-GIT.git
cd TP-FINAL-GIT
git lfs pull
\`\`\`

**2. Créer et activer l'environnement virtuel**
\`\`\`bash
# Sur Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
\`\`\`

**3. Installer les dépendances Python**
\`\`\`bash
pip install --upgrade pip
pip install -r requirements.txt
\`\`\`

## Configuration des Git Hooks (Pre-commit)
Pour garantir la qualité du code et éviter la fuite de données sensibles, ce projet utilise `pre-commit`. **Cette étape est obligatoire pour tous les développeurs du projet.**

Une fois les dépendances installées, exécutez cette commande pour lier les hooks à votre Git local :
\`\`\`bash
pre-commit install
\`\`\`
Désormais, à chaque `git commit`, les outils suivants vérifieront automatiquement votre code :
- **Check-added-large-files** : Bloque les fichiers > 5Mo (oblige l'usage de LFS).
- **Black** : Formateur de code Python automatique.
- **Flake8** : Linter (vérifie le respect de la norme PEP8).
- **Detect-secrets** : Empêche le commit accidentel de mots de passe ou clés API.

##  Exécuter les tests unitaires
Le modèle est soumis à un test d'accuracy (doit être > 80%). Pour lancer l'examen localement :
\`\`\`bash
pytest -s test_model.py
\`\`\`

##  Lancer l'Application Web (Gradio)
Pour tester le modèle d'Intelligence Artificielle via une interface graphique :
\`\`\`bash
python app.py
\`\`\`
Ouvrez ensuite le lien local généré dans votre navigateur (généralement `http://127.0.0.1:7860`).
