# Nutriscore_pipeline
Prédicteur de Nutri-Score - Pipeline Data Science Complet
Ce projet vise à prédire le Nutri-Score (de A à E) d'un produit alimentaire en se basant sur ses informations nutritionnelles. Il couvre l'intégralité du cycle de vie d'un projet de Data Science : de l'acquisition des données par scraping au déploiement d'une interface web interactive.

🚀 Aperçu des Performances

Modèle Champion : Stacking Classifier (Random Forest, XGBoost, KNN)

Précision (Accuracy) : 88.33%

Technologies : Python, Flask, Scikit-Learn, Pandas, Joblib.

🛠️ Pipeline du Projet

1. Acquisition & Ingénierie des Données

Web Scraping : Extraction initiale de données sur Open Food Facts (2 449 produits).

Enrichissement : Fusion avec un dataset Kaggle officiel pour atteindre un volume robuste de 358 476 lignes.

Traitement de formats complexes : Transformation de dictionnaires JSON imbriqués en colonnes structurées.

2. Prétraitement & Nettoyage

Filtrage des 6 features clés (énergie, sucres, sel, acides gras saturés, fibres, protéines).

Imputation par la médiane pour une gestion robuste des valeurs manquantes.

Suppression des doublons et uniformisation de la variable cible (A-E).

Dataset final d'entraînement : 188 288 lignes.

3. Modélisation

Comparaison de plusieurs algorithmes avant l'adoption d'une approche ensembliste :

Random Forest : 87.50%

XGBoost : 85.19%
                                          
Stacking Classifier : 88.33% (Méta-modèle : Régression Logistique).

4. Déploiement

Le modèle est exposé via une API Flask. Une interface web permet de saisir les valeurs nutritionnelles et d'obtenir en temps réel :

Le grade Nutri-Score.

La couleur officielle associée.

Un conseil nutritionnel personnalisé.
