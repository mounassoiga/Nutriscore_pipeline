import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

# --- CONFIGURATION INITIALE ---

CHEMIN_MODELE = 'modele_nutriscore_random_forest.joblib' 
app = Flask(__name__)

# Mapping inverse (Conversion de chiffre à lettre)
INVERSE_MAPPING_NUTRISCORE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# ATTENTION : CET ORDRE DOIT CORRESPONDRE EXACTEMENT À X_train !
# J'ai restauré la liste de 6 features pour un Random Forest standard.
COLUMNS_ORDER = [
    'énergie', 
    'sucres',
    'sel', 
    'acides_gras_saturés', 
    'fibres_alimentaires',
          # Assurez-vous d'avoir cette 6e feature si elle est dans votre modèle
] 

# Fonction utilitaire pour associer la couleur officielle au grade
def obtenir_couleur(grade):
    """Retourne le code couleur Nutri-Score."""
    couleurs = {
        'A': '#00814F', # Vert foncé
        'B': '#8DC63F', # Vert clair
        'C': '#FFC400', # Jaune
        'D': '#F18838', # Orange
        'E': '#E53935'  # Rouge
    }
    return couleurs.get(grade, '#CCCCCC') # Gris par défaut

def obtenir_message_feedback(grade):
    """Retourne un message explicatif en fonction du grade Nutri-Score."""
    messages = {
        'A': "Excellente",       # <-- Change ce texte ici
        'B': "Bonne",            # <-- Change ce texte ici
        'C': "Moyenne",          # <-- Change ce texte ici
        'D': "Médiocre",         # <-- Change ce texte ici
        'E': "Trés mauvaise"          # <-- Change ce texte ici
    }
    # Si le grade n'est pas trouvé, on affiche "Score non classifié"
    return messages.get(grade, "Score non classifié.")

# --- CHARGEMENT DU MODÈLE (Une seule fois au démarrage) ---
try:
    model_champion = joblib.load(CHEMIN_MODELE)
    print(f"✅ Modèle chargé depuis {CHEMIN_MODELE}")
except Exception as e:
    print(f"❌ Erreur de chargement du modèle : {e}")
    model_champion = None


# --- DÉFINITION DES ROUTES DE L'API ---

# ROUTE 1 : PAGE D'ACCUEIL (GET /)
@app.route('/')
def home():
    """Route d'accueil, affiche l'interface HTML."""
    return render_template('index.html')


# ROUTE 2 : PRÉDICTION API (POST /predict)
@app.route('/predict', methods=['POST'])
def predict():
    """Route API pour la prédiction Nutri-Score."""
    if model_champion is None:
        return jsonify({'error': 'Le modèle n\'est pas disponible.'}), 500

    try:
        data = request.get_json(force=True)
        
        # 1. Extraction et validation des données dans l'ordre du modèle (COLUMNS_ORDER)
        input_data = []
        for col in COLUMNS_ORDER:
            # Tente de récupérer la valeur et la convertit en float
            # Cette étape lèvera une KeyError si une donnée est manquante
            input_data.append(float(data[col]))

        # 2. Préparation du DataFrame
        # Assurez-vous que les données sont passées sous forme de liste de listes (ou tableau)
        df_input = pd.DataFrame([input_data], columns=COLUMNS_ORDER)

        # 3. Prédiction et conversion en lettre
        prediction_encodee = model_champion.predict(df_input)[0]
        prediction_lettre = INVERSE_MAPPING_NUTRISCORE[prediction_encodee]

        # 4. Renvoi du résultat en JSON
        return jsonify({
            'status': 'success',
            'nutriscore_predit': prediction_lettre,
            'couleur': obtenir_couleur(prediction_lettre),
            'feedback':obtenir_message_feedback(prediction_lettre)
        })

    except KeyError as e:
        # Erreur si une donnée est manquante dans le formulaire
        return jsonify({'error': f"Donnée manquante : {e}. Toutes les clés ({COLUMNS_ORDER}) sont requises."}), 400
    except ValueError:
        # Erreur si les données ne sont pas numériques
        return jsonify({'error': 'Les valeurs doivent être numériques (float ou int).'}), 400
    except Exception as e:
        # Erreur inconnue (ex: problème de prédiction)
        return jsonify({'error': f"Erreur interne lors de la prédiction : {str(e)}"}), 500


if __name__ == '__main__':
    # Lance le serveur
    app.run(debug=True)