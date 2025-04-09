from features.build_features_updated import DataImporter, TextPreprocessor, ImagePreprocessor
from models.train_model import TextLSTMModel, ImageVGG16Model, concatenate
from tensorflow import keras
import pickle
import tensorflow as tf
import os
import json

print("Démarrage de l'entraînement avec les données externes...")

# Créer le répertoire models s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Initialiser les importateurs et préprocesseurs
data_importer = DataImporter()
df = data_importer.load_data()
X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

# Prétraitement du texte et des images
print("Prétraitement des données...")
text_preprocessor = TextPreprocessor()
image_preprocessor = ImagePreprocessor()
text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
image_preprocessor.preprocess_images_in_df(X_train)
image_preprocessor.preprocess_images_in_df(X_val)

# Entraîner le modèle LSTM
print("Entraînement du modèle LSTM...")
text_lstm_model = TextLSTMModel()
text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Entraînement LSTM terminé")

# Entraîner le modèle VGG16
print("Entraînement du modèle VGG16...")
image_vgg16_model = ImageVGG16Model()
image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Entraînement VGG16 terminé")

# Charger les modèles et le tokenizer
print("Chargement des modèles entraînés...")
with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
lstm = keras.models.load_model("models/best_lstm_model.h5")
vgg16 = keras.models.load_model("models/best_vgg16_model.h5")

# Optimiser le modèle combiné
print("Optimisation du modèle combiné...")
model_concatenate = concatenate(tokenizer, lstm, vgg16)
lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train, y_train)
best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
print("Optimisation terminée")

# Enregistrer les poids optimaux
with open("models/best_weights.pkl", "wb") as file:
    pickle.dump(best_weights, file)

# Enregistrer les poids au format JSON pour l'API
with open("models/best_weights.json", "w") as json_file:
    json.dump(best_weights, json_file)

# Créer le modèle combiné final
num_classes = 27
proba_lstm = keras.layers.Input(shape=(num_classes,))
proba_vgg16 = keras.layers.Input(shape=(num_classes,))

weighted_proba = keras.layers.Lambda(
    lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
)([proba_lstm, proba_vgg16])

concatenate_model = keras.models.Model(
    inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
)

# Enregistrer le modèle combiné
concatenate_model.save("models/concatenate.h5")
print("Entraînement complet terminé avec succès!")
