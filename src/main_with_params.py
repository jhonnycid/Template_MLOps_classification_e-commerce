from features.build_features import DataImporter, TextPreprocessor, ImagePreprocessor
from models.train_model import TextLSTMModel, ImageVGG16Model, concatenate
from tensorflow import keras
import pickle
import tensorflow as tf
import json
import os

# Récupérer les paramètres d'entraînement depuis les variables d'environnement
lstm_epochs = int(os.environ.get('LSTM_EPOCHS', 3))
vgg_epochs = int(os.environ.get('VGG_EPOCHS', 1))
batch_size = int(os.environ.get('BATCH_SIZE', 32))
samples_per_class = int(os.environ.get('SAMPLES_PER_CLASS', 50))  # Nouveau paramètre

print(f"Paramètres d'entraînement:")
print(f"  - LSTM Epochs: {lstm_epochs}")
print(f"  - VGG Epochs: {vgg_epochs}")
print(f"  - Batch Size: {batch_size}")
print(f"  - Samples per class: {samples_per_class}")

data_importer = DataImporter()
df = data_importer.load_data()
X_train, X_val, _, y_train, y_val, _ = data_importer.split_train_test(df)

# Preprocess text and images
text_preprocessor = TextPreprocessor()
image_preprocessor = ImagePreprocessor()
text_preprocessor.preprocess_text_in_df(X_train, columns=["description"])
text_preprocessor.preprocess_text_in_df(X_val, columns=["description"])
image_preprocessor.preprocess_images_in_df(X_train)
image_preprocessor.preprocess_images_in_df(X_val)

# Train LSTM model
print("Training LSTM Model")
text_lstm_model = TextLSTMModel()
text_lstm_model.epochs = lstm_epochs
text_lstm_model.batch_size = batch_size
text_lstm_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Finished training LSTM")

print("Training VGG")
# Train VGG16 model
image_vgg16_model = ImageVGG16Model()
image_vgg16_model.epochs = vgg_epochs
image_vgg16_model.batch_size = batch_size
image_vgg16_model.preprocess_and_fit(X_train, y_train, X_val, y_val)
print("Finished training VGG")

with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
    tokenizer_config = json_file.read()
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_config)
lstm = keras.models.load_model("models/best_lstm_model.h5")
vgg16 = keras.models.load_model("models/best_vgg16_model.h5")

print("Training the concatenate model")
model_concatenate = concatenate(tokenizer, lstm, vgg16)

# Utiliser un sous-ensemble de données pour l'optimisation
print(f"Sélection de données pour l'optimisation...")
from sklearn.model_selection import train_test_split
X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)
print(f"Sous-ensemble pour optimisation: {len(X_train_subset)} sur {len(X_train)} échantillons")

lstm_proba, vgg16_proba, new_y_train = model_concatenate.predict(X_train_subset, y_train_subset, new_samples_per_class=samples_per_class)
best_weights = model_concatenate.optimize(lstm_proba, vgg16_proba, new_y_train)
print("Finished training concatenate model")

with open("models/best_weights.pkl", "wb") as file:
    pickle.dump(best_weights, file)

with open("models/best_weights.json", "w") as json_file:
    json.dump(best_weights, json_file)

num_classes = 27

proba_lstm = keras.layers.Input(shape=(num_classes,))
proba_vgg16 = keras.layers.Input(shape=(num_classes,))

weighted_proba = keras.layers.Lambda(
    lambda x: best_weights[0] * x[0] + best_weights[1] * x[1]
)([proba_lstm, proba_vgg16])

concatenate_model = keras.models.Model(
    inputs=[proba_lstm, proba_vgg16], outputs=weighted_proba
)

# Enregistrer le modèle au format h5
concatenate_model.save("models/concatenate.h5")

print("Training complete!")
