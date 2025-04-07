import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from tensorflow import keras
import pandas as pd
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import math

def preprocess_text(text):
    if isinstance(text, float) and math.isnan(text):
        return ""
    # Supprimer les balises HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # Supprimer les caractères non alphabétiques
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenization
    words = word_tokenize(text.lower())
    
    # Suppression des stopwords et lemmatisation
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("french"))  # Vous pouvez choisir une autre langue si nécessaire
    
    filtered_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words]
    return " ".join(filtered_words[:10])

def preprocess_text_in_df(df, columns):
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download('punkt_tab')
    for column in columns:
        df[column] = df[column].apply(preprocess_text)
    return df

def preprocess_images_in_df(df, img_filepath):
        df["image_path"] = (
            f"{img_filepath}/image_"
            + df["imageid"].astype(str)
            + "_product_"
            + df["productid"].astype(str)
            + ".jpg"
        )
        return df
    
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
        
    img_array = preprocess_input(img_array)
    return img_array

def predict(tokenizer, lstm, vgg16, best_weights, mapper, filepath, imagepath):
        
        X = pd.read_csv(filepath)[:10] #

        X = preprocess_text_in_df(X, columns=["description"])
        X = preprocess_images_in_df(X, imagepath)
        
        sequences = tokenizer.texts_to_sequences(X["description"])
        padded_sequences = pad_sequences(
            sequences, maxlen=10, padding="post", truncating="post"
        )

        target_size = (224, 224, 3)
        images = X["image_path"].apply(lambda x: preprocess_image(x, target_size))
        images = tf.convert_to_tensor(images.tolist(), dtype=tf.float32)

        lstm_proba = lstm.predict([padded_sequences])
        vgg16_proba = vgg16.predict([images])

        concatenate_proba = (
            best_weights[0] * lstm_proba + best_weights[1] * vgg16_proba
        )
        final_predictions = np.argmax(concatenate_proba, axis=1)

        return {
            i: mapper[str(final_predictions[i])]
            for i in range(len(final_predictions))
        }
        
def main():
    parser = argparse.ArgumentParser(description= "Input data")
    
    parser.add_argument("--dataset_path", default = "data/preprocessed/X_train_update.csv", type=str,help="File path for the input CSV file.")
    parser.add_argument("--images_path", default = "data/preprocessed/images/images/image_train", type=str,  help="Base path for the images.")
    args = parser.parse_args()
    filepath= args.dataset_path
    imagepath = args.images_path
    
    # Charger les configurations et modèles
    with open("models/tokenizer_config.json", "r", encoding="utf-8") as json_file:
        tokenizer_config = json_file.read()
    tokenizer = keras.preprocessing.text.tokenizer_from_json(tokenizer_config)

    lstm = keras.models.load_model("models/best_lstm_model.h5")
    vgg16 = keras.models.load_model("models/best_vgg16_model.h5")

    with open("models/best_weights.json", "r") as json_file:
        best_weights = json.load(json_file)

    with open("models/mapper.json", "r") as json_file:
        mapper = json.load(json_file)

    # Création de l'instance Predict et exécution de la prédiction
  
    predictions = predict(tokenizer, lstm, vgg16, best_weights, mapper, filepath, imagepath)

    # Sauvegarde des prédictions
    with open("data/predictions.json", "w", encoding="utf-8") as json_file:
        json.dump(predictions, json_file, indent=2)
        
if __name__ == "__main__":
    main()