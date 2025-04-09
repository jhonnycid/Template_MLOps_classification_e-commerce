import pandas as pd
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import math
import os

# Chemins des données externes
EXTERNAL_DATA_PATH = "/app/data/external"
EXTERNAL_IMAGES_PATH = os.path.join(EXTERNAL_DATA_PATH, "images/image_train")  # Chemin corrigé

class DataImporter:
    def __init__(self, filepath=EXTERNAL_DATA_PATH):
        self.filepath = filepath

    def load_data(self):
        print(f"Chargement des données depuis {self.filepath}")
        data = pd.read_csv(os.path.join(self.filepath, "X_train_update.csv"))
        data["description"] = data["designation"] + str(data["description"])
        data = data.drop(["Unnamed: 0", "designation"], axis=1)

        target = pd.read_csv(os.path.join(self.filepath, "Y_train_CVw08PX.csv"))
        target = target.drop(["Unnamed: 0"], axis=1)
        modalite_mapping = {
            modalite: i for i, modalite in enumerate(target["prdtypecode"].unique())
        }
        target["prdtypecode"] = target["prdtypecode"].replace(modalite_mapping)

        # Créer le répertoire models s'il n'existe pas
        os.makedirs("models", exist_ok=True)
        
        with open("models/mapper.json", "w") as f:
            import json
            json.dump(modalite_mapping, f)

        # Pour la compatibilité avec le code existant
        with open("models/mapper.pkl", "wb") as fichier:
            pickle.dump(modalite_mapping, fichier)

        df = pd.concat([data, target], axis=1)
        print(f"Données chargées : {len(df)} lignes")

        return df

    def split_train_test(self, df, samples_per_class=600):
        print(f"Séparation des données avec {samples_per_class} échantillons par classe")
        grouped_data = df.groupby("prdtypecode")

        X_train_samples = []
        X_test_samples = []

        for _, group in grouped_data:
            if len(group) >= samples_per_class:
                samples = group.sample(n=samples_per_class, random_state=42)
            else:
                # Si pas assez d'échantillons, prendre tout ce qui est disponible
                samples = group
                print(f"Attention: Classe avec seulement {len(group)} échantillons")
            
            X_train_samples.append(samples)

            remaining_samples = group.drop(samples.index)
            X_test_samples.append(remaining_samples)

        X_train = pd.concat(X_train_samples)
        X_test = pd.concat(X_test_samples)

        X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
        X_test = X_test.sample(frac=1, random_state=42).reset_index(drop=True)

        y_train = X_train["prdtypecode"]
        X_train = X_train.drop(["prdtypecode"], axis=1)

        y_test = X_test["prdtypecode"]
        X_test = X_test.drop(["prdtypecode"], axis=1)

        val_samples_per_class = 50

        grouped_data_test = pd.concat([X_test, y_test], axis=1).groupby("prdtypecode")

        X_val_samples = []
        y_val_samples = []

        for _, group in grouped_data_test:
            if len(group) >= val_samples_per_class:
                samples = group.sample(n=val_samples_per_class, random_state=42)
            else:
                samples = group
                print(f"Attention: Classe de validation avec seulement {len(group)} échantillons")
                
            X_val_samples.append(samples[["description", "productid", "imageid"]])
            y_val_samples.append(samples["prdtypecode"])

        X_val = pd.concat(X_val_samples)
        y_val = pd.concat(y_val_samples)

        X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
        y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Train: {len(X_train)} échantillons, Val: {len(X_val)} échantillons, Test: {len(X_test)} échantillons")
        return X_train, X_val, X_test, y_train, y_val, y_test


class ImagePreprocessor:
    def __init__(self, filepath=os.path.join(EXTERNAL_IMAGES_PATH, "image_train")):
        self.filepath = filepath
        print(f"Utilisation des images dans: {self.filepath}")

    def preprocess_images_in_df(self, df):
        """Ajoute le chemin des images au DataFrame"""
        image_paths = []
        
        for _, row in df.iterrows():
            image_path = f"{self.filepath}/image_{row['imageid']}_product_{row['productid']}.jpg"
            if not os.path.exists(image_path):
                print(f"Attention: Image non trouvée: {image_path}")
            image_paths.append(image_path)
            
        df["image_path"] = image_paths
        print(f"Chemins d'images ajoutés à {len(df)} lignes")


class TextPreprocessor:
    def __init__(self):
        # Télécharger les ressources NLTK au démarrage
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
        except Exception as e:
            print(f"Erreur lors du téléchargement des ressources NLTK: {e}")
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("french"))

    def preprocess_text(self, text):
        """Prétraite un texte"""
        if isinstance(text, float) and math.isnan(text):
            return ""
            
        # Supprimer les balises HTML
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except:
            text = str(text)

        # Supprimer les caractères non alphabétiques
        text = re.sub(r"[^a-zA-Z]", " ", text)

        # Tokenization
        words = word_tokenize(text.lower())

        # Suppression des stopwords et lemmatisation
        filtered_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word not in self.stop_words
        ]

        return " ".join(filtered_words[:10])

    def preprocess_text_in_df(self, df, columns):
        """Prétraite les colonnes textuelles du DataFrame"""
        for column in columns:
            print(f"Prétraitement de la colonne '{column}'")
            df[column] = df[column].apply(self.preprocess_text)
        print("Prétraitement des textes terminé")
