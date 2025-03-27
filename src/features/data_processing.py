#Libraries

import pandas as pd
import pickle
import os 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import math


def load_data(filepath):
        data = pd.read_csv(f"{filepath}/X_train_update.csv")
        data["description"] = data["designation"] + str(data["description"])
        data = data.drop(["Unnamed: 0", "designation"], axis=1)

        target = pd.read_csv(f"{filepath}/Y_train_CVw08PX.csv")
        target = target.drop(["Unnamed: 0"], axis=1)
        modalite_mapping = {
            modalite: i for i, modalite in enumerate(target["prdtypecode"].unique())
        }
        target["prdtypecode"] = target["prdtypecode"].replace(modalite_mapping)

        with open("models/mapper.pkl", "wb") as fichier:
            pickle.dump(modalite_mapping, fichier)

        df = pd.concat([data, target], axis=1)

        return df

def split_train_test(df, samples_per_class=600):
        grouped_data = df.groupby("prdtypecode")
        X_train_samples = []
        X_test_samples = []

        for _, group in grouped_data:
            samples = group.sample(n=samples_per_class, random_state=42)
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
            samples = group.sample(n=val_samples_per_class, random_state=42)
            X_val_samples.append(samples[["description", "productid", "imageid"]])
            y_val_samples.append(samples["prdtypecode"])

        X_val = pd.concat(X_val_samples)
        y_val = pd.concat(y_val_samples)

        X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
        y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)
 
        return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_images_in_df(df, img_filepath):
        df["image_path"] = (
            f"{img_filepath}/image_"
            + df["imageid"].astype(str)
            + "_product_"
            + df["productid"].astype(str)
            + ".jpg"
        )
        return df

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

if __name__ == "__main__":
    ##Variable definition
    filepath_source = "data/preprocessed"
    filepath_dest = "data/processed"
    img_filepath="data/preprocessed/images/images/image_train"
    
    ### Import dataset
    df = load_data(filepath_source)
    print("Data imported")
    
    #Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(df)
    print("Data splitted")
    
    #Process datasets
    X_train = preprocess_text_in_df(X_train, columns=["description"])
    X_val = preprocess_text_in_df(X_val, columns=["description"])
    X_train = preprocess_images_in_df(X_train, img_filepath)
    X_val = preprocess_images_in_df(X_val, img_filepath)
    print("Data processed")
     
    ## check and create folder for processed files
    if os.path.exists(filepath_dest) == False :
            os.makedirs(filepath_dest)
    X_train.to_csv(os.path.join(filepath_dest, 'X_train_processed.csv'), index=False)
    X_val.to_csv(os.path.join(filepath_dest, 'X_val_processed.csv'), index=False)
    X_test.to_csv(os.path.join(filepath_dest, 'X_test_processed.csv'), index=False)
    y_train.to_csv(os.path.join(filepath_dest, 'y_train_processed.csv'), index=False)
    y_val.to_csv(os.path.join(filepath_dest, 'y_val_processed.csv'), index=False)
    y_test.to_csv(os.path.join(filepath_dest, 'y_test_processed.csv'), index=False)
    print("Processed data exported")