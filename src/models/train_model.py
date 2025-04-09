import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.utils import resample
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score
from tensorflow import keras
import pickle
import json
import mlflow
import mlflow.keras


class TextLSTMModel:
    def __init__(self, max_words=10000, max_sequence_length=10):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.model = None
        self.batch_size = 32
        self.epochs = 3  # Réduit de 10 à 3 époques
        self.embedding_dim = 128

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        # Configurer MLflow pour suivre l'expérience
        mlflow.set_experiment("text_lstm_model")
        
        # Démarrer une nouvelle exécution MLflow
        with mlflow.start_run():
            # Enregistrer les hyperparamètres
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("embedding_dim", self.embedding_dim)
            mlflow.log_param("max_words", self.max_words)
            mlflow.log_param("max_sequence_length", self.max_sequence_length)

            # Prétraitement et entraînement du modèle (code existant)
            self.tokenizer.fit_on_texts(X_train["description"])

            tokenizer_config = self.tokenizer.to_json()
            with open("models/tokenizer_config.json", "w", encoding="utf-8") as json_file:
                json_file.write(tokenizer_config)

            train_sequences = self.tokenizer.texts_to_sequences(X_train["description"])
            train_padded_sequences = pad_sequences(
                train_sequences,
                maxlen=self.max_sequence_length,
                padding="post",
                truncating="post",
            )

            val_sequences = self.tokenizer.texts_to_sequences(X_val["description"])
            val_padded_sequences = pad_sequences(
                val_sequences,
                maxlen=self.max_sequence_length,
                padding="post",
                truncating="post",
            )

            text_input = Input(shape=(self.max_sequence_length,))
            embedding_layer = Embedding(input_dim=self.max_words, output_dim=self.embedding_dim)(
                text_input
            )
            lstm_layer = LSTM(128)(embedding_layer)
            output = Dense(27, activation="softmax")(lstm_layer)

            self.model = Model(inputs=[text_input], outputs=output)

            self.model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

            class MLflowCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    mlflow.log_metrics(logs, step=epoch)

            lstm_callbacks = [
                ModelCheckpoint(
                    filepath="models/best_lstm_model.h5", save_best_only=True
                ),  # Enregistre le meilleur modèle
                EarlyStopping(
                    patience=3, restore_best_weights=True
                ),  # Arrête l'entraînement si la performance ne s'améliore pas
                TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
                MLflowCallback(),  # Enregistre les métriques dans MLflow
            ]

            history = self.model.fit(
                [train_padded_sequences],
                tf.keras.utils.to_categorical(y_train, num_classes=27),
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(
                    [val_padded_sequences],
                    tf.keras.utils.to_categorical(y_val, num_classes=27),
                ),
                callbacks=lstm_callbacks,
            )
            
            # Évaluer le modèle final
            val_loss, val_accuracy = self.model.evaluate(
                [val_padded_sequences],
                tf.keras.utils.to_categorical(y_val, num_classes=27)
            )
            
            # Enregistrer les métriques finales
            mlflow.log_metric("final_val_loss", val_loss)
            mlflow.log_metric("final_val_accuracy", val_accuracy)
            
            # Enregistrer le modèle et le tokenizer dans MLflow
            mlflow.keras.log_model(self.model, "lstm_model")
            
            # Enregistrer le tokenizer comme artefact
            with open("tokenizer.json", "w") as f:
                f.write(self.tokenizer.to_json())
            mlflow.log_artifact("tokenizer.json")


class ImageVGG16Model:
    def __init__(self):
        self.model = None
        self.batch_size = 32
        self.epochs = 1

    def preprocess_and_fit(self, X_train, y_train, X_val, y_val):
        # Configurer MLflow pour suivre l'expérience
        mlflow.set_experiment("image_vgg16_model")
        
        # Démarrer une nouvelle exécution MLflow
        with mlflow.start_run():
            # Enregistrer les hyperparamètres
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("epochs", self.epochs)
            
            # Paramètres
            num_classes = 27

            df_train = pd.concat([X_train, y_train.astype(str)], axis=1)
            df_val = pd.concat([X_val, y_val.astype(str)], axis=1)

            # Créer un générateur d'images pour le set d'entraînement
            train_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=df_train,
                x_col="image_path",
                y_col="prdtypecode",
                target_size=(224, 224),  # Adapter à la taille d'entrée de VGG16
                batch_size=self.batch_size,
                class_mode="categorical",  # Utilisez 'categorical' pour les entiers encodés en one-hot
                shuffle=True,
            )

            # Créer un générateur d'images pour le set de validation
            val_datagen = ImageDataGenerator()  # Normalisation des valeurs de pixel
            val_generator = val_datagen.flow_from_dataframe(
                dataframe=df_val,
                x_col="image_path",
                y_col="prdtypecode",
                target_size=(224, 224),
                batch_size=self.batch_size,
                class_mode="categorical",
                shuffle=False,  # Pas de mélange pour le set de validation
            )

            image_input = Input(
                shape=(224, 224, 3)
            )  # Adjust input shape according to your images

            vgg16_base = VGG16(
                include_top=False, weights="imagenet", input_tensor=image_input
            )

            x = vgg16_base.output
            x = Flatten()(x)
            x = Dense(256, activation="relu")(x)  # Add some additional layers if needed
            output = Dense(num_classes, activation="softmax")(x)

            self.model = Model(inputs=vgg16_base.input, outputs=output)

            for layer in vgg16_base.layers:
                layer.trainable = False

            self.model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

            class MLflowCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    mlflow.log_metrics(logs, step=epoch)

            vgg_callbacks = [
                ModelCheckpoint(
                    filepath="models/best_vgg16_model.h5", save_best_only=True
                ),  # Enregistre le meilleur modèle
                EarlyStopping(
                    patience=3, restore_best_weights=True
                ),  # Arrête l'entraînement si la performance ne s'améliore pas
                TensorBoard(log_dir="logs"),  # Enregistre les journaux pour TensorBoard
                MLflowCallback(),  # Enregistre les métriques dans MLflow
            ]

            history = self.model.fit(
                train_generator,
                epochs=self.epochs,
                validation_data=val_generator,
                callbacks=vgg_callbacks,
            )
            
            # Évaluer le modèle final
            # Nous devons réinitialiser les générateurs pour l'évaluation
            val_generator.reset()
            val_steps = len(val_generator)
            val_metrics = self.model.evaluate(val_generator, steps=val_steps)
            
            # Enregistrer les métriques finales
            mlflow.log_metric("final_val_loss", val_metrics[0])
            mlflow.log_metric("final_val_accuracy", val_metrics[1])
            
            # Enregistrer le modèle dans MLflow
            mlflow.keras.log_model(self.model, "vgg16_model")


class concatenate:
    def __init__(self, tokenizer, lstm, vgg16):
        self.tokenizer = tokenizer
        self.lstm = lstm
        self.vgg16 = vgg16

    def preprocess_image(self, image_path, target_size):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return img_array

    def predict(self, X_train, y_train, new_samples_per_class=50, max_sequence_length=10):
        print(f"Début de la prédiction avec {new_samples_per_class} échantillons par classe")
        num_classes = 27
        
        # Vérifier si X_train et y_train contiennent des données valides
        print(f"Taille de X_train: {len(X_train)}, Taille de y_train: {len(y_train)}")
        print(f"Colonnes disponibles: {X_train.columns.tolist()}")
        
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Les données d'entraînement sont vides")
        
        if "description" not in X_train.columns or "image_path" not in X_train.columns:
            raise ValueError("Colonnes requises 'description' et/ou 'image_path' manquantes dans X_train")

        new_X_train = pd.DataFrame(columns=X_train.columns)
        new_y_train = pd.DataFrame(columns=['prdtypecode'])  # Créez la structure pour les étiquettes

        # Boucle à travers chaque classe
        classes_found = 0
        for class_label in range(num_classes):
            # Indices des échantillons appartenant à la classe actuelle
            indices = np.where(y_train == class_label)[0]

            if len(indices) == 0:
                print(f"Attention: Aucun échantillon pour la classe {class_label}. La classe sera ignorée.")
                continue

            # Sous-échantillonnage aléatoire pour sélectionner 'new_samples_per_class' échantillons
            if len(indices) < new_samples_per_class:
                print(f"Attention: Pas assez d'échantillons pour la classe {class_label}. Utilisation de tous les {len(indices)} échantillons disponibles.")
                sampled_indices = indices
            else:
                sampled_indices = resample(
                    indices, n_samples=new_samples_per_class, replace=False, random_state=42
                )
            
            # Convertir les indices en liste d'entiers
            sampled_indices_list = sampled_indices.tolist() if isinstance(sampled_indices, np.ndarray) else list(sampled_indices)
            
            # Utiliser iloc au lieu de loc pour sélectionner par position, pas par index
            selected_X = X_train.iloc[sampled_indices_list]
            selected_y = y_train.iloc[sampled_indices_list] if isinstance(y_train, pd.Series) else pd.Series(y_train[sampled_indices_list])

            # Ajout des échantillons sous-échantillonnés et de leurs étiquettes aux DataFrames
            new_X_train = pd.concat([new_X_train, selected_X])
            new_y_train = pd.concat([new_y_train, pd.DataFrame({'prdtypecode': selected_y})])
            classes_found += 1
            print(f"Classe {class_label}: {len(sampled_indices)} échantillons ajoutés")

        # Réinitialiser les index des DataFrames
        new_X_train = new_X_train.reset_index(drop=True)
        new_y_train = new_y_train.reset_index(drop=True)
        print(f"Total: {len(new_X_train)} échantillons pour {classes_found} classes")
        
        if len(new_X_train) == 0:
            raise ValueError("Aucun échantillon n'a été sélectionné pour l'entraînement")
        
        # Vérifier les valeurs NaN
        nan_count = new_y_train.isna().sum().sum()
        if nan_count > 0:
            print(f"Attention: {nan_count} valeurs NaN détectées dans y_train. Elles seront remplacées par 0.")
            
        # Nettoyer les valeurs NaN avant de convertir en entiers
        new_y_train = new_y_train.fillna(0)  # Remplacer les NaN par 0 ou une autre valeur appropriée
        print(f"Conversion en tableau d'entiers...")
        new_y_train = new_y_train.values.reshape(-1).astype("int")  # Utiliser -1 pour déterminer automatiquement la taille
        print(f"Forme finale de new_y_train: {new_y_train.shape}")

        # Charger les modèles préalablement sauvegardés
        tokenizer = self.tokenizer
        lstm_model = self.lstm
        vgg16_model = self.vgg16

        train_sequences = tokenizer.texts_to_sequences(new_X_train["description"])
        train_padded_sequences = pad_sequences(
            train_sequences, maxlen=10, padding="post", truncating="post"
        )

        # Paramètres pour le prétraitement des images
        target_size = (
            224,
            224,
            3,
        )  # Taille cible pour le modèle VGG16, ajustez selon vos besoins

        images_train = new_X_train["image_path"].apply(
            lambda x: self.preprocess_image(x, target_size)
        )

        images_train = tf.convert_to_tensor(images_train.tolist(), dtype=tf.float32)

        lstm_proba = lstm_model.predict([train_padded_sequences])

        vgg16_proba = vgg16_model.predict([images_train])

        return lstm_proba, vgg16_proba, new_y_train

    def optimize(self, lstm_proba, vgg16_proba, y_train):
        # Configurer MLflow pour suivre l'expérience
        mlflow.set_experiment("concatenate_model_optimization")
        
        print(f"Début de l'optimisation des poids")
        print(f"Dimensions: lstm_proba: {lstm_proba.shape}, vgg16_proba: {vgg16_proba.shape}, y_train: {y_train.shape}")
        
        # Démarrer une nouvelle exécution MLflow
        with mlflow.start_run():
            # Recherche des poids optimaux en utilisant la validation croisée
            best_weights = None
            best_accuracy = 0.0
            
            # Enregistrer les données d'entrée
            mlflow.log_param("num_samples", len(y_train))
            
            results = []

            for lstm_weight in np.linspace(0, 1, 101):  # Essayer différents poids pour LSTM
                vgg16_weight = 1.0 - lstm_weight  # Le poids total doit être égal à 1

                combined_predictions = (lstm_weight * lstm_proba) + (
                    vgg16_weight * vgg16_proba
                )
                final_predictions = np.argmax(combined_predictions, axis=1)
                accuracy = accuracy_score(y_train, final_predictions)
                
                # Enregistrer chaque combinaison de poids testée
                mlflow.log_metric(f"accuracy_{lstm_weight:.2f}_{vgg16_weight:.2f}", accuracy)
                results.append((lstm_weight, vgg16_weight, accuracy))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = (lstm_weight, vgg16_weight)
            
            # Enregistrer les meilleurs poids et la meilleure précision
            mlflow.log_metric("best_accuracy", best_accuracy)
            mlflow.log_param("best_lstm_weight", best_weights[0])
            mlflow.log_param("best_vgg16_weight", best_weights[1])
            
            # Enregistrer un fichier résumant les résultats de l'optimisation
            results_df = pd.DataFrame(results, columns=["lstm_weight", "vgg16_weight", "accuracy"])
            results_df.to_csv("optimization_results.csv", index=False)
            mlflow.log_artifact("optimization_results.csv")
            
            # Enregistrer également les poids sous forme de fichier JSON pour faciliter le chargement ultérieur
            with open("models/best_weights.json", "w") as f:
                json.dump(best_weights, f)
            mlflow.log_artifact("models/best_weights.json")

            return best_weights
