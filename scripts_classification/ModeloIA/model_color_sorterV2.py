import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Función para cargar el conjunto de datos desde un archivo CSV
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    colors = df[['R', 'G', 'B']].values
    labels = df['Color'].values
    return colors, labels

# Cargar el conjunto de datos
dataset_file = 'dataset.csv'  # Cambia a la ruta de tu archivo CSV
train_data, train_labels = load_dataset(dataset_file)

# Mapeo de etiquetas a números
label_mapping = {"1_Violet": 0, "2_Blue": 1, "3_Cyan": 2, "4_Green": 3, 
                 "5_Yellow": 4, "6_Orange": 5, "7_Red": 6, 
                 "8_White": 7, "9_Black": 8}
train_labels = [label_mapping[label] for label in train_labels]
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=9)

# Definir el modelo
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(3,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(9, activation='softmax')  # 9 categorías
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con más épocas
model.fit(train_data, train_labels, epochs=1000, batch_size=32)  # Aumentar el número de épocas

# Guardar el modelo entrenado
model.save("color_classifier_model.h5")

# Cargar el modelo en otro programa
loaded_model = keras.models.load_model("color_classifier_model.h5")

# Función para predecir el color con el modelo cargado
def predict_color_with_loaded_model(color_rgb):
    color_rgb_normalized = color_rgb / 255.0
    prediction = loaded_model.predict(np.array([color_rgb_normalized]))
    predicted_label = np.argmax(prediction)
    
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    predicted_color = reverse_mapping[predicted_label]
    
    print(f'Color predicho con modelo cargado: {predicted_color}')

# Ejemplo de uso en otro programa
color_ejemplo = np.array([0, 255, 0])  # Verde en formato RGB
predict_color_with_loaded_model(color_ejemplo)
color_ejemplo = np.array([255, 0, 0])  # Verde en formato RGB
predict_color_with_loaded_model(color_ejemplo)
color_ejemplo = np.array([0, 0, 255])  # Verde en formato RGB
predict_color_with_loaded_model(color_ejemplo)

