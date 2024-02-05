import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Función para generar datos de entrenamiento
def generate_training_data(num_samples=1000):
    colors = np.random.randint(0, 256, size=(num_samples, 3), dtype=np.uint8)
    labels = []
    
    for color in colors:
        if color[0] > color[1] and color[0] > color[2]:
            labels.append("Rojo")
        elif color[1] > color[0] and color[1] > color[2]:
            labels.append("Verde")
        elif color[2] > color[0] and color[2] > color[1]:
            labels.append("Azul")
        else:
            labels.append("Otros")
    
    return colors, labels

# Generar datos de entrenamiento
train_data, train_labels = generate_training_data()

# Preprocesar los datos
train_data = train_data / 255.0  # Normalizar los valores a un rango de 0 a 1

# Convertir etiquetas a números
label_mapping = {"Rojo": 0, "Verde": 1, "Azul": 2, "Otros": 3}
train_labels = [label_mapping[label] for label in train_labels]
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=4)

# Definir el modelo
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(3,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 categorías (Rojo, Verde, Azul, Otros)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(train_data, train_labels, epochs=10, batch_size=32)

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
