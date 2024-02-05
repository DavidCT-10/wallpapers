import os
import cv2
import shutil
# import random
import numpy as np
import matplotlib.pyplot as plt

# Función para obtener el color primario de una imagen
def get_primary_color(image_path):
    print(image_path)
    # Cargar la imagen
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Difuminar la imagen
    # blurred_image = cv2.GaussianBlur(image, (75, 75),100)  # Puedes ajustar el tamaño del kernel según tus necesidades
    # Convertir la imagen difuminada a formato RGB
    # blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

    # Mostrar la imagen difuminada
    # plt.imshow(blurred_image_rgb)
    # plt.title('Blurred Image')
    # plt.axis('off')
    # plt.show()

    # Obtener los píxeles de la imagen difuminada
    pixels = image_rgb.reshape((-1, 3))
    # Encontrar los colores únicos y sus frecuencias
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    # Determinar el color primario
    primary_color = unique_colors[np.argmax(counts)]
    return tuple(primary_color)

    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pixels = image.reshape((-1, 3))
    # unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    # primary_color = unique_colors[np.argmax(counts)]
    # return tuple(primary_color)

# Función para crear las carpetas por colores
def create_color_folders(base_folder):
    color_folders = ['1_Violet', '2_Blue', '3_Cyan', '4_Green', '5_Yellow', '6_Orange', '7_Red', '8_White', '9_Black']
    by_colors_folder = os.path.join(base_folder, 'byColors')

    if not os.path.exists(by_colors_folder):
        os.makedirs(by_colors_folder)

    for folder in color_folders:
        color_folder_path = os.path.join(by_colors_folder, folder)
        if not os.path.exists(color_folder_path):
            os.makedirs(color_folder_path)

# Función para clasificar las imágenes en las carpetas por colores
def classify_images(input_folder):
    by_colors_folder = os.path.join(input_folder, 'byColors')

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            
            primary_color = get_primary_color(image_path)
            
            color_class = find_nearest_color_class(primary_color)

            new_filename = f'img_classif_{primary_color[0]}_{primary_color[1]}_{primary_color[2]}'
            new_filepath = os.path.join(by_colors_folder, color_class, new_filename + os.path.splitext(filename)[1])

            # Comprobar si el archivo ya existe
            counter = 0
            while os.path.exists(new_filepath):
                counter += 1
                new_filename_with_counter = f'{new_filename}_{counter}'
                new_filepath = os.path.join(by_colors_folder, color_class, new_filename_with_counter + os.path.splitext(filename)[1])

            shutil.copy(image_path, new_filepath)

# Función para encontrar la clase de color más cercana
def find_nearest_color_class(color):
    color_classes = ['1_Violet', '2_Blue', '3_Cyan', '4_Green', '5_Yellow', '6_Orange', '7_Red', '8_White', '9_Black']
    distances = [np.linalg.norm(np.array(color) - np.array(get_rgb_from_class(c))) for c in color_classes]
    # print_color(color)
    nearest_class = color_classes[np.argmin(distances)]
    return nearest_class

# Función para obtener el valor RGB asociado a una clase de color
def get_rgb_from_class(color_class):
    # Define RGB values for each color class
    color_dict = {'1_Violet': (138, 43, 226), '2_Blue': (0, 0, 255), '3_Cyan': (0, 255, 255),
                  '4_Green': (0, 128, 0), '5_Yellow': (255, 255, 0), '6_Orange': (255, 165, 0),
                  '7_Red': (255, 0, 0), '8_White': (255, 255, 255), '9_Black': (0, 0, 0)}

    return color_dict[color_class]

def print_color(color_rgb):
    # Crear una figura con un solo píxel y mostrar el color
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow([[color_rgb]])
    ax.axis('off')
    plt.show()


if __name__ == "__main__":
    input_folder = "E:\Fondos de pantalla"

    # Crear carpetas por colores
    create_color_folders(input_folder)

    # Clasificar las imágenes en las carpetas por colores
    classify_images(input_folder)
