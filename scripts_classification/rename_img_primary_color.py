import os
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt


def extraer_color_primario(image_path):
    image = cv2.imread(image_path)
    imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    altura, ancho, _ = image.shape
    segmentos = [
        imagen_rgb[0:altura//2, 0:ancho//3],
        imagen_rgb[0:altura//2, ancho//3:2*ancho//3],
        imagen_rgb[0:altura//2, 2*ancho//3:ancho],
        imagen_rgb[altura//2:altura, 0:ancho//3],
        imagen_rgb[altura//2:altura, ancho//3:2*ancho//3],
        imagen_rgb[altura//2:altura, 2*ancho//3:ancho]
    ]
    colores_primarios_segmento = []
    for segmento in segmentos:
        color_primario = np.mean(segmento, axis=(0, 1)).astype(int)
        colores_primarios_segmento.append(color_primario)
    color_primario_general = np.mean(imagen_rgb, axis=(0, 1)).astype(int)
    return colores_primarios_segmento, tuple(color_primario_general)


# Función para clasificar las imágenes en las carpetas por colores
def rename_images(input_folder):
    by_colors_folder = os.path.join(input_folder, 'byColorsRename')

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, filename)
            
            # primary_color = get_primary_color(image_path)
            _,primary_color = extraer_color_primario(image_path)
        
            new_filename = f'img_classif_{primary_color[0]}_{primary_color[1]}_{primary_color[2]}'

            # Comprobar si el archivo ya existe
            counter = 0
            while os.path.exists(input_folder):
                counter += 1
                new_filename_with_counter = f'{new_filename}_{counter}'
                new_filepath = os.path.join(by_colors_folder, new_filename_with_counter + os.path.splitext(filename)[1])

            shutil.copy(image_path, new_filepath)


if __name__ == "__main__":
    input_folder = "E:\Fondos de pantalla\All_img"

    # Clasificar las imágenes en las carpetas por colores
    rename_images(input_folder)
