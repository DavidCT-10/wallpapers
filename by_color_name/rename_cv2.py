import cv2
import numpy as np
import os

def get_primary_color(image):
    # Obtiene el color primario de una imagen
    pixels = image.reshape(-1, 3)
    primary_color = tuple(np.uint8(np.median(pixels, axis=0)))
    return primary_color

def get_unique_filename(renamed_folder, base_filename):
    # Asegura un nombre de archivo único en la carpeta "renamed"
    count = 1
    new_filename = base_filename
    while os.path.exists(os.path.join(renamed_folder, new_filename)):
        count += 1
        new_filename = f"{os.path.splitext(base_filename)[0]}_{count}{os.path.splitext(base_filename)[1]}"
    return new_filename

def segment_and_rename_images(folder_path):
    # Crea la carpeta "renamed" si no existe
    renamed_folder_path = os.path.join(folder_path, "renamed_cv2")
    os.makedirs(renamed_folder_path, exist_ok=True)

    # Itera sobre las imágenes en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # Intenta leer la imagen con OpenCV
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Verifica si la imagen se cargó correctamente
            if image is None:
                print(f"Error al leer la imagen: {image_path}")
                continue

            # Divide la imagen en 8 segmentos
            height, width, _ = image.shape
            segments = [image[:, i:i + width // 4, :] for i in range(0, width, width // 4)]

            # Obtiene el color primario de cada segmento
            segment_colors = [get_primary_color(segment) for segment in segments]

            # Obtiene el color primario general de la imagen
            general_color = get_primary_color(image)

            # Genera el nuevo nombre de la imagen
            base_filename = f"img_classif_{general_color[2]}_{general_color[1]}_{general_color[0]}.png"
            new_filename = get_unique_filename(renamed_folder_path, base_filename)

            # Guarda la imagen en la carpeta "renamed"
            new_image_path = os.path.join(renamed_folder_path, new_filename)
            cv2.imwrite(new_image_path, image)

if __name__ == "__main__":
    folder_path = "."  # Puedes cambiar esto al directorio donde se encuentran tus imágenes
    segment_and_rename_images(folder_path)
