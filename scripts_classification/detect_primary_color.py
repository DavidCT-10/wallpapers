import cv2
import numpy as np

def extraer_color_primario(imagen):
    # Convertir la imagen de BGR a RGB
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Obtener la forma de la imagen
    altura, ancho, _ = imagen.shape

    # Segmentar la imagen en 6 partes
    segmentos = [
        imagen_rgb[0:altura//2, 0:ancho//3],
        imagen_rgb[0:altura//2, ancho//3:2*ancho//3],
        imagen_rgb[0:altura//2, 2*ancho//3:ancho],
        imagen_rgb[altura//2:altura, 0:ancho//3],
        imagen_rgb[altura//2:altura, ancho//3:2*ancho//3],
        imagen_rgb[altura//2:altura, 2*ancho//3:ancho]
    ]

    colores_primarios_segmento = []

    # Extraer el color primario de cada segmento
    for segmento in segmentos:
        color_primario = np.mean(segmento, axis=(0, 1)).astype(int)
        colores_primarios_segmento.append(color_primario)

    # Extraer el color primario general de la imagen
    color_primario_general = np.mean(imagen_rgb, axis=(0, 1)).astype(int)

    return colores_primarios_segmento, color_primario_general

# Cargar la imagen
ruta_imagen = 'abc.jpg'  # Reemplaza con la ruta de tu imagen
imagen = cv2.imread(ruta_imagen)

# Verificar si la imagen se cargó correctamente
if imagen is not None:
    colores_segmento, color_general = extraer_color_primario(imagen)
    media=[0,0,0]

    # Imprimir los resultados
    print("Colores primarios por segmento:")
    for i, color_segmento in enumerate(colores_segmento):
        print(f"Segmento {i+1}: RGB({color_segmento[0]}, {color_segmento[1]}, {color_segmento[2]})")
        media[0] += color_segmento[0]
        media[1] += color_segmento[1]
        media[2] += color_segmento[2]
    
    media[0]/=6
    media[1]/=6
    media[2]/=6
    
    print("\nColor mediano:")
    print(media)

    print("\nColor primario general:")
    print(f"RGB({color_general[0]}, {color_general[1]}, {color_general[2]})")

else:
    print("No se pudo cargar la imagen. Verifica la ruta de la imagen.")
