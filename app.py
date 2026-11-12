import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Cargar el "cerebro" guardado (ya no entrenamos nada)
modelo = tf.keras.models.load_model('modelo_numeros.keras')

def adivinar_dibujo(datos_dibujo):
    # Gradio puede enviar un diccionario con capas. Extraemos la imagen final ('composite')
    if isinstance(datos_dibujo, dict):
        imagen = datos_dibujo['composite']
    else:
        imagen = datos_dibujo

    # Redimensionar lo que dibujaste a 28x28 píxeles (el tamaño que conoce la red)
    imagen_pil = Image.fromarray(imagen).resize((28, 28))
    img_array = np.array(imagen_pil)
    
    # MNIST usa fondo negro y trazos blancos. 
    # Si la imagen viene con fondo blanco (luminosidad alta), invertimos los colores.
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
        
    # Normalizar (valores entre 0 y 1) y adaptar forma a (1, 28, 28)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28)
    
    # Hacer la predicción
    prediccion = modelo.predict(img_array)
    numero_ganador = np.argmax(prediccion)
    
    return f"¡Estoy casi seguro de que es un {numero_ganador}!"

# 2. Construir la interfaz visual
interfaz = gr.Interface(
    fn=adivinar_dibujo,
    inputs=gr.Sketchpad(type="numpy", image_mode="L"), # Lienzo de dibujo en blanco y negro
    outputs=gr.Text(label="Predicción de la Inteligencia Artificial"),
    title="Reconocedor de Números",
    description="Dibuja un número del 0 al 9 en el recuadro y presiona Submit."
)

# 3. Encender la aplicación
interfaz.launch()