# Clasificador de Números Escritos a Mano (MNIST)

Este proyecto consiste en la creación de una red neuronal artificial construida desde cero con Python y TensorFlow / Keras, capaz de reconocer e identificar dígitos escritos a mano (del 0 al 9) utilizando el conjunto de datos MNIST.

El modelo alcanzó una precisión del 97.72% en el conjunto de prueba tras 5 epochs de entrenamiento.

## Tecnologías Utilizadas

Python 3

TensorFlow / Keras (Construcción y entrenamiento de la red neuronal)

NumPy (Procesamiento de datos numéricos)

Matplotlib (Visualización de imágenes)

Git & GitHub (Control de versiones)

## Arquitectura de la Red Neuronal

La red neuronal tiene una estructura secuencial sencilla y efectiva:

Capa de Entrada (Flatten): Transforma la imagen 2D de 28x28 píxeles a un vector plano de 784 entradas.

Capa Oculta (Dense): 128 neuronas con función de activación ReLU para detectar patrones en los trazos.

Capa de Salida (Dense): 10 neuronas con función de activación Softmax, donde cada una representa la probabilidad de que la imagen pertenezca a un número del 0 al 9.

## Requisitos e Instalación

Si deseas ejecutar este proyecto localmente, sigue estos pasos:

1. Clonar el repositorio

git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO


2. Crear y activar el entorno virtual

En Windows:

python -m venv venv
venv\Scripts\activate


En Mac / Linux:

python -m venv venv
source venv/bin/activate


3. Instalar las dependencias

pip install tensorflow matplotlib numpy


## Ejecución

Ejecuta el archivo principal para entrenar y evaluar el modelo:

python main.py


Al finalizar la ejecución, el script mostrará en la terminal el progreso del entrenamiento y la precisión final obtenida con el conjunto de datos de prueba.

## Resultados

Epochs de entrenamiento: 5

Precisión (Accuracy): ~97.72%

Pérdida (Loss): ~0.0773

## Autor

Proyecto creado como práctica introductoria al aprendizaje profundo (Deep Learning).