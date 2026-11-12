import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Cargar la base de datos MNIST (los números escritos a mano)
mnist = tf.keras.datasets.mnist

# 2. Separar los datos en "datos para aprender" y "datos para examninar"
(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = mnist.load_data()

# 3. Monstrar la primera imagen del conjunto de entrenamiento
plt.imshow(x_entrenamiento[0], cmap='gray')
plt.title(f"Este número es un: {y_entrenamiento[0]}")
plt.show()
