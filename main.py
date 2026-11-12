import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Cargar la base de datos MNIST (los números escritos a mano)
mnist = tf.keras.datasets.mnist

# 2. Separar los datos en "datos para aprender" y "datos para examninar"
(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = mnist.load_data()

# 3. Monstrar la primera imagen del conjunto de entrenamiento
#plt.imshow(x_entrenamiento[0], cmap='gray')
#plt.title(f"Este número es un: {y_entrenamiento[0]}")
#plt.show()

# 3. Normalizar los datos
# Los colores de los píxeles van del 0 al 255. Los dividimos entre 255.0 para que quede entre 0 y 1.
# A las redes neuronales les cuesta digerir número grandes; con valores entre 0 y 1 aprenden mucho más rápido.
x_entrenamiento = x_entrenamiento / 255.0
x_prueba = x_prueba / 255.0

# 4. Construir el "cerebro" (La red neuronal)
modelo = tf.keras.models.Sequential([
    #  Capa de entrada: Aplana la imagen (que es un cuadrito de 28x28) a una sola fila larga de 784 píxeles.
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Capa de pensamiento (oculta): 128 neuronas que buscan los trazos de los números.
    tf.keras.layers.Dense(128, activation='relu'),

    # Capa de salida: Exactamente 10 neuronas, una para cada posible respuesta (del 0 al 9).
    tf.keras.layers.Dense(10, activation='softmax')
])
print("¡Red neuronal construida exitosamente")

# 5. Configurar el método de aprendizaje (Compilar el modelo)
modelo.compile(
    optimizer='adam', # El "maestro" que ajusta los errores para que mejore en cada intento
    loss='sparse_categorical_crossentropy', # La regla para medir qué tan equivocada está la red
    metrics=['accuracy'] # Queremos que nos muestre el porcentaje de aciertos (precisión)
)

# 6. ¡A estudiar! (Entrenar la red)
print("\n--- Iniciando el entrenamiento ---")
# 'epochs=5' significa que la red repasará todos los datos 5 veces
modelo.fit(x_entrenamiento, y_entrenamiento, epochs=5)

# 7. El examen final (Evaluar con datos que nunca ha visto)
print("\n--- Evaluando con el examen final ---")
perdida, precision = modelo.evaluate(x_prueba, y_prueba)
print(f"\n¡Listo! La precisión de tu red neuronal es del: {precision * 100:.2f}%")

modelo.save('modelo_numeros.keras')
print("Modelo guardado exitosamente en 'modelo_numeros.keras'")
