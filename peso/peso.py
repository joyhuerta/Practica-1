import tensorflow as tf
import numpy as np

kilogramos = np.array([1, 2, 5, 10, 15, 20], dtype=float)
gramos = np.array([1000, 2000, 5000, 10000, 15000, 20000], dtype=float)

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error'
)

print("Entrenando . . .")
historial = modelo.fit(kilogramos, gramos, epochs=300, verbose=False)
print("¡Modelo entrenado!")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de Pérdida")
plt.plot(historial.history["loss"])
plt.show()

print("Realiza una predicción")
resultado = modelo.predict([5])
print("El resultado es: " + str(resultado) + " gramos")

modelo.save('kg_a_gramos.h5')