import tensorflow as tf
import numpy as np

tf.random.set_seed(1)
X_train = np.array([[1], [2], [3], [4], [5],[6]], dtype=float)
y_train = np.array([[0], [0], [0], [0], [1],[1]], dtype=float)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, verbose=0)
model.summary()

model.evaluate(X_train,  y_train, verbose=2)

predictions = model.predict(X_train)
print("Predictions:", predictions)

for layer in model.layers:
    weights, biases = layer.get_weights()
    print("weights::", weights)
    print("biases:", biases)
