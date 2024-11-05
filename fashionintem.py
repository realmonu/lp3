import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load dataset
train_data = pd.read_csv('fashion-mnist_train.csv')
test_data = pd.read_csv('fashion-mnist_test.csv')

# Separate features and labels
X_train = train_data.iloc[:, 1:].values / 255.0
Y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values / 255.0
Y_test = test_data.iloc[:, 0].values

# One-hot encode labels
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

# Reshape input data to match image format (28x28 pixels, 1 channel)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

#2

history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_test, Y_test))

#3

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")


#4

class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict_classes(X):
    predictions = model.predict(X)
    return np.argmax(predictions, axis=1)

predicted_classes = predict_classes(X_test)
actual_classes = np.argmax(Y_test, axis=1)

# Plot actual vs predicted images
def plot_images(images, actual, predicted, class_labels):
    plt.figure(figsize=(12, 12))
    for i in range(10):
        plt.subplot(5, 5, i+1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Actual: {class_labels[actual[i]]}\nPredicted: {class_labels[predicted[i]]}", fontsize=10)
        plt.axis('off')
        plt.subplots_adjust(hspace=0.5)  # Adjust the space between plots
    plt.show()


plot_images(X_test, actual_classes, predicted_classes, class_labels)

# Calculate accuracy
accuracy = np.sum(predicted_classes == actual_classes) / actual_classes.size
print(f"Accuracy: {accuracy * 100:.2f}%")

