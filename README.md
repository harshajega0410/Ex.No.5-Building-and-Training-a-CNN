# Ex.No.5-Building-and-Training-a-CNN
## Building and Training a CNN: Use a framework like TensorFlow or PyTorch to build and train a CNN
## Aim :To build and train a CNN 

### Procedure
1.Import the required libraries such as TensorFlow, Keras, NumPy, and Matplotlib for building, training, and visualizing the CNN model.

2.Load the CIFAR-10 dataset using datasets.cifar10.load_data(). The dataset contains 50,000 training images and 10,000 testing images belonging to 10 different classes.

3.Preprocess the images by converting the pixel values from the range 0–255 to the range 0–1 using normalization.

4.Perform one-hot encoding on the class labels using to_categorical() so that the 10 classes can be used for multiclass classification.

5.Visualize sample images by displaying 16 training images in a 4×4 grid and showing their corresponding class names.

6.Build the CNN model using convolutional layers to extract image features, max-pooling layers to reduce feature-map size, and dropout layers to reduce overfitting.

7.Flatten the extracted features and pass them through a dense layer with 512 neurons. A final dense layer with 10 neurons and softmax activation is used to classify the images into the 10 CIFAR-10 classes.

8.Compile the CNN model using the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.

9.Train the model using the training dataset for 30 epochs with a batch size of 64 and a validation split of 20%.

10.Plot the training history by displaying training and validation accuracy and loss graphs to observe the learning performance and check for overfitting.

11.Evaluate the trained model using the CIFAR-10 test dataset and obtain the test loss and test accuracy.

12.Predict test images using the trained CNN model and display the true class and predicted class for selected test images.
### PROGRAM
```
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

print("Training images:", X_train.shape)
print("Testing images :", X_test.shape)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

plt.figure(figsize=(10, 10))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    plt.imshow(X_train[i])

    plt.xlabel(
        class_names[np.argmax(y_train[i])]
    )

plt.show()

model = models.Sequential()

model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same',
        input_shape=(32, 32, 3)
    )
)

model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.MaxPooling2D(
        (2, 2)
    )
)

model.add(
    layers.Dropout(0.25)
)

model.add(
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.MaxPooling2D(
        (2, 2)
    )
)

model.add(
    layers.Dropout(0.25)
)

model.add(
    layers.Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.Conv2D(
        128,
        (3, 3),
        activation='relu',
        padding='same'
    )
)

model.add(
    layers.MaxPooling2D(
        (2, 2)
    )
)

model.add(
    layers.Dropout(0.25)
)

model.add(
    layers.Flatten()
)

model.add(
    layers.Dense(
        512,
        activation='relu'
    )
)

model.add(
    layers.Dropout(0.5)
)

model.add(
    layers.Dense(
        10,
        activation='softmax'
    )
)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2
)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.subplot(1, 2, 2)

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.show()

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=2
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


def plot_predictions(index):

    img = X_test[index]


    true_label = class_names[
        np.argmax(y_test[index])
    ]

    pred_probs = model.predict(
        np.expand_dims(img, axis=0),
        verbose=0
    )

    pred_label = class_names[
        np.argmax(pred_probs)
    ]

    plt.imshow(img)

    plt.title(
        f"True: {true_label} | Pred: {pred_label}"
    )

    plt.axis('off')

    plt.show()

plot_predictions(0)
plot_predictions(1)
plot_predictions(2)
plot_predictions(3)
```
## OUTPUT
<img width="832" height="170" alt="image" src="https://github.com/user-attachments/assets/87b52d8b-a9e0-469c-8b4a-ddf639d1fc11" />
<img width="1011" height="742" alt="image" src="https://github.com/user-attachments/assets/ec6ba21f-59fc-4f8e-bbb9-f71ac42f0509" />
<img width="752" height="750" alt="image" src="https://github.com/user-attachments/assets/ccd40935-2cda-42ad-aafd-3fce32a3aeb8" />
<img width="1286" height="682" alt="image" src="https://github.com/user-attachments/assets/a9b7ae9b-bbcb-47eb-9676-96010a30f1a6" />
<img width="1286" height="682" alt="image" src="https://github.com/user-attachments/assets/552f6c61-2122-49ef-a0ee-bffe7a617543" />
## RESULT
The Convolutional Neural Network was successfully built and trained using the TensorFlow framework. The CIFAR-10 dataset was preprocessed and used for training and testing the CNN model. The model learned important features from the images and classified them into ten different categories.
## CONCLUSION
Thus successfully implemented and trained a CNN to recognize objects across ten distinct categories using the CIFAR 10 dataset.
