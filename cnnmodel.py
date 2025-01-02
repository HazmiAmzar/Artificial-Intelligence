# Two lines that remove tensorflow GPU logs
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from math import ceil
from tensorflow import keras
from sklearn import model_selection
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay



# Loads csv files and appends pixels to X and labels to y
def preprocess_data():
    data = pd.read_csv('fer2013.csv')
    labels = pd.read_csv('fer2013new.csv')

    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    n_samples = len(data)
    w = 48
    h = 48

    y = np.array(labels[orig_class_names])
    X = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        X[i] = np.fromstring(data[' pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

    return X, y


def clean_data_and_normalize(X, y):
    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = y_mask < orig_class_names.index('unknown')
    X = X[mask]
    y = y[mask]

    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    X = X / 255.0

    return X, y


def split_data(X, y):
    test_size = ceil(len(X) * 0.1)

    # Split Data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size, random_state=42)
    return x_train, y_train, x_val, y_val, x_test, y_test


def data_augmentation(x_train):
    shift = 0.1
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        height_shift_range=shift,
        width_shift_range=shift)
    datagen.fit(x_train)
    return datagen


def show_augmented_images(datagen, x_train, y_train):
    it = datagen.flow(x_train, y_train, batch_size=1)
    plt.figure(figsize=(10, 7))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(it.next()[0][0], cmap='gray')
        # plt.xlabel(class_names[y_train[i]])
    show_augmented_images(datagen, x_train, y_train)
    plt.show()

def define_model(input_shape=(48, 48, 1), classes=7):
    num_features = 64

    model = Sequential([
    # 1st stage
    layers.Input(shape=(48,48,1)),
    layers.Conv2D(num_features, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation(activation='relu'),
    layers.Conv2D(num_features, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation(activation='relu'),
    layers.Dropout(0.5),

    # 2nd stage
    layers.Conv2D(num_features, (3, 3), activation='relu'),
    layers.Conv2D(num_features, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # 3rd stage
    layers.Conv2D(2 * num_features, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation(activation='relu'),
    layers.Conv2D(2 * num_features, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation(activation='relu'),
    
    # 4th stage
    layers.Conv2D(2 * num_features, (3, 3), activation='relu'),
    layers.Conv2D(2 * num_features, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
    # 5th stage
    layers.Conv2D(4 * num_features, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation(activation='relu'),
    layers.Conv2D(4 * num_features, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation(activation='relu'),

    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),

    layers.Dense(classes, activation='softmax')
    ])

    return model


def plot_acc_loss(history):
    # Plot accuracy graph
    plt.plot(history.history['accuracy'], label='training_accuracy')
    plt.plot(history.history['val_accuracy'], label='validation_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.ylim([0, 1.0])
    plt.legend(loc='upper left')
    plt.show()

    # Plot loss graph
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def evaluation(fer_classes, y_pred_classes, y_true_classes):
    # Print classification report
    print(classification_report(y_true_classes, y_pred_classes, target_names=fer_classes))
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fer_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def run_model():
    fer_classes = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    X, y = preprocess_data()
    X, y = clean_data_and_normalize(X, y)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(X, y)
    datagen = data_augmentation(x_train)

    epochs = 100
    batch_size = 64

    print("X_train shape: " + str(x_train.shape))
    print("Y_train shape: " + str(y_train.shape))
    print("X_test shape: " + str(x_test.shape))
    print("Y_test shape: " + str(y_test.shape))
    print("X_val shape: " + str(x_val.shape))
    print("Y_val shape: " + str(y_val.shape))

    # Training model from scratch
    model = define_model(input_shape=x_train[0].shape, classes=len(fer_classes))
    #model.summary()
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights = True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

    # Start the timer
    start_time = time.time()
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_val, y_val), verbose=2, callbacks=[callback, reduce_lr])
    # Stop the timer
    end_time = time.time()
    #,steps_per_epoch=len(x_train) // batch_size
    test_acc, test_loss = model.evaluate(x_test, y_test, batch_size=batch_size)

    # Calculate the elapsed time
    training_time = (end_time - start_time) / 60
    print(f"Time taken for model training: {training_time:.4f} minutes")

    # Generate predictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Save the model
    model.save('cnnmodel.keras')
    print('Model saved to cnnmodel.keras')

    return fer_classes, history, y_pred_classes, y_true_classes, test_acc, test_loss


fer_classes, history, y_pred_classes, y_true_classes, test_acc, test_loss = run_model()

evaluation(fer_classes, y_pred_classes, y_true_classes)
plot_acc_loss(history)