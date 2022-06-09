import tensorflow as tf
import keras

import os
import random
import shutil
import pathlib

from keras_preprocessing.image import ImageDataGenerator


def create_folder_structure():
    shutil.rmtree('./train')
    pathlib.Path("./train/train/cats").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./train/validate/cats").mkdir(parents=True, exist_ok=True)


def copy_images(source_list, destination_path):
    for image in source_list:
        shutil.copyfile(f'./input_data/{image}', f'./train/{destination_path}/{image}')


def create_training_and_validation_set(train_validation_split):
    cat_images = os.listdir('./input_data')

    random.shuffle(cat_images)

    split_index = int(len(cat_images) * train_validation_split)

    training_cats = cat_images[:split_index]
    validation_cats = cat_images[split_index:]

    create_folder_structure()
    copy_images(training_cats, 'train/cats')
    copy_images(validation_cats, 'validate/cats')


def train_model():
    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_iterator = train_gen.flow_from_directory('./train/train',
                                                   target_size=(150, 150),
                                                   batch_size=20,
                                                   class_mode='binary')

    validation_gen = ImageDataGenerator(rescale=1. / 255.0)
    validation_iterator = validation_gen.flow_from_directory('./train/validate',
                                                             target_size=(150, 150),
                                                             batch_size=10,
                                                             class_mode='binary')

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units=512, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

    model.fit(train_iterator,
              validation_data=validation_iterator)

    model.save('cats.h5')


create_training_and_validation_set(0.8)
train_model()
