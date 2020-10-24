from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# This class build a vgg-16 model from the original paper
# at https://arxiv.org/abs/1409.1556


class vgg16Model():
    def __init__(self):
        self.model = Sequential()

        self.model.add(Conv2D(filters=64, kernel_size=(
            3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        self.model.add(Conv2D(filters=64, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(filters=128, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=128, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(filters=256, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=256, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=256, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(filters=512, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=512, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=512, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(filters=512, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=512, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(filters=512, kernel_size=(
            3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

    def build(self):
        if not (self.model is None):
            return self.model
        else:
            return None
