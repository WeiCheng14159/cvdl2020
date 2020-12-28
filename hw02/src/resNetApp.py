import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from resNetModel import resnet_50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class resNetApp():
    def __init__(self):
        # Dataset class label to string
        self.asirra_dict = {0: 'cat', 1: 'dog'}
        # Dimensions of our images.
        self.img_width, self.img_height = 224, 224
        # Dataset directory
        self.train_data_dir = './dataset/train'
        self.valid_data_dir = './dataset/valid'
        # Model hyperparameter
        self.bch_size = 64
        self.lr = 0.01
        self.mome = 0.9
        self.opt = SGD(learning_rate=self.lr, momentum=self.mome)
        self.nb_epoch = 50
        # Model check point path
        self.chkp_file = './pretrain/chkp.hdf5'
        # Log directory
        self.log_dir = './log'
        # TensorBoard log path
        self.tb_log_dir = self.log_dir + '/tb_log/'
        # Progress log directory
        self.pg_log_dir = self.log_dir + '/pg_log/'
        # Inference photo index
        self.inf_idx = 0
        # Load dataset
        self.__load_dataset()
        # Build resNet50 model
        self.__build()

    # This function loads CIFAR-10 dataset
    def __load_dataset(self):

        # Use data generator
        datagen = ImageDataGenerator()

        # Automagically retrieve images and their classes for train and validation sets
        self.train_generator = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.bch_size,
            class_mode='binary')

        self.validation_generator = datagen.flow_from_directory(
            self.valid_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.bch_size,
            class_mode='binary')

    # This function builds vgg-16 model
    def __build(self):
        self.model = resnet_50()
        self.model.build(input_shape=(
            None, self.img_height, self.img_width, 3))
        self.model.summary()

    # This function loads pretrained check point / weights
    def __load_pretrained(self):
        try:
            self.model.load_weights(self.chkp_file)
        except:
            print("Fail to load saved weight at " + self.chkp_file)

    # Show 10 random images from training dataset
    def show_rand_imgs(self):
        idx = np.arange(50000)
        np.random.shuffle(idx)
        selectIdx = idx[:10]

        fig, axes = plt.subplots(
            1, 10, sharex=True, sharey=True, figsize=(20, 2))

        for i in range(10):
            img = self.x_train[selectIdx[i]]
            labelNum = self.y_train[selectIdx[i]][0]
            label = self.asirra_dict[labelNum]
            axes[i].imshow(img)
            axes[i].set_xlabel(label)
        plt.show()

    # Show progress
    def show_progress(self):
        fig, axes = plt.subplots(1, 2, sharex=True)

        for idx, fname in enumerate(['acc.jpg', 'loss.jpg']):
            img = plt.imread(self.pg_log_dir+fname)
            axes[idx].imshow(img)
            axes[idx].axis('off')
        plt.show()

    # Train the model
    def train(self):
        # Load from check point if available
        if os.path.exists(self.chkp_file):
            print("Loading pretrained weights!")
            self.__load_pretrained()

        # Compile the model
        self.model.compile(
            optimizer=self.opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Evaluate the model
        self.model.evaluate(self.validation_generator)

        # Callback functions
        # Check point callback
        cp_cb = ModelCheckpoint(
            filepath=self.chkp_file, monitor='val_accuracy', save_weights_only=True, save_best_only=True)
        # Early stopping callback
        es_cb = EarlyStopping(monitor="val_accuracy", patience=10)
        # TensorBoard callback
        tb_cb = TensorBoard(log_dir=self.tb_log_dir)

        # Training
        history = self.model.fit(x=self.train_generator, epochs=self.nb_epoch, validation_data=self.validation_generator,
                                 callbacks=[cp_cb, es_cb, tb_cb])

        print(history)

    def get_inference_index(self, idxStr):
        try:
            self.inf_idx = int(idxStr)
        except:
            return

    # Model inference
    def inference(self):
        # Load pretrained weights
        self.__load_pretrained()

        # Load test image
        test_image = self.x_test[self.inf_idx]
        test_image_4d = np.reshape(test_image, (1, 32, 32, 3))

        # Predict
        predicted_class_probabilities = self.model.predict(test_image_4d)

        # Concatenate prob and label
        classes = []
        probs = predicted_class_probabilities[0]
        for idx, _ in enumerate(probs):
            classes.append(self.asirra_dict[idx])

        # Finalize prediction
        predicted_class = np.argmax(predicted_class_probabilities)
        print("Prediction: ", self.asirra_dict[predicted_class], "  Expected: ",
              self.asirra_dict[self.y_test[self.inf_idx][0]], " Confidence: ", probs[predicted_class])

        # Show predicted results
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        img = self.x_test[self.inf_idx]
        labelNum = self.y_test[self.inf_idx][0]
        label = self.asirra_dict[labelNum]
        axes[0].imshow(img)
        axes[0].set_xlabel(label)
        axes[1].bar(classes, probs)
        plt.setp(axes[1].get_xticklabels(), rotation='vertical')
        plt.show()


if __name__ == "__main__":
    g = resNetApp()
    g.train()
