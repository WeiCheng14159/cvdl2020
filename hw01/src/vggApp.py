import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from vgg16Model import vgg16Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


class vggApp():
    def __init__(self):
        # Map CIFAR-10 class label to string
        self.cifar10_dict = {0: 'airplane', 1: 'automobile', 2: 'bird',
                             3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        # Model hyperparameter
        self.bch_size = 128
        self.lr = 0.01
        self.mome = 0.9
        self.opt = SGD(learning_rate=self.lr, momentum=self.mome)
        # Model check point path
        self.chkp_file = './pretrain/chkp.hdf5'
        # Log directory
        self.log_dir = './log'
        # TensorBoard log path
        self.tb_log_dir = self.log_dir+'/tb_log/'
        # Progress log directory
        self.pg_log_dir = self.log_dir + '/pg_log/'
        # Inference photo index
        self.inf_idx = 0
        # Load dataset
        self.__load_dataset()
        # Build vgg16 model
        self.__build_vgg16()

    # This function loads CIFAR-10 dataset
    def __load_dataset(self):
        (self.x_train, self.y_train), (self.x_test,
                                       self.y_test) = tf.keras.datasets.cifar10.load_data()

    # This function builds vgg-16 model
    def __build_vgg16(self):
        self.model = vgg16Model().build()

    # This function loads pretrained check point / weights
    def __load_pretrained(self):
        try:
            self.model.load_weights(self.chkp_file)
        except:
            print("Fail to load saved weight at " + self.chkp_file)

    # Print model structure
    def show_model_structure(self):
        self.model.summary()

    # Print out model hyperparameter
    def show_hyperparemeter(self):
        print("Hyperparameters:")
        print("Batch size = ", self.bch_size)
        print("Optimizer = ", self.opt._name)
        print("Learning rate = ", self.lr)
        print("Momentum = ", self.mome)

    # Show 5 random images from training dataset
    def show_rand_imgs(self):
        idx = np.arange(50000)
        np.random.shuffle(idx)
        selectIdx = idx[:5]

        fig, axes = plt.subplots(1, 5, sharex=True)

        for i in range(5):
            img = self.x_train[selectIdx[i]]
            labelNum = self.y_train[selectIdx[i]][0]
            label = self.cifar10_dict[labelNum]
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

        self.model.compile(
            optimizer=self.opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Evaluate the model
        loss, acc = self.model.evaluate(self.x_test, self.y_test)

        # Callback functions
        # Check point callback
        cp_cb = ModelCheckpoint(
            filepath=self.chkp_file, monitor='val_acc', save_weights_only=True, save_best_only=True)
        # Early stopping callback
        es_cb = EarlyStopping(monitor="val_acc", patience=10)
        # TensorBoard callback
        tb_cb = TensorBoard(log_dir=self.tb_log_dir)

        # Training
        history = self.model.fit(self.x_train, self.y_train, batch_size=self.bch_size, epochs=30, validation_data=(
            self.x_test, self.y_test), callbacks=[cp_cb, es_cb, tb_cb])

        # Save images of training progress
        # Loss and Validation Loss
        for k in ['loss', 'val_loss']:
            print(history.history[k])
            plt.plot(history.history[k])
        plt.title('loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.savefig(self.pg_log_dir+'loss.jpg')

        # Accuracy and Validation Accuracy
        plt.figure()
        for k in ['acc', 'val_acc']:
            print(history.history[k])
            plt.plot(history.history[k])
        plt.title('acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['acc', 'val_acc'], loc='upper left')
        plt.savefig(self.pg_log_dir+'acc.jpg')

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
            classes.append(self.cifar10_dict[idx])

        # Finalize prediction
        predicted_class = np.argmax(predicted_class_probabilities)
        print("Prediction: ", self.cifar10_dict[predicted_class], "  Expected: ",
              self.cifar10_dict[self.y_test[self.inf_idx][0]], " Confidence: ", probs[predicted_class])

        # Show predicted results
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        img = self.x_test[self.inf_idx]
        labelNum = self.y_test[self.inf_idx][0]
        label = self.cifar10_dict[labelNum]
        axes[0].imshow(img)
        axes[0].set_xlabel(label)
        axes[1].bar(classes, probs)
        plt.setp(axes[1].get_xticklabels(), rotation='vertical')
        plt.show()


if __name__ == "__main__":
    g = vggApp()
    g.train()
