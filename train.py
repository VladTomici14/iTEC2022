from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow
import xlrd
import csv
import os


# TODO: test the speed in colab
# TODO: implement some graphs for callbacks

class Trainer:
    def __init__(self):
        self.classes = ["rectangleR",
                        "rectangleG",
                        "rectangleB",
                        "ellipseR",
                        "ellipseG",
                        "ellipseB",
                        "triangleR",
                        "triangleG",
                        "triangleB"]

    def train_model(self):
        """
        The main function that will do the training.

        :return:
            @:param model: The trained model.
            @:param xtest:
            @:param ytest:
        """

        # ---------------- callbacks ---------
        lr_reduce = ReduceLROnPlateau(monitor="val_loss", patience=20, verbose=1, factor=0.50, min_lr=1e-10)
        model_checkpoint = ModelCheckpoint("model.h5")
        early_stopping = EarlyStopping(verbose=1, patience=20)
        CALLBACKS = [lr_reduce, model_checkpoint, early_stopping]

        # -------------- metrics -----------
        METRICS = [
            tensorflow.keras.metrics.BinaryAccuracy(name="accuracy"),
            tensorflow.keras.metrics.Precision(name="precision"),
            tensorflow.keras.metrics.Recall(name="recall"),
            tensorflow.keras.metrics.AUC(name="auc")]

        # ---------- declaring the variables ------
        n_classes = len(self.classes)
        image_size = (96, 96, 3)

        # ---------- declaring the DenseNet model ---------
        base_model = tensorflow.keras.applications.DenseNet121(input_shape=image_size,
                                                               include_top=False,
                                                               weights="imagenet")

        # -------- adding layers to the model -------
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        preds = Dense(n_classes, activation="softmax")(x)

        model = tensorflow.keras.models.Model(inputs=base_model.inputs,
                                              outputs=preds)
        print("Created the model !")

        # -------- modifying layers of the model --------
        for layer in base_model.layers[:-8]:
            layer.trainable = False

        for layer in base_model.layers[-8:]:
            layer.trainable = True

        # ------ plotting the model into an image -------
        tensorflow.keras.utils.plot_model(model, "model_layers.png", True)
        print("Saved the models's layers !")

        # -------- compiling the model -------
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=METRICS)
        print("Compiled the model !")

        # model.summary()

        # ---------- data augmentation ---------
        training_set_path = "images/train"
        validation_set_path = "images/test"

        train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                                  zoom_range=0.2,
                                                  rotation_range=5,
                                                  horizontal_flip=True)
        validation_data_generator = ImageDataGenerator(rescale=1. / 255)

        training_set = train_data_generator.flow_from_directory(training_set_path,
                                                                target_size=(96, 96),
                                                                batch_size=16,
                                                                # TODO: find what represents the batch size and fine tune it
                                                                class_mode="categorical")
        validation_set = validation_data_generator.flow_from_directory(validation_set_path,
                                                                       target_size=(96, 96),
                                                                       batch_size=16,
                                                                       class_mode="categorical")

        (xtrain, ytrain) = training_set.next()
        (xtest, ytest) = validation_set.next()

        print("Loaded the data sets !")



        # -------- fitting the model ----------
        history = model.fit(training_set,
                            validation_data=validation_set,
                            steps_per_epoch=50,
                            epochs=5,
                            verbose=2,
                            callbacks=CALLBACKS)

        print("Fitted the model ! \n")
        print("\nThe training is done !")

        return model, xtest, ytest

    def benchmark_accuracy(self, model, xtest, ytest):
        """
        This functions calculates the accuracy of the training.

            :param model: the model. (could you believe that ? me neither..)
            :param xtest:
            :param ytest:

        :return: prints the accuracy alongside with some results
        """

        ypred = model.predict(xtest)

        total = 0
        accurate = 0
        accurate_index = []
        wrong_index = []

        for i in range(len(ypred)):
            if np.argmax(ypred[i]) == np.argmax(ytest[i]):
                accurate += 1
                accurate_index.append(i)
            else:
                wrong_index.append(i)

            total += 1

        print(f"Total test data: {total}")
        print(f"Accurately predicted data: {accurate}")
        print(f"Wrongly predicted data: {total - accurate}")
        print(f"Accuracy: {round(accurate / total * 100, 3)}%")


if __name__ == "__main__":
    # ---- disabling TF wanings, cause i don't have a gpu and they're annoying ;-; ----
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    trainer = Trainer()

    model, xtest, ytest = trainer.train_model()
    print("Trained the model !")

    print("Results of the benchmark: ")
    trainer.benchmark_accuracy(model, xtest, ytest)
