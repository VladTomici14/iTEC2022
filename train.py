from tensorflow.keras.layers import BatchNormalization, Dense, Reshape, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import tensorflow
import numpy as np
import h5py
import cv2

n_imgs = 100

classes = {0: "-",
           1: "rectangleR",
           2: "rectangleG",
           3: "rectangleB",
           4: "ellipseR",
           5: "ellipseG",
           6: "ellipseB",
           7: "triangleR",
           8: "triangleG",
           9: "triangleB"}

# --------- loading the data set -------
# dataset_name = "images_10k_10000bb.h5"
# xy_h5py = h5py.File(f"images/{dataset_name}", "r")
#
# x = np.array(xy_h5py["x_train"])
# y = np.array(xy_h5py["y_train"])
# images_mean = np.array(xy_h5py["imgs_mean"])
# images_std = np.array(xy_h5py["imgs_std"])

# del x, y
#
# xy_h5py.close()

# ------ declaring some fine tuning parameters for the model --------
# n_examples = x.shape[0]
# side_dim = x.shape[1]
# n_bb = y.shape[1] // 5
n_classes = 10
img_size = (96, 96, 3)
base_model_name = "DenseNet201"

# --------- declaring the model, based on DenseNet -------
base_model = tensorflow.keras.applications.DenseNet121(input_shape=img_size,
                                                       include_top=False)

for layer in base_model.layers:
    layer.trainable = False

# ------- adding more layers ------

base_model_input = base_model.output
model = Sequential()
model.add(base_model)

# flatten = Flatten()(base_model)
# model.add(flatten)
model.add(Flatten())

# batch_normalization = BatchNormalization()(flatten)
# model.add(batch_normalization)
model.add(BatchNormalization())

# class_categorical = Dense(10, activation="softmax")(batch_normalization)
# model.add(class_categorical)
model.add(Dense(10, activation="softmax"))

# class_output = Reshape((n_bb, n_classes), name="class_output")(class_categorical)
#
# score_confidence = Dense((n_bb), name="score_confidence", activation="tanh")(batch_normalization)
# score_coords = Dense((n_bb * 4), name="score_coords")(batch_normalization)

model.summary()

tensorflow.keras.utils.plot_model(model, "model_layers.png", True)

# --------------- data set augmentation ---------
# TODO: delete/edit for fine tuning
train_data_generator = ImageDataGenerator(rotation_range=5,
                                          validation_split=0.2,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          vertical_flip=True,
                                          horizontal_flip=True,
                                          fill_mode="nearest")

valid_data_generator = ImageDataGenerator(rescale=1. / 255,
                                          validation_split=0.2)

# ---------- loading data set ----------
training_set_path = "images/train"
training_set = train_data_generator.flow_from_directory(training_set_path,
                                                        target_size=(96, 96),
                                                        batch_size=64,
                                                        class_mode="categorical",
                                                        subset="training")

# -------- metrics for the compilation --------
METRICS = [
    tensorflow.keras.metrics.BinaryAccuracy(name="accuracy"),
    tensorflow.keras.metrics.Precision(name="precision"),
    tensorflow.keras.metrics.Recall(name="recall"),
    tensorflow.keras.metrics.AUC(name="auc")
]

# -------------- model compiling ---------
# model.compile()
