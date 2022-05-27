from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import tensorflow

colors_labels = ["red", "green", "blue"]
shape_labels = ["triangles", "rectangles", "ellipses"]

filter_size = 3
pool_size = 2

img_size = (96, 96)

# --------------- data set augmentation ---------
train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                          rotation_range=5,
                                          validation_split=0.2,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          vertical_flip=True,
                                          horizontal_flip=True,
                                          fill_mode="nearest")

# ---------- loading data set ----------
training_set_path = "images/"
training_set = train_data_generator.flow_from_directory(training_set_path,
                                                        target_size=(96, 96),
                                                        batch_size=64,
                                                        class_mode="categorical",
                                                        subset="training")

model = Sequential([
    Convolution2D(32, 6, 6, input_shape=X.shape[1:], dim_ordering='tf', activation='relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Convolution2D(64, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'),
    Flatten(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(y.shape[-1])
])

# -------- metrics for the compilation --------
METRICS = [
        tensorflow.keras.metrics.BinaryAccuracy(name="accuracy"),
        tensorflow.keras.metrics.Precision(name="precision"),
        tensorflow.keras.metrics.Recall(name="recall"),
        tensorflow.keras.metrics.AUC(name="auc")
]

# -------------- model compiling ---------
model.compile(optimizer="adadelta",
              loss="mse",
              metrics=METRICS)
