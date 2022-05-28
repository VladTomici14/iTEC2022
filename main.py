from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from generate_database import GenerateDatabase
import matplotlib.pyplot as plt
from train import Trainer
from PIL import Image
import numpy as np
import tensorflow
import argparse
import cv2

# ----------- argparsing arguments --------
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path of the image.")
# ap.add_argument("-g", "--generation", required=True, default=False,
#                 help="Do you want to randomly generate the datasets for training ?")
# args = vars(ap.parse_args())

classes = ["ellipses", "rectangles", "triangles"]


def detect(model, image_path):
    image = cv2.imread(image_path)
    arrayImage = image.astype("float") / 255.0
    arrayImage = img_to_array(arrayImage)
    arrayImage = np.expand_dims(arrayImage, axis=0)
    print(f"arrayImage shape: {arrayImage.shape}")
    prediction = model.predict(arrayImage)[0]

    print(prediction)

    return classes[prediction.argmax()]

    # validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory("images/test",
    #                                                                               (96, 96),
    #                                                                               16,
    #                                                                               class_mode="categorical")
    #
    # image_batch, classes_batch = next(validation_generator)
    # predicted_batch = model.predict(image_batch)
    # for k in range(0, image_batch.shape[0]):
    #     image = image_batch[k]
    #     pred = predicted_batch[k]
    #     the_pred = np.argmax(pred)
    #     predicted = classes[the_pred]
    #     val_pred = max(pred)
    #     the_class = np.argmax(classes_batch[k])
    #     value = classes[np.argmax(classes_batch[k])]
    #     plt.figure(k)
    #     isTrue = (the_pred == the_class)
    #     plt.title(str(isTrue) + ' - class: ' + value + ' - ' + 'predicted: ' + predicted + '[' + str(val_pred) + ']')
    #     plt.imshow(image)

def detect_color(image):
    pass

def main():
    tensorflow.get_logger().setLevel('INFO')

    # ----- randomly generate the data sets ----
    generateDatabase = GenerateDatabase()

    # ----- train the NN -----
    trainer = Trainer()
    # model, xtest, ytest = trainer.train_model()
    # trainer.benchmark_accuracy(model, xtest, ytest)
    #
    # ----- delete the data sets -----

    model = load_model("model.h5")

    # ----- use the parsed image to obtain results -----
    # images = ["763-ellipse-green.png", "2100-triangle-green.png", "4301-rectangle-red.png", "4399-rectangle-blue.png"]
    # index = 0
    # for image_path in images:
    #     image = cv2.imread(f"{image_path}")
    #     # cv2.putText(image, str(detect(model, f"images/test/{image_path}")), (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
    #     index += 1
    #     detection = str(detect(model, f"{image_path}"))
    #     cv2.imshow(f"{detection}{index}", image)

    # cv2.waitKey(0)

    print(detect(model, "21-rectangle-green.png"))


if __name__ == "__main__":
    main()
