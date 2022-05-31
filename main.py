from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from generate_database import GenerateDatabase
import matplotlib.pyplot as plt
from train import Trainer
from PIL import Image, ImageFilter
import numpy as np
import tensorflow
import argparse
import cv2
import os

# ----------- argparsing arguments --------
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path of the image.")
# ap.add_argument("-t", "--train", required=True, type=bool, default=False, help="Do you want to train the model right now ?")
args = vars(ap.parse_args())

classes = ["ellipseB",
           "ellipseG",
           "ellipseR",
           "rectangleB",
           "rectangleG",
           "rectangleR",
           "triangleB",
           "triangleG",
           "triangleR"]


def detect(model, image_path):
    # ----- preparing the image -----
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # -------- transforming the image into an array -------
    arrayImage = image.astype("float") / 255.0
    arrayImage = img_to_array(arrayImage)
    arrayImage = np.expand_dims(arrayImage, axis=0)

    # ----- making the predictions -------
    predictions = model.predict(arrayImage)[0]

    # -------- printing the results ----------
    print(f"\n< ------ RESULTS: {image_path} ------ >")
    k = 0
    for prediction in predictions:
        prediction_percentage = float(prediction * 100000) / 1000
        print(f"{classes[k]}: {prediction_percentage}% \n")
        k += 1

    result = classes[np.argmax(predictions)]

    # this could have been better implemented instead of 6 ifs :)
    if result.find("rectangle"):
        shape = "rectangle"
    elif result.find("ellipse"):
        shape = "ellipse"
    elif result.find("triangle"):
        shape = "triangle"

    if result[len(result) - 1] == "R":
        color = "red"
    elif result[len(result) - 1] == "G":
        color = "green"
    elif result[len(result) - 1] == "B":
        color = "blue"

    return color, shape, result


def calculate_area(image, color, shape):
    # ------- the HSV values for each color -------
    boundaries = [([0, 100, 100], [20, 100, 100]),
                  ([100, 100, 100], [120, 100, 100]),
                  ([220, 100, 100], [240, 100, 100])]

    color_values = []

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        color_values.append(np.sum(mask == 255))

    print(color_values)

    if color == "red":
        return float(100 * color_values[0] / (96 * 96))
    if color == "green":
        return float(100 * color_values[1] / (96 * 96))
    if color == "blue":
        return float(100 * color_values[2] / (96 * 96))


def plotting_results(folder_path, model):
    fig = plt.figure(figsize=(10, 7))

    rows = 2
    columns = 6

    k = 0
    for file in os.listdir(folder_path):
        fig.add_subplot(rows, columns, k + 1)

        image = cv2.imread(f"pictures/{file}")
        plt.imshow(image)
        plt.axis("off")
        color, shape, result = detect(model, f"pictures/{file}")
        plt.title(result)
        k += 1

    plt.show()


def main():
    tensorflow.get_logger().setLevel('INFO')

    # ----- randomly generate the data sets ----
    generateDatabase = GenerateDatabase()

    # ----- train the NN -----
    trainer = Trainer()
    model, xtest, ytest = trainer.train_model()

    # or we can load the model
    # model = load_model("model.h5")  # cause we used a callback

    color, shape, result = detect(model, args["image"])
    image = cv2.imread(args["image"])
    area = calculate_area(image, color, shape)

    plotting_results("pictures", model)

    print(f"Area: {area}")


if __name__ == "__main__":
    main()

"""

Pregatire manuale pentru saptamana viitoare.
1) bilet cu ce manuale sunt, nume, numar etc,
2) 11:45 gala olimpicilor 
"""
