from PIL import Image, ImageDraw
from utils import Math10thGrade, GeometricFigures
import numpy as np
import argparse
import random
import time
import math
import h5py
import os

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=False, type=str, help="do you wanna generate a test/train data set ?")
args = vars(ap.parse_args())

# TODO: restructure for 9 classes and csv
# TODO: add number of images
# TODO: generate some random images for testing

class GenerateDatabase:
    def __init__(self):
        self.figures = ["rectangle", "ellipse", "triangle"]
        self.colours = ["red", "green", "blue"]
        self.math10th = Math10thGrade()
        self.geometricFigures = GeometricFigures()

    # -------- shape, colour, name, file size -----------
    def generate_meta_data(self, final_time):
        # -------- preparing the data variables ------
        if int(final_time) == 0:
            final_time = "{:.3f}".format(final_time)
        else:
            final_time = int(final_time)

        # ------- finding the number of figures created -----
        triangles_no = len(os.listdir("images/test/triangles")) + len(os.listdir("images/train/triangles"))
        rectangles_no = len(os.listdir("images/test/rectangles")) + len(os.listdir("images/train/rectangles"))
        ellipses_no = len(os.listdir("images/test/ellipses")) + len(os.listdir("images/train/ellipses"))

        f = open("images/summarize.txt", "w")
        f.write(f"EXECUTION TIME: {final_time}s \n")
        f.write(f"TRIANGLES GENERATED: {triangles_no} \n")
        f.write(f"RECTANGLES GENERATED: {rectangles_no} \n")
        f.write(f"ELLIPSES GENERATED: {ellipses_no} \n")
        f.close()

    # ------- generating random images ---------
    def generate_random_figure(self, index, bound=False):
        random_figure = random.choice(self.figures)
        random_color = random.choice(self.colours)

        # TODO: add rotations of the rectangles
        # TODO: add noise to some pictures

        image = Image.new("RGB", (96, 96), "black")
        draw = ImageDraw.Draw(image)

        if random_figure == "rectangle":
            return self.geometricFigures.generate_random_rectangle_figures(image, draw, random_color, index, box=bound)

        elif random_figure == "ellipse":
            return self.geometricFigures.generate_random_ellipse_figures(image, draw, random_color, index, box=bound)

        elif random_figure == "triangle":
            return self.geometricFigures.generate_random_triangle_figures(image, draw, random_color, index, box=bound)

    def new_dataset(self, n_images, type):
        imgs = []
        if type == "training":
            for i in range(n_images):
                image = self.generate_random_figure(i)
                imgs.append(np.array(image))

        elif type == "testing":
            for i in range(n_images):
                image = self.generate_random_figure(i, True)
                imgs.append(np.array(image))

        return imgs

    # def generating_grid(self, images, rows, cols):
    #     assert len(images) == rows * cols
    #
    #     (w, h) = (96, 96)
    #     grid = Image.new('RGB', size=(cols * w, rows * h))
    #     grid_w, grid_h = grid.size
    #
    #     for i, img in enumerate(images):
    #         grid.paste(img, box=(i % cols * w, i // cols * h))
    #
    #     return grid

    # def save_dataset(self, images, n_images, n_bb):
    #     # -------- generating all the pictures and adding them in a single picture -----
    #     n_imgs = 10000
    #     bbs = 59
    #     # imgs, bbs = generate_database.new_dataset(n_imgs, bbs, 96)
    #
    #     # ------- preparing the dataset -------
    #     imgs_mean = np.mean(images)
    #     imgs_std = np.std(images)
    #
    #     # normalizing our images
    #     x = images - imgs_mean
    #     x /= imgs_std
    #
    #     del images
    #     y = np.array(bbs).reshape(n_images, -1)
    #
    #     del bbs
    #
    #     i = int(0.85 * n_images)
    #     x_train = x[:i]
    #     y_train = y[:i]
    #
    #     x_test = x[i:]
    #     y_test = y[i:]
    #
    #     data_path = "images/"
    #     data_images = 'images_' + str(n_images // 1000) + 'k_' + str(n_bb) + 'bb.h5'
    #
    #     xy_h5f = h5py.File(f"{data_path}{data_images}", "w")
    #     xy_h5f.create_dataset("x_train", data=x_train)
    #     xy_h5f.create_dataset("y_train", data=y_train)
    #     xy_h5f.create_dataset("imgs_mean", data=imgs_mean)
    #     xy_h5f.create_dataset("imgs_std", data=imgs_std)
    #
    #     # ------ saving the data set ------
    #     trainingset_name = 'XYdata_' + str(n_images // 1000) + 'k_' + str(n_bb) + 'bb.h5'
    #     testset_name = 'XYdata_' + str(n_images // 1000) + 'k_' + str(n_bb) + 'bb_testset.h5'
    #
    #     imgs_h5f = h5py.File(f"{data_path}{data_images}", "w")
    #
    #     imgs_h5f.create_dataset('images', data=images)
    #     imgs_h5f.close()


if __name__ == "__main__":
    generate_database = GenerateDatabase()

    initialTime = time.time()

    imgs = generate_database.new_dataset(10000, args["type"])

    print("Generated the data set !")

    # --------- saving the the database as a h5py file -------
    # generate_database.save_dataset(imgs, 10000, 10000)

    # grid = generate_database.generating_grid(images, 10, 10)
    # grid.show()

    # --------- calculating the final time -----
    final_time = time.time() - initialTime

    # ------ writing all the database generating data into a file ------
    generate_database.generate_meta_data(final_time)

# FIXME : weird error
# generate_database.py:106: RuntimeWarning: invalid value encountered in double_scalars
# distance2 = np.linalg.norm(np.cross(A - C, C - B)) / np.linalg.norm(A - C)
