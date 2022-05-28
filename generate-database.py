from PIL import Image, ImageDraw
import numpy as np
import argparse
import random
import time
import math
import h5py
import os

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", required=True, type=str, help="do you wanna generate a test/train data set ?")
args = vars(ap.parse_args())

# TODO: ask about the way the files are arranged
# TODO: ask about if we are interested in the ROI of the shapes
# TODO: write a new class that creates the ROI shapes
# TODO: generate bounding boxes

class GenerateDatabase:
    def __init__(self):
        self.figures = ["rectangle", "ellipse", "triangle"]
        self.colours = ["red", "green", "blue"]

    # -------- shape, colour, name, file size -----------
    def generate_meta_data(self, final_time):
        # -------- preparing the data variables ------
        if int(final_time) == 0:
            final_time = "{:.3f}".format(final_time)
        else:
            final_time = int(final_time)

        # ------- finding the number of figures created -----
        triangles_no = len(os.listdir("images/triangles"))
        rectangles_no = len(os.listdir("images/rectangles"))
        ellipses_no = len(os.listdir("images/ellipses"))

        f = open("images/summarize.txt", "w")
        f.write(f"EXECUTION TIME: {final_time}s \n")
        f.write(f"TRIANGLES GENERATED: {triangles_no} \n")
        f.write(f"RECTANGLES GENERATED: {rectangles_no} \n")
        f.write(f"ELLIPSES GENERATED: {ellipses_no} \n")
        f.close()

    # ------- generating random images ---------
    def generate_random_figure(self, index, bound = False):
        random_figure = random.choice(self.figures)
        random_color = random.choice(self.colours)

        # TODO: add rotations of the rectangles
        # TODO: add noise to some pictures

        image = Image.new("RGB", (96, 96), "black")
        draw = ImageDraw.Draw(image)

        if random_figure == "rectangle":
            return self.generate_random_rectangle_figures(image, draw, random_color, index, box=bound)

        elif random_figure == "ellipse":
            return self.generate_random_ellipse_figures(image, draw, random_color, index, box=bound)

        elif random_figure == "triangle":
            return self.generate_random_triangle_figures(image, draw, random_color, index, box=bound)

    def new_dataset(self, n_images, type):
        if type == "training":
            imgs = []
            for i in range(n_images):
                image = self.generate_random_figure(i)
                # TODO: generate bounding boxes
                imgs.append(np.array(image))
            return imgs

        elif type == "testing":
            bounding_boxes = []
            for i in range(n_images / 2):
                box = self.generate_random_box(i)
                bounding_boxes.append(box)
            return bounding_boxes

        else:
            print("Please select a valid type of data set generating ! ")
            pass

    def generate_random_ellipse_figures(self, image, draw, random_color, index, box=False):
        # -------- generating random points for the ellipse ---------
        x = random.choice(range(5, 70))
        y = random.choice(range(5, 70))
        w = random.choice(range(x, 90))
        h = random.choice(range(y, 90))

        # ------ making sure the up/down limits or the right/left limits are not too close
        if w - x < 10 or h - y < 10:
            return self.generate_random_ellipse_figures(image, draw, random_color, index, box)
        else:
            # -------- drawing and saving the image ---------
            draw.ellipse((x, y, w, h), fill=random_color)

            if box:
                image = image.crop((x, y, x+w, y+h))
                image.save(f"images/test/ellipses/{index}-ellipse-{random_color}.png")

            else:
                image.save(f"images/train/ellipses/{index}-ellipse-{random_color}.png")

            return image

    def generate_random_rectangle_figures(self, image, draw, random_color, index, box=False):
        # -------- generating random points for the rectangle ---------
        x = random.choice(range(5, 70))
        y = random.choice(range(5, 70))
        w = random.choice(range(x, 90))
        h = random.choice(range(y, 90))

        # ------ making sure the up/down limits or the right/left limits are not too close
        if w - x < 10 or h - y < 10:
            return self.generate_random_rectangle_figures(image, draw, random_color, index, box)
        else:
            # -------- drawing and saving the image ---------
            draw.rectangle((x, y, w, h), fill=random_color)

            if box:
                image = image.crop(x, y, x+w, y+h)
                image.save(f"images/test/rectangles/{index}-rectangle-{random_color}.png")

            else:
                image.save(f"images/train/rectangles/{index}-rectangle-{random_color}.png")

            return image

    def generate_random_triangle_figures(self, image, draw, random_color, index, box=False):
        # why did i build this ? maybe that s your question.
        # if not, ignore the next comment.

        # i wanted to be sure that all the 3 points are not collinear on the same line.
        # so basically, i applied some 10th grade math to be sure that something like this does not happen :)

        # -------- generating random points for the triangle ---------
        A = np.array([math.floor(random.choice(range(5, 90))), math.floor(random.choice(range(5, 90)))])
        B = np.array([math.floor(random.choice(range(5, 90))), math.floor(random.choice(range(5, 90)))])
        C = np.array([math.floor(random.choice(range(5, 90))), math.floor(random.choice(range(5, 90)))])
        points = [A, B, C]

        # ---------- calculating if the points are collinear -------
        m = (C[0] - A[0]) * (B[1] - A[1])
        n = (C[1] - A[1]) * (B[0] - A[0])

        # ------ calculating the distance between a point and the line which goes through the other 2 points -----
        distance1 = np.linalg.norm(np.cross(B - A, A - C)) / np.linalg.norm(B - A)
        distance2 = np.linalg.norm(np.cross(A - C, C - B)) / np.linalg.norm(A - C)
        distance3 = np.linalg.norm(np.cross(C - B, B - A)) / np.linalg.norm(C - B)

        if m == n or distance1 < 10 or distance2 < 10 or distance3 < 10:
            return self.generate_random_triangle_figures(image, draw, random_color, index, box)
        else:
            dots = ((A[0], A[1]),
                    (B[0], B[1]),
                    (C[0], C[1]))

            # -------- drawing and saving the image ---------
            draw.polygon(dots, fill=random_color)

            if box:
                (minX, maxX) = (0, 100)
                (minY, maxY) = (0, 100)
                for (x, y) in points:
                    if x < minX: minX = x
                    if x > maxX: maxX = x
                    if y < minY: minY = y
                    if y > minY: minY = y

                image = image.crop(minX, minY, maxX, maxY)

                image.save(f"images/test/triangles/{index}-triangle-{random_color}.png")
            else:
                image.save(f"images/train/triangles/{index}-triangle-{random_color}.png")

            return image

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
# generate-database.py:106: RuntimeWarning: invalid value encountered in double_scalars
# distance2 = np.linalg.norm(np.cross(A - C, C - B)) / np.linalg.norm(A - C)
