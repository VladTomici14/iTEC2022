from PIL import Image, ImageDraw
import numpy as np
import random
import time
import math
import os

# TODO: maybe add graphs
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
    def generate_random_figure(self, index):
        random_figure = random.choice(self.figures)
        random_color = random.choice(self.colours)

        # TODO: add rotations of the rectangles
        # TODO: add noise to some pictures

        image = Image.new("RGB", (96, 96), "black")
        draw = ImageDraw.Draw(image)

        if random_figure == "rectangle":
            return self.generate_random_rectangle_figures(image, draw, random_color, index)

        elif random_figure == "ellipse":
            return self.generate_random_ellipse_figures(image, draw, random_color, index)

        elif random_figure == "triangle":
            return self.generate_random_triangle_figures(image, draw, random_color, index)

    def generate_random_ellipse_figures(self, image, draw, random_color, index):
        # -------- generating random points for the ellipse ---------
        x = random.choice(range(5, 70))
        y = random.choice(range(5, 70))
        w = random.choice(range(x, 90))
        h = random.choice(range(y, 90))

        # ------ making sure the up/down limits or the right/left limits are not too close
        if w - x < 10 or h - y < 10:
            return self.generate_random_ellipse_figures(image, draw, random_color, index)
        else:
            # -------- drawing and saving the image ---------
            draw.ellipse((x, y, w, h), fill=random_color)
            image.save(f"images/ellipses/{index}-ellipse-{random_color}.png")

            return image

    def generate_random_rectangle_figures(self, image, draw, random_color, index):
        # -------- generating random points for the rectangle ---------
        x = random.choice(range(5, 70))
        y = random.choice(range(5, 70))
        w = random.choice(range(x, 90))
        h = random.choice(range(y, 90))

        # ------ making sure the up/down limits or the right/left limits are not too close
        if w - x < 10 or h - y < 10:
            return self.generate_random_rectangle_figures(image, draw, random_color, index)
        else:
            # -------- drawing and saving the image ---------
            draw.rectangle((x, y, w, h), fill=random_color)  # FIXME: random lines
            image.save(f"images/rectangles/{index}-rectangle-{random_color}.png")

            return image

    def generate_random_triangle_figures(self, image, draw, random_color, index):
        # why did i build this ? maybe that s your question.
        # if not, ignore the next comment.

        # i wanted to be sure that all the 3 points are not collinear on the same line.
        # so basically, i applied some 10th grade math to be sure that something like this does not happen :)

        # -------- generating random points for the triangle ---------
        A = np.array([math.floor(random.choice(range(5, 90))), math.floor(random.choice(range(5, 90)))])
        B = np.array([math.floor(random.choice(range(5, 90))), math.floor(random.choice(range(5, 90)))])
        C = np.array([math.floor(random.choice(range(5, 90))), math.floor(random.choice(range(5, 90)))])

        # ---------- calculating if the points are collinear -------
        m = (C[0] - A[0]) * (B[1] - A[1])
        n = (C[1] - A[1]) * (B[0] - A[0])

        # ------ calculating the distance between a point and the line which goes through the other 2 points -----
        distance1 = np.linalg.norm(np.cross(B - A, A - C)) / np.linalg.norm(B - A)
        distance2 = np.linalg.norm(np.cross(A - C, C - B)) / np.linalg.norm(A - C)
        distance3 = np.linalg.norm(np.cross(C - B, B - A)) / np.linalg.norm(C - B)

        if m == n or distance1 < 10 or distance2 < 10 or distance3 < 10:
            return self.generate_random_triangle_figures(image, draw, random_color, index)
        else:
            dots = ((A[0], A[1]),
                    (B[0], B[1]),
                    (C[0], C[1]))

            # -------- drawing and saving the image ---------
            draw.polygon(dots, fill=random_color)
            image.save(f"images/triangles/{index}-triangle-{random_color}.png")

            return image

    def generating_grid(self, images, rows, cols):
        assert len(images) == rows * cols

        (w, h) = (96, 96)
        grid = Image.new('RGB', size=(cols * w, rows * h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(images):
            grid.paste(img, box=(i % cols * w, i // cols * h))

        return grid


if __name__ == "__main__":
    generate_database = GenerateDatabase()

    initialTime = time.time()

    # -------- generating all the pictures and adding them in a single picture -----
    images = []
    for i in range(10000):
        img = generate_database.generate_random_figure(i)
        images.append(img)

    # grid = generate_database.generating_grid(images, 10, 10)
    # grid.show()

    # --------- calculating the final time -----
    final_time = time.time() - initialTime

    # ------ writing all the database generating data into a file ------
    generate_database.generate_meta_data(final_time, )


# FIXME : weird error
# generate-database.py:106: RuntimeWarning: invalid value encountered in double_scalars
# distance2 = np.linalg.norm(np.cross(A - C, C - B)) / np.linalg.norm(A - C)

