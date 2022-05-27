from PIL import Image, ImageDraw
import numpy as np
import random
import math


class GenerateDatabase:
    def __init__(self):
        self.figures = ["rectangle", "ellipse", "triangle"]
        self.colours = ["red", "green", "blue"]

    # -------- shape, colour, name, file size -----------
    def generate_meta_data(self):
        # TODO: generate the meta data into a csv
        pass

    # ------- generating random images ---------
    def generate_random_figure(self, index):
        random_figure = random.choice(self.figures)
        random_color = random.choice(self.colours)

        # TODO: add rotations of the rectangles
        # TODO: add noise to the backgrounds

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

        # ------ calculating the distance between C and the AB line -----
        distance = np.linalg.norm(np.cross(B - A, A - C)) / np.linalg.norm(B - A)

        if m == n or distance < 5:
            return self.generate_random_triangle_figures(image, draw, random_color, index)
        else:
            dots = ((A[0], A[1]),
                    (B[0], B[1]),
                    (C[0], C[1]))

            # -------- drawing and saving the image ---------
            draw.polygon(dots, fill=random_color)
            image.save(f"images/triangles/{index}-triangle-{random_color}.png")

            return image

    # TODO: delete me when done
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

    # -------- generating all the pictures and adding them in a single picture -----
    images = []
    for i in range(100):
        img = generate_database.generate_random_figure(i)
        images.append(img)

    grid = generate_database.generating_grid(images, 10, 10)
    grid.show()
