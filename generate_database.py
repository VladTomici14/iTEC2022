from PIL import Image, ImageDraw
from utils import Math10thGrade, GeometricFigures
import random
import time


class GenerateDatabase:
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
        self.R = (255, 0, 0)
        self.G = (0, 255, 0)
        self.B = (0, 0, 255)
        self.math10th = Math10thGrade()
        self.geometricFigures = GeometricFigures()

    # -------- shape, colour, name, file size -----------
    def generate_meta_data(self, final_time, n_train):
        # -------- preparing the data variables ------
        if int(final_time) == 0:
            final_time = "{:.3f}".format(final_time)
        else:
            final_time = int(final_time)

        f = open("images/summarize.txt", "w")
        f.write(f"GENERATION TIME: {final_time}s \n")
        f.write(f"TRAINING DATA SET: {n_train} \n")
        f.write(f"VALIDATION DATA SET: {n_train // 5} \n")
        f.close()

    # ------- generating random images ---------
    def generate_random_figure(self, index, bound=False):
        x = random.randint(0, 9)
        random_figure = self.classes[x-1]

        image = Image.new("RGB", (96, 96), "black")
        draw = ImageDraw.Draw(image)

        if random_figure.find("triangle"):
            if random_figure[len(random_figure) - 1] == "R":
                self.geometricFigures.generate_random_triangle_figures(image, draw, self.R, index, bound)
                return index, "triangle", "R"

            if random_figure[len(random_figure) - 1] == "G":
                self.geometricFigures.generate_random_triangle_figures(image, draw, self.G, index, bound)
                return index, "triangle", "G"

            if random_figure[len(random_figure) - 1] == "B":
                self.geometricFigures.generate_random_triangle_figures(image, draw, self.B, index, bound)
                return index, "triangle", "B"

        elif random_figure.find("rectangle"):
            if random_figure[len(random_figure) - 1] == "R":
                self.geometricFigures.generate_random_rectangle_figures(image, draw, self.R, index, bound)
                return index, "rectangle", "R"

            if random_figure[len(random_figure) - 1] == "G":
                self.geometricFigures.generate_random_rectangle_figures(image, draw, self.G, index, bound)
                return index, "rectangle", "G"

            if random_figure[len(random_figure) - 1] == "B":
                self.geometricFigures.generate_random_rectangle_figures(image, draw, self.B, index, bound)
                return index, "rectangle", "B"

        elif random_figure.find("ellipse"):
            if random_figure[len(random_figure) - 1] == "R":
                self.geometricFigures.generate_random_ellipse_figures(image, draw, self.R, index, bound)
                return index, "ellipse", "R"

            if random_figure[len(random_figure) - 1] == "G":
                self.geometricFigures.generate_random_ellipse_figures(image, draw, self.G, index, bound)
                return index, "ellipse", "G"

            if random_figure[len(random_figure) - 1] == "B":
                self.geometricFigures.generate_random_ellipse_figures(image, draw, self.B, index, bound)
                return index, "ellipse", "B"

    def new_dataset(self, n_images):
        for i in range(n_images):
            image, shape, color = self.generate_random_figure(i, False)

        for i in range(n_images):
            image, shape, color = self.generate_random_figure(i, True)


if __name__ == "__main__":
    generate_database = GenerateDatabase()

    initialTime = time.time()

    generate_database.new_dataset(10000)

    print("Generated the data set !")

    # --------- calculating the final time -----
    final_time = time.time() - initialTime

    # ------ writing all the database generating data into a file ------
    generate_database.generate_meta_data(final_time, 10000)
