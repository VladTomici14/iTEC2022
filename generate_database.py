from PIL import Image, ImageDraw
from utils import Math10thGrade, generate_random_triangle_figures, generate_random_ellipse_figures, generate_random_rectangle_figures
import random
import time
import os

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
        random_figure = random.choice(self.classes)

        image = Image.new("RGB", (96, 96), "black")
        draw = ImageDraw.Draw(image)

        if random_figure.find("triangle"):
            if random_figure[len(random_figure) - 1] == "R":
                generate_random_triangle_figures(image, draw, self.R, index, bound)
                return index, "triangle", "R"

            if random_figure[len(random_figure) - 1] == "G":
                generate_random_triangle_figures(image, draw, self.G, index, bound)
                return index, "triangle", "G"

            if random_figure[len(random_figure) - 1] == "B":
                generate_random_triangle_figures(image, draw, self.B, index, bound)
                return index, "triangle", "B"

        elif random_figure.find("rectangle"):
            if random_figure[len(random_figure) - 1] == "R":
                generate_random_rectangle_figures(image, draw, self.R, index, bound)
                return index, "rectangle", "R"

            if random_figure[len(random_figure) - 1] == "G":
                generate_random_rectangle_figures(image, draw, self.G, index, bound)
                return index, "rectangle", "G"

            if random_figure[len(random_figure) - 1] == "B":
                generate_random_rectangle_figures(image, draw, self.B, index, bound)
                return index, "rectangle", "B"

        if random_figure == "ellipseR":
            generate_random_ellipse_figures(image, draw, self.R, index, bound)
            return index, "ellipse", "R"

        if random_figure == "ellipseG":
            generate_random_ellipse_figures(image, draw, self.G, index, bound)
            return index, "ellipse", "G"

        if random_figure == "ellipseB":
            print("shit")
            generate_random_ellipse_figures(image, draw, self.B, index, bound)
            return index, "ellipse", "B"

    def new_dataset(self, n_images):
        for i in range(n_images):
            image, shape, color = self.generate_random_figure(i, False)

        for i in range(n_images):
            image, shape, color = self.generate_random_figure(i, True)

    def generate_folders(self):
        try:
            os.mkdir("images/train/ellipse-red")
            os.mkdir("images/train/ellipse-green")
            os.mkdir("images/train/ellipse-blue")
            os.mkdir("images/train/triangle-red")
            os.mkdir("images/train/triangle-green")
            os.mkdir("images/train/triangle-blue")
            os.mkdir("images/train/rectangle-red")
            os.mkdir("images/train/rectangle-green")
            os.mkdir("images/train/rectangle-blue")
        except Exception:
            print("folders already present!")

if __name__ == "__main__":
    generate_database = GenerateDatabase()

    initialTime = time.time()

    generate_database.generate_folders()

    generate_database.new_dataset(10000)


    print("Generated the data set !")

    # --------- calculating the final time -----
    final_time = time.time() - initialTime

    # ------ writing all the database generating data into a file ------
    generate_database.generate_meta_data(final_time, 10000)
