import numpy as np
import random
import math

class Math10thGrade:
    def __init__(self):
        pass

    def calculate_triangle_area(self, A, B, C):
        a = self.calculate_line_length_2_points(A, B)
        b = self.calculate_line_length_2_points(B, C)
        c = self.calculate_line_length_2_points(C, A)
        p = (a + b + c) // 2

        return int(math.sqrt(p * (p - a) * (p - b) * (p - c)))

    def calculate_line_length_2_points(self, p1, p2):
        return int(math.sqrt(math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2)))

    def calculate_ellipse_area(self, x, y, w, h):
        width = w - x
        height = h - y

        return int(math.pi * (width // 2) * (height // 2))

class GeometricFigures:
    def __init__(self):
        self.math10th = Math10thGrade()

    def generate_random_ellipse_figures(self, image, draw, random_color, index, box=False):
        # -------- generating random points for the ellipse ---------
        x = random.choice(range(5, 70))
        y = random.choice(range(5, 70))
        w = random.choice(range(x, 90))
        h = random.choice(range(y, 90))

        # ------ making sure the up/down limits or the right/left limits are not too close
        if w - x < 10 or h - y < 10 or self.math10th.calculate_ellipse_area(x, y, w, h) < 30:
            return self.generate_random_ellipse_figures(image, draw, random_color, index, box)
        else:
            # -------- drawing and saving the image ---------
            draw.ellipse((x, y, w, h), fill=random_color)

            if box:
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
        if w - x < 10 or h - y < 10 or (w - x) * (h - y) < 30:
            return self.generate_random_rectangle_figures(image, draw, random_color, index, box)
        else:
            # -------- drawing and saving the image ---------
            draw.rectangle((x, y, w, h), fill=random_color)

            if box:
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

        if m == n or distance1 < 10 or distance2 < 10 or distance3 < 10 or self.math10th.calculate_triangle_area(A, B, C) < 50:
            return self.generate_random_triangle_figures(image, draw, random_color, index, box)
        else:
            dots = ((A[0], A[1]),
                    (B[0], B[1]),
                    (C[0], C[1]))

            # -------- drawing and saving the image ---------
            draw.polygon(dots, fill=random_color)
            if box:
                image.save(f"images/test/triangles/{index}-triangle-{random_color}.png")
            else:
                image.save(f"images/train/triangles/{index}-triangle-{random_color}.png")

            return image