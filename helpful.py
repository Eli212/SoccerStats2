import math


def euclidean_distance(point1, point2):
    dis = math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))
    return dis
