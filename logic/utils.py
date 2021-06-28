import math
import logic.consts as consts


def euclidean_distance(point1, point2):
    dis = math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))
    return dis


def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)


def pixels_to_meters(val):
    return val * consts.FIELD_CM_IN_REAL_LIFE / consts.FIELD_PIXELS / 100
