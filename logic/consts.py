import numpy as np
import cv2

# Kernel Consts
KERNEL_STRONG = np.ones((15, 15), np.uint8)
KERNEL_LIGHT = np.ones((5, 5), np.uint8)
KERNEL_VERY_LIGHT = np.ones((2, 2), np.uint8)

# Circle Consts
# Radius of circle
CIRCLE_RADIUS = 1

# Blue color in BGR
TEAM_A = (255, 0, 0)
TEAM_B = (80, 127, 255)
CIRCLE_COLOR2 = (0, 255, 0)

# Line thickness of 2 px
CIRCLE_THICKNESS = 10

# Text Consts
# Font
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Font Scale
TEXT_FONTSCALE = 0.5

# Red color in BGR
TEXT_COLOR = (0, 0, 255)

# Line thickness of 2 px
TEXT_THICKNESS = 2

# Field Points (x, y)
FIELD_POINTS = {
    "B": (1203, 57),
    "O": (988, 160),
    "P": (1203, 160),
    "T": (988, 328),
    "U": (1132, 302),
    "V": (1203, 302),
    "O2": (988, 57),
    "S": (988, 519),
    "X": (1132, 542),
    "W": (1203, 542),
    "Y": (650, 57),
    "Y2": (650, 790),
    "C": (1203, 790)
}

# Field Points Resized(x, y)
# FIELD_POINTS = {
#     "B": (1763, 70),
#     "O": (1445, 200),
#     "P": (1203, 160),
#     "T": (1445, 413),
#     "U": (1658, 380),
#     "V": (1203, 302),
#     "O2": (1445, 70),
#     "S": (988, 519),
#     "X": (1132, 542),
#     "W": (1203, 542)
# }
MARACANA_HOMEMADE_FIELD_POINTS = {
    "B": (875, 7),
    "Y": (434, 7),
    "Y2": (434, 457),
    "C": (875, 457),
    "P": (875, 53),
    "Q": (875, 411),
    "V2": (875, 174),
    "W2": (875, 290),
    "VW2": (875, 232)
}

MARACANA_FIELD_POINTS = {
    "B": (980, 103),
    "Y": (155, 320),
    "Y2": (1303, 1186),
    "C": (1783, 285),
    "P": (1040, 110),
    "Q": (1701, 258),
    "V2": (1261, 143),
    "W2": (1414, 149),
    "VW2": (1331, 161)
}

MARACANA_FIELD_POINTS2 = {
    "B": (493, 50),
    "Y": (140, 120),
    "Y2": (658, 450),
    "C": (824, 133),
    "P": (518, 54),
    "Q": (787, 120),
    "V2": (613, 74),
    "W2": (675, 87),
    "VW2": (639, 80)
}

# Game Maxes
MAX_FRAMES = 349
MAX_PLAYER_ZIG_ZAGS = 0.1
MAX_PLAYER_JUMPING = 120
MAX_BALL_JUMPING = 150
MAX_DISTANCE_BALL_PLAYER_CONNECTION = 25
