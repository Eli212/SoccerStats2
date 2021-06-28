import collections
import logic.utils as utils


class Line:
    def __init__(self, the_line, degrees):
        # point is (x, y)
        self.the_line = the_line
        self.degrees = degrees


class IntersectionPoint:
    def __init__(self, point, thetas, lines):
        self.point = point
        self.in_group = False
        self.thetas = thetas
        self.lines = lines
        self.separated = True
        self.best_match_name = None
        self.best_match_score = -9999


class FieldPoint:
    def __init__(self, point_name):
        self.point_name = point_name
        self.points = []

    def add_point(self, field_point_image_info):
        self.points.append(field_point_image_info)


class FieldPointImageInfo:
    def __init__(self, image_height, image_width, point):
        self.image_height = image_height
        self.image_width = image_width
        self.point = point

    def point_in_detected_field_point(self, point):
        if self.point[0] < point[0] < self.point[0] + self.image_height and\
                self.point[1] < point[1] < self.point[1] + self.image_width:
            return 1
        return -1

    def __str__(self):
        return "Height: " + str(self.image_height) + ". Width: " + str(self.image_width) + \
               ". Point: (" + str(self.point[0]) + "," + str(self.point[1]) + ")"


class Frame:
    """
    'ball' is (x, y) point of the bottom center
    """
    def __init__(self, frame_number, players):
        self.frame_number = frame_number
        self.players = players
        self.ball = None


class Player3D:
    """
    'tl_x' and 'tl_y' are the point in top left.
    'point' is the bottom center.
    """
    def __init__(self, number, tl_x, tl_y, height, width, point_x, point_y):
        self.number = number
        self.tl_x = tl_x
        self.tl_y = tl_y
        self.height = height
        self.width = width
        self.point_x = point_x
        self.point_y = point_y

    def __str__(self):
        return str(self.number) + " " + str(self.tl_x) + " " + str(self.tl_y) + " " + str(self.height) + " " + \
               str(self.width) + " " + str(self.point_x) + " " + str(self.point_y)


class Player2D:
    def __init__(self, number):
        self.number = number
        self.location_in_frames = {}
        self.location_in_frames_perspective = {}
        self.player_box = {}
        self.distance_covered = 0
        self.team = 0
        self.is_active = True
        self.in_frame = []
        self.out_frame = []
        self.potential_identities = []
        self.heat_lines = []

    def sort_location_in_frames(self):
        location_in_frames_sorted = collections.OrderedDict(sorted(self.location_in_frames.items()))
        self.location_in_frames = location_in_frames_sorted

    def compute_heat_lines_color(self, max_value):
        for heat_line in self.heat_lines:
            heat_line.compute_color(max_value)


class BallSeparated:
    def __init__(self):
        self.location_in_frames = {}
        self.location_in_frames_perspective = {}
        self.player_with_ball_in_frames_perspective = {}


class Game:
    def __init__(self, name, players, ball):
        self.name = name
        self.players = players
        self.ball = ball

        self.frame_width = 0
        self.frame_height = 0


class HeatLine:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.dis = utils.euclidean_distance(start, end)
        self.color = None

    def compute_color(self, max_value, min_value=0):
        if self.dis <= int((min_value + max_value) / 2):
            r = 255
            g = int(255 * self.dis / int((min_value + max_value) / 2))
            b = 0
        else:
            r = int(255 * (max_value - self.dis) / int((min_value + max_value) / 2))
            g = 255
            b = 0
        self.color = (r, g, b)
