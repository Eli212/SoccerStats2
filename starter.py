import math

import track
import logic.Classes as CLASSES
from old import tracking

import cv2


def write_frames_to_file(frames_arr, file_name):
    f = open(file_name, "w")
    f.write(str(len(frames_arr)) + "\n")
    for idx_frame, frame in enumerate(frames_arr):
        print(idx_frame)
        if frame.ball is None:
            f.write("None\n")
        else:
            f.write(str(int(frame.ball[0])) + " " + str(int(frame.ball[1])) + "\n")
        f.write(str(len(frame.players)) + "\n")
        for idx_player, players in enumerate(frame.players):
            f.write(players.__str__() + "\n")

    f.close()


def read_from_file_to_frame(file_name):
    print("Starting to read from file: " + file_name)
    frames_arr = []
    f = open(file_name, "r")
    num_of_frames = int(f.readline())
    for idx_frame in range(num_of_frames):
        # print(idx_frame)
        frame_ball = None
        ball_line = f.readline()
        if ball_line != "None\n":
            # print(ball_line)
            ball_line = ball_line.split(" ")
            frame_ball = (int(ball_line[0]), int(ball_line[1]))
        players = []
        num_of_players = int(f.readline())
        for idx_players in range(num_of_players):
            player_info = f.readline()
            player_info = player_info.split(" ")
            players.append(CLASSES.Player(int(player_info[0]), float(player_info[1]), float(player_info[2]),
                                          float(player_info[3]), float(player_info[4]), float(player_info[5]),
                                          float(player_info[6])))
        current_frame = CLASSES.Frame(idx_frame, players)
        current_frame.ball = frame_ball
        frames_arr.append(current_frame)

    f.close()
    print("Finished reading from file: " + file_name)
    return frames_arr


def separate_players(frames_arr):
    players_arr = {}
    for idx_frame, frame in enumerate(frames_arr):
        for player in frame.players:
            if str(player.number) not in players_arr:
                players_arr[str(player.number)] = CLASSES.PlayerSeparated(player.number)
            players_arr[str(player.number)].location_in_frames[str(idx_frame)] = (player.point_x, player.point_y)
        distance()
    print(players_arr["3"].location_in_frames)

def distance(point1, point2):
    dis = math.sqrt(((point1[0]-point2[0])**2)+((point1[1]-point2[1])**2))
    print("point 1", point1)
    print("point ", point2)
    print("dis = ", dis)

if __name__ == '__main__':
    max_frames = 178
    vid_path = 'sources/TestVideos/vid2.mp4'
    txt_file_name = "demofile2.txt"
    field_img = cv2.imread('sources/TestImages/maracana_homemade.png')

    # frames = track.start_tracking()
    # write_frames_to_file(frames, txt_file_name)

    frames = read_from_file_to_frame(txt_file_name)
    # tracking.start_vid(vid_path, field_img, frames, max_frames)

    separate_players(frames)
