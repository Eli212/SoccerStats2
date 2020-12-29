# import track
# from old import tracking
import math

import track
import logic.Classes as Classes
import logic.CONSTS as CONSTS
import old.tracking as tracking
import numpy as np

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
            players.append(Classes.Player(int(player_info[0]), float(player_info[1]), float(player_info[2]),
                                          float(player_info[3]), float(player_info[4]), float(player_info[5]),
                                          float(player_info[6])))
        current_frame = Classes.Frame(idx_frame, players)
        current_frame.ball = frame_ball
        frames_arr.append(current_frame)

    f.close()
    print("Finished reading from file: " + file_name)
    return frames_arr


def separate_players_and_ball(frames_arr):
    players_arr = {}
    ball = Classes.BallSeparated()
    for idx_frame, frame in enumerate(frames_arr):
        for player in frame.players:
            if player.number not in players_arr:
                players_arr[player.number] = Classes.PlayerSeparated(player.number)
            players_arr[player.number].location_in_frames[idx_frame] = (player.point_x, player.point_y)
        ball.location_in_frames[idx_frame] = frame.ball

    for player_number in players_arr:
        for idx_frame in range(CONSTS.MAX_FRAMES):
            if idx_frame not in players_arr[player_number].location_in_frames:
                players_arr[player_number].location_in_frames[idx_frame] = None

    # Sort if you want
    for player_number in players_arr:
        players_arr[player_number].sort_location_in_frames()

    return Classes.Game("Maracana", players_arr, ball)


def stats_players_distance_covered(players_arr):
    for player_number in players_arr:
        for idx_frame in range(CONSTS.MAX_FRAMES - 1):
            if players_arr[player_number].location_in_frames_perspective[idx_frame] is not None\
                    and players_arr[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                players_arr[player_number].distance_covered += \
                    euclidean_distance(players_arr[player_number].location_in_frames_perspective[idx_frame],
                                       players_arr[player_number].location_in_frames_perspective[idx_frame + 1])
        # player_distance = player_distance / 100

    for player_number in players_arr:
        print("player - ", players_arr[player_number].number, "  player_distance  - ",
              players_arr[player_number].distance_covered)


def euclidean_distance(point1, point2):
    dis = math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))
    return dis


def change_perspective(game):
    points = ["B", "Y", "Y2", "C", "P", "Q", "V2", "W2", "VW2"]

    pts1 = np.float32([CONSTS.MARACANA_FIELD_POINTS[points[0]],
                       CONSTS.MARACANA_FIELD_POINTS[points[1]],
                       CONSTS.MARACANA_FIELD_POINTS[points[2]],
                       CONSTS.MARACANA_FIELD_POINTS[points[3]],
                       CONSTS.MARACANA_FIELD_POINTS[points[4]],
                       CONSTS.MARACANA_FIELD_POINTS[points[5]],
                       CONSTS.MARACANA_FIELD_POINTS[points[6]],
                       CONSTS.MARACANA_FIELD_POINTS[points[7]],
                       CONSTS.MARACANA_FIELD_POINTS[points[8]]])
    pts2 = np.float32([CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[0]],
                       CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[1]],
                       CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[2]],
                       CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[3]],
                       CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[4]],
                       CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[5]],
                       CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[6]],
                       CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[7]],
                       CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[8]]])

    h, status = cv2.findHomography(pts1, pts2)
    # im_dst = cv2.warpPerspective(current_frame, h, (field_img.shape[1], field_img.shape[0]))
    #
    # initCamera = cv2.initCameraMatrix2D(pts1, pts2, (1090, 1080))
    # print(initCamera)
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    for player_number in game.players:
        for frame_number in game.players[player_number].location_in_frames:
            if game.players[player_number].location_in_frames[frame_number] is not None:
                px = (h[0][0] * game.players[player_number].location_in_frames[frame_number][0] + h[0][1] * game.players[player_number].location_in_frames[frame_number][1] + h[0][2]) / (
                    (h[2][0] * game.players[player_number].location_in_frames[frame_number][0] + h[2][1] * game.players[player_number].location_in_frames[frame_number][1] + h[2][2]))
                py = (h[1][0] * game.players[player_number].location_in_frames[frame_number][0] + h[1][1] * game.players[player_number].location_in_frames[frame_number][1] + h[1][2]) / (
                    (h[2][0] * game.players[player_number].location_in_frames[frame_number][0] + h[2][1] * game.players[player_number].location_in_frames[frame_number][1] + h[2][2]))
                game.players[player_number].location_in_frames_perspective[frame_number] = (int(px), int(py))
            else:
                game.players[player_number].location_in_frames_perspective[frame_number] = None

    for frame_number in game.ball.location_in_frames:
        if game.ball.location_in_frames[frame_number] is not None:
            px = (h[0][0] * game.ball.location_in_frames[frame_number][0] + h[0][1] * game.ball.location_in_frames[frame_number][1] + h[0][2]) / (
                (h[2][0] * game.ball.location_in_frames[frame_number][0] + h[2][1] * game.ball.location_in_frames[frame_number][1] + h[2][2]))
            py = (h[1][0] * game.ball.location_in_frames[frame_number][0] + h[1][1] * game.ball.location_in_frames[frame_number][1] + h[1][2]) / (
                (h[2][0] * game.ball.location_in_frames[frame_number][0] + h[2][1] * game.ball.location_in_frames[frame_number][1] + h[2][2]))
            game.ball.location_in_frames_perspective[frame_number] = (int(px), int(py))
        else:
            game.ball.location_in_frames_perspective[frame_number] = None


def delete_out_of_field_players(players_arr, field_img):
    players_arr_copy = players_arr.copy()
    sum_deleted_players = 0
    for player_number in players_arr:
        score = 0
        for frame in players_arr[player_number].location_in_frames:
            if players_arr[player_number].location_in_frames_perspective[frame] is not None:
                if check_in_frame(players_arr[player_number].location_in_frames_perspective[frame], field_img.shape):
                    score += 1
                else:
                    score -= 1
        if score < 0:
            # print("deleted: " + str(player_number) + " score: " + str(score))
            sum_deleted_players += 1
            del players_arr_copy[player_number]
    print("Deleted {0} players outside of frame".format(sum_deleted_players))
    return players_arr_copy


def check_in_frame(point, frame_shape):
    if 0 < point[0] < frame_shape[1] and 0 < point[1] < frame_shape[0]:
        return True
    return False


if __name__ == '__main__':
    max_frames = 178
    vid_path = 'sources/TestVideos/vid2.mp4'
    txt_file_name = "demofile2.txt"
    field_img = cv2.imread('sources/TestImages/maracana_homemade.png')

    # frames = track.start_tracking()
    # write_frames_to_file(frames, txt_file_name)

    frames = read_from_file_to_frame(txt_file_name)
    # tracking.start_vid(vid_path, field_img, frames, max_frames)

    maracana_game = separate_players_and_ball(frames)
    # tracking.start_vid(vid_path, field_img, game, max_frames)
    change_perspective(maracana_game)
    maracana_game.players = delete_out_of_field_players(maracana_game.players, field_img)
    stats_players_distance_covered(maracana_game.players)
    print(maracana_game.players[5].location_in_frames_perspective)
    # for player_number in game.players:
    #     print(game.players[player_number].distance_covered)
