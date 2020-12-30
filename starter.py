# import track
# from old import tracking
import logic.outputs as outputs
import playerteam

import matplotlib.pyplot as plt

import math
import random
import track
import logic.Classes as Classes
import logic.CONSTS as CONSTS
import old.tracking as tracking
import numpy as np

import cv2


def separate_players_and_ball(frames_arr):
    players_arr = {}
    ball = Classes.BallSeparated()
    for idx_frame, frame in enumerate(frames_arr):
        for player in frame.players:
            if player.number not in players_arr:
                players_arr[player.number] = Classes.PlayerSeparated(player.number)
            players_arr[player.number].location_in_frames[idx_frame] = (player.point_x, player.point_y)
            players_arr[player.number].player_box[idx_frame] = [player.tl_x, player.tl_y, player.height, player.width]
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
            if players_arr[player_number].location_in_frames_perspective[idx_frame] is not None \
                    and players_arr[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                # if player_number == 21:
                #     print(euclidean_distance(players_arr[player_number].location_in_frames_perspective[idx_frame],
                #                        players_arr[player_number].location_in_frames_perspective[idx_frame + 1]))
                players_arr[player_number].distance_covered += \
                    euclidean_distance(players_arr[player_number].location_in_frames_perspective[idx_frame],
                                       players_arr[player_number].location_in_frames_perspective[idx_frame + 1])

    # print stats
    # for player_number in players_arr:
    #     print("player - ", players_arr[player_number].number, "  player_distance  - ",
    #           players_arr[player_number].distance_covered)


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
                px = (h[0][0] * game.players[player_number].location_in_frames[frame_number][0] + h[0][1] *
                      game.players[player_number].location_in_frames[frame_number][1] + h[0][2]) / (
                         (h[2][0] * game.players[player_number].location_in_frames[frame_number][0] + h[2][1] *
                          game.players[player_number].location_in_frames[frame_number][1] + h[2][2]))
                py = (h[1][0] * game.players[player_number].location_in_frames[frame_number][0] + h[1][1] *
                      game.players[player_number].location_in_frames[frame_number][1] + h[1][2]) / (
                         (h[2][0] * game.players[player_number].location_in_frames[frame_number][0] + h[2][1] *
                          game.players[player_number].location_in_frames[frame_number][1] + h[2][2]))
                game.players[player_number].location_in_frames_perspective[frame_number] = (int(px), int(py))
            else:
                game.players[player_number].location_in_frames_perspective[frame_number] = None

    for frame_number in game.ball.location_in_frames:
        if game.ball.location_in_frames[frame_number] is not None:
            px = (h[0][0] * game.ball.location_in_frames[frame_number][0] + h[0][1] *
                  game.ball.location_in_frames[frame_number][1] + h[0][2]) / (
                     (h[2][0] * game.ball.location_in_frames[frame_number][0] + h[2][1] *
                      game.ball.location_in_frames[frame_number][1] + h[2][2]))
            py = (h[1][0] * game.ball.location_in_frames[frame_number][0] + h[1][1] *
                  game.ball.location_in_frames[frame_number][1] + h[1][2]) / (
                     (h[2][0] * game.ball.location_in_frames[frame_number][0] + h[2][1] *
                      game.ball.location_in_frames[frame_number][1] + h[2][2]))
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


def get_avg_distance_in_frame_of_all_players(game):
    sum_distance = 0
    count_of_sums = 0
    for player_number in game.players:
        for idx_frame in range(CONSTS.MAX_FRAMES - 1):
            if game.players[player_number].location_in_frames_perspective[idx_frame] is not None \
                    and game.players[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                sum_distance += \
                    euclidean_distance(game.players[player_number].location_in_frames_perspective[idx_frame],
                                       game.players[player_number].location_in_frames_perspective[idx_frame + 1])
                count_of_sums += 1

    # print("Avg distance of all players between each frame: " + str(sum_distance / count_of_sums))
    return sum_distance / count_of_sums


def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)


def fix_players_zig_zags(game, loop_times=1):
    for _ in range(loop_times):
        for player_number in game.players:
            for idx_frame in range(CONSTS.MAX_FRAMES - 1):
                if game.players[player_number].location_in_frames_perspective[idx_frame] is not None \
                        and game.players[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                    if euclidean_distance(game.players[player_number].location_in_frames_perspective[idx_frame],
                                          game.players[player_number].location_in_frames_perspective[idx_frame + 1]) \
                            > CONSTS.MAX_ZIG_ZAGS:
                        # print(midpoint(game.players[player_number].location_in_frames_perspective[idx_frame],
                        #                game.players[player_number].location_in_frames_perspective[idx_frame + 1]))
                        game.players[player_number].location_in_frames_perspective[idx_frame + 1] = \
                            midpoint(game.players[player_number].location_in_frames_perspective[idx_frame],
                                     game.players[player_number].location_in_frames_perspective[idx_frame + 1])


def identify_players_team(game, vid_path):
    cap = cv2.VideoCapture(vid_path)
    dominant_color_arr = []
    for player_number in game.players:
        player_avg_color = []
        # do this for avg in case the random frame hits two players in box
        for i in range(3):
            player_box_frames_keys = game.players[player_number].player_box.keys()
            player_box_random_len = random.randint(0, len(player_box_frames_keys)-1)
            player_random_box_frame = list(player_box_frames_keys)[player_box_random_len]
            current_box = game.players[player_number].player_box[player_random_box_frame]
            current_box = [int(x) for x in current_box]
            cap.set(1, player_random_box_frame)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            player_frame = frame[current_box[1]:current_box[1]+current_box[2], current_box[0]:int(current_box[0]+current_box[3]/2), :]
            player_dominant_color = playerteam.get_dominant_color(player_frame)
            player_avg_color.append(player_dominant_color)
        dominant_color_arr.append(playerteam.get_avg_color(player_avg_color))
    teams_arr = playerteam.get_kmeans_teams_arr(dominant_color_arr)
    print(list(game.players.keys()))
    print(teams_arr)
    for idx, player_number in enumerate(game.players):
        game.players[player_number].team = teams_arr[idx]


if __name__ == '__main__':
    max_frames = 178
    vid_path = 'sources/TestVideos/vid2.mp4'
    txt_file_name = "demofile2.txt"
    field_img = cv2.imread('sources/TestImages/maracana_homemade.png')

    # frames = track.start_tracking()
    # outputs.write_frames_to_file(frames, txt_file_name)

    frames = outputs.read_from_file_to_frame(txt_file_name)
    # tracking.start_vid(vid_path, field_img, frames, max_frames)

    maracana_game = separate_players_and_ball(frames)
    # tracking.start_vid(vid_path, field_img, game, max_frames)
    change_perspective(maracana_game)
    maracana_game.players = delete_out_of_field_players(maracana_game.players, field_img)
    stats_players_distance_covered(maracana_game.players)
    get_avg_distance_in_frame_of_all_players(maracana_game)
    fix_players_zig_zags(maracana_game, 1000)
    get_avg_distance_in_frame_of_all_players(maracana_game)

    identify_players_team(maracana_game, vid_path)
    # save videos:
    # outputs.create_vid_one_player(maracana_game.players[21], field_img)
    outputs.create_vid_all_player(maracana_game, field_img)

    # print(maracana_game.players[92].location_in_frames_perspective)
    # for player_number in game.players:
    #     print(game.players[player_number].distance_covered)
