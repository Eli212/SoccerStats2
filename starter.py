# import track
# from old import tracking
import matplotlib as plt
import os
import logic.outputs as outputs
from logic import utils, playerteam, statistics, track, userInterface

import logic.classes as classes
import logic.consts as consts
import numpy as np

import cv2


def separate_players_and_ball(frames_arr):
    players_arr = {}
    ball = classes.BallSeparated()
    for idx_frame, frame in enumerate(frames_arr):
        for player in frame.players:
            if player.number not in players_arr:
                players_arr[player.number] = classes.PlayerSeparated(player.number)
            players_arr[player.number].location_in_frames[idx_frame] = (player.point_x, player.point_y)
            players_arr[player.number].player_box[idx_frame] = [player.tl_x, player.tl_y, player.height, player.width]
        ball.location_in_frames[idx_frame] = frame.ball

    for player_number in players_arr:
        for idx_frame in range(consts.MAX_FRAMES):
            if idx_frame not in players_arr[player_number].location_in_frames:
                players_arr[player_number].location_in_frames[idx_frame] = None

    # Sort if you want
    for player_number in players_arr:
        players_arr[player_number].sort_location_in_frames()

    return classes.Game("Maracana", players_arr, ball)


def change_perspective(game):
    points = ["B", "Y", "Y2", "C", "P", "Q", "V2", "W2", "VW2"]

    field_points = consts.MARACANA_FIELD_POINTS2

    pts1 = np.float32([field_points[points[0]],
                       field_points[points[1]],
                       field_points[points[2]],
                       field_points[points[3]],
                       field_points[points[4]],
                       field_points[points[5]],
                       field_points[points[6]],
                       field_points[points[7]],
                       field_points[points[8]]])
    pts2 = np.float32([consts.MARACANA_HOMEMADE_FIELD_POINTS[points[0]],
                       consts.MARACANA_HOMEMADE_FIELD_POINTS[points[1]],
                       consts.MARACANA_HOMEMADE_FIELD_POINTS[points[2]],
                       consts.MARACANA_HOMEMADE_FIELD_POINTS[points[3]],
                       consts.MARACANA_HOMEMADE_FIELD_POINTS[points[4]],
                       consts.MARACANA_HOMEMADE_FIELD_POINTS[points[5]],
                       consts.MARACANA_HOMEMADE_FIELD_POINTS[points[6]],
                       consts.MARACANA_HOMEMADE_FIELD_POINTS[points[7]],
                       consts.MARACANA_HOMEMADE_FIELD_POINTS[points[8]]])

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

            # set location to None if it out of boundaries
            if int(px) < 0 or int(py) < 0:
                game.ball.location_in_frames_perspective[frame_number] = None
            else:
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


def fix_players_zig_zags(game, loop_times=1):
    for _ in range(loop_times):
        for player_number in game.players:
            for idx_frame in range(consts.MAX_FRAMES - 1):
                if game.players[player_number].location_in_frames_perspective[idx_frame] is not None \
                        and game.players[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                    if utils.euclidean_distance(game.players[player_number].location_in_frames_perspective[idx_frame],
                                                game.players[player_number].location_in_frames_perspective[
                                                      idx_frame + 1]) \
                            > consts.MAX_PLAYER_ZIG_ZAGS:
                        # print(midpoint(game.players[player_number].location_in_frames_perspective[idx_frame],
                        #                game.players[player_number].location_in_frames_perspective[idx_frame + 1]))
                        game.players[player_number].location_in_frames_perspective[idx_frame + 1] = \
                            utils.midpoint(game.players[player_number].location_in_frames_perspective[idx_frame],
                                           game.players[player_number].location_in_frames_perspective[idx_frame + 1])


def fix_ball_zig_zags(game, loop_times=1):
    for _ in range(loop_times):
        for idx_frame in range(consts.MAX_FRAMES - 1):
            if game.ball.location_in_frames_perspective[idx_frame] is not None \
                    and game.ball.location_in_frames_perspective[idx_frame + 1] is not None:
                if utils.euclidean_distance(game.ball.location_in_frames_perspective[idx_frame],
                                            game.ball.location_in_frames_perspective[idx_frame + 1]) \
                        > consts.MAX_PLAYER_ZIG_ZAGS:
                    # print(midpoint(game.players[player_number].location_in_frames_perspective[idx_frame],
                    #                game.players[player_number].location_in_frames_perspective[idx_frame + 1]))
                    game.ball.location_in_frames_perspective[idx_frame + 1] = \
                        utils.midpoint(game.ball.location_in_frames_perspective[idx_frame],
                                       game.ball.location_in_frames_perspective[idx_frame + 1])


def fill_empty_frames(game):
    # PLAYERS
    for player_number in game.players:
        count_frames = 0
        player_none = 0
        valid_fill_frames = False
        while count_frames < consts.MAX_FRAMES:
            if game.players[player_number].location_in_frames_perspective[count_frames] is None:
                player_none += 1
            elif game.players[player_number].location_in_frames_perspective[count_frames] is not\
                    None and player_none != 0 and valid_fill_frames:
                if utils.euclidean_distance(
                        game.players[player_number].location_in_frames_perspective[count_frames - player_none - 1],
                        game.players[player_number].location_in_frames_perspective[count_frames]) < \
                        consts.MAX_PLAYER_JUMPING:
                    for empty_frame_index in range(0, player_none + 1):
                        current_x = game.players[player_number].location_in_frames_perspective[count_frames
                                                                                               - player_none - 1
                                                                                               + empty_frame_index][0] + \
                                    ((game.players[player_number].location_in_frames_perspective[count_frames][0] - \
                                      game.players[player_number].location_in_frames_perspective[count_frames
                                                                                                 - player_none - 1
                                                                                                 + empty_frame_index][0])
                                     / player_none
                                     * empty_frame_index)

                        current_y = game.players[player_number].location_in_frames_perspective[count_frames
                                                                                               - player_none - 1
                                                                                               + empty_frame_index][1] + \
                                    ((game.players[player_number].location_in_frames_perspective[count_frames][1] - \
                                      game.players[player_number].location_in_frames_perspective[count_frames
                                                                                                 - player_none - 1
                                                                                                 + empty_frame_index][1])
                                     / player_none
                                     * empty_frame_index)
                        game.players[player_number].location_in_frames_perspective[count_frames
                                                                                   - player_none + empty_frame_index] = \
                            (int(current_x), int(current_y))

                    player_none = 0
                else:
                    game.players[player_number].location_in_frames_perspective[count_frames] = None
                    player_none += 1
            else:
                valid_fill_frames = True
                player_none = 0
            count_frames += 1

    # BALL
    count_frames = 0
    ball_none = 0
    valid_fill_frames = False
    while count_frames < consts.MAX_FRAMES:
        if game.ball.location_in_frames_perspective[count_frames] is None:
            ball_none += 1
        elif game.ball.location_in_frames_perspective[count_frames] is \
                not None and ball_none != 0 and valid_fill_frames:
            if utils.euclidean_distance(
                    game.ball.location_in_frames_perspective[count_frames - ball_none - 1],
                    game.ball.location_in_frames_perspective[count_frames]) < consts.MAX_BALL_JUMPING:
                for empty_frame_index in range(0, ball_none + 1):
                    current_x = game.ball.location_in_frames_perspective[count_frames
                                                                         - ball_none - 1
                                                                         + empty_frame_index][0] + \
                                ((game.ball.location_in_frames_perspective[count_frames][0] - \
                                  game.ball.location_in_frames_perspective[count_frames
                                                                           - ball_none - 1
                                                                           + empty_frame_index][0])
                                 / ball_none
                                 * empty_frame_index)

                    current_y = game.ball.location_in_frames_perspective[count_frames
                                                                         - ball_none - 1
                                                                         + empty_frame_index][1] + \
                                ((game.ball.location_in_frames_perspective[count_frames][1] - \
                                  game.ball.location_in_frames_perspective[count_frames
                                                                           - ball_none - 1
                                                                           + empty_frame_index][1])
                                 / ball_none
                                 * empty_frame_index)
                    game.ball.location_in_frames_perspective[count_frames - ball_none + empty_frame_index] = \
                        (int(current_x), int(current_y))

                ball_none = 0
            else:
                game.ball.location_in_frames_perspective[count_frames] = None
                ball_none += 1
        else:
            valid_fill_frames = True
            ball_none = 0
        count_frames += 1


def remove_irrelevant_players(game):
    for player_number in game.players:
        count_player_appearance = 0
        for idx_frame in range(consts.MAX_FRAMES - 1):
            if game.players[player_number].location_in_frames_perspective[idx_frame] is not None:
                count_player_appearance += 1
        if count_player_appearance < consts.MAX_FRAMES/5:
            game.players[player_number].is_active = False
            print("Removed player {0} for being inactive".format(player_number))


def connect_players_to_one(game):
    for player_number in game.players:
        player_in_frame = True
        # Check if player is in the frame in frame 0
        if game.players[player_number].location_in_frames_perspective[0] is None:
            player_in_frame = False

        start_index = 0
        for idx_frame in range(0, consts.MAX_FRAMES):
            if game.players[player_number].location_in_frames_perspective[idx_frame] is None and player_in_frame:
                game.players[player_number].in_frame.append((start_index, idx_frame - 1))
                start_index = idx_frame
                player_in_frame = False
                continue

            if game.players[player_number].location_in_frames_perspective[idx_frame] is not None \
                    and not player_in_frame:
                game.players[player_number].out_frame.append((start_index, idx_frame - 1))
                start_index = idx_frame
                player_in_frame = True

        if player_in_frame:
            game.players[player_number].in_frame.append((start_index, idx_frame))
        else:
            game.players[player_number].out_frame.append((start_index, idx_frame))

    for player_number in game.players:
        potential_connections = []
        for other_player_number in game.players:
            if player_number != other_player_number and game.players[player_number].is_active \
                    and game.players[other_player_number].is_active:
                combined_arr = sorted(game.players[player_number].in_frame + game.players[other_player_number].in_frame,
                                      key=lambda x: x[1])
                combined = True
                for in_frame_idx in range(1, len(combined_arr)):
                    if combined_arr[in_frame_idx][0] < combined_arr[in_frame_idx-1][1]:
                        combined = False
                        break

                if combined:
                    potential_connections.append(other_player_number)
        game.players[player_number].potential_identities = potential_connections
        if len(potential_connections) > 0:
            print("After phase 1 - For player {0}, the potentials are: {1}".format(player_number, potential_connections))

    for player_number in game.players:
        potential_connections = []
        for other_player_number in game.players[player_number].potential_identities:
            for in_frame_player in game.players[player_number].in_frame:
                for in_frame_other_player in game.players[other_player_number].in_frame:
                    if in_frame_player[0] > in_frame_other_player[0]:
                        if utils.euclidean_distance(game.players[player_number].location_in_frames_perspective[in_frame_player[0]],
                                                    game.players[other_player_number].location_in_frames_perspective[in_frame_other_player[1]])\
                                                      < 50:
                            potential_connections.append(other_player_number)
                            connect_players(game, player_number, potential_connections)
                    else:
                        if utils.euclidean_distance(game.players[player_number].location_in_frames_perspective[in_frame_player[1]],
                                                    game.players[other_player_number].location_in_frames_perspective[in_frame_other_player[0]]) \
                                                      < 50:
                            potential_connections.append(other_player_number)
                            connect_players(game, player_number, potential_connections)
        if len(potential_connections) > 0:
            print("After phase 2 - For player {0}, the potentials are: {1}".format(player_number, potential_connections))


def connect_players(game, player_number, connections):
    for other_player_number in connections:
        if game.players[other_player_number].is_active and game.players[player_number].is_active:
            game.players[other_player_number].is_active = False
            for i in range(consts.MAX_FRAMES):
                if game.players[other_player_number].location_in_frames_perspective[i] is not None:
                    game.players[player_number].location_in_frames_perspective[i] = \
                        game.players[other_player_number].location_in_frames_perspective[i]


def connect_ball_to_players(game):
    for frame_idx in range(consts.MAX_FRAMES):
        closest_distance = 99999
        closest_player = None
        if game.ball.location_in_frames_perspective[frame_idx] is not None:
            for player_number in game.players:
                if game.players[player_number].location_in_frames_perspective[frame_idx] is not None:
                    if game.players[player_number].is_active:
                        ball_player_distance = utils.euclidean_distance(game.ball.location_in_frames_perspective[frame_idx], game.players[player_number].location_in_frames_perspective[frame_idx])
                        if ball_player_distance < consts.MAX_DISTANCE_BALL_PLAYER_CONNECTION and ball_player_distance < closest_distance:
                            closest_distance = ball_player_distance
                            closest_player = player_number
        game.ball.player_with_ball_in_frames_perspective[frame_idx] = closest_player


def set_video_frame_size(game, file_path):
    vid = cv2.VideoCapture(file_path)
    game.frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    game.frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(game.frame_height, game.frame_width)


if __name__ == '__main__':
    # Clone YoloV5 & Deep sort pytorch. Update & Install requirements.txt ###
    # os.system('git clone https://github.com/ultralytics/yolov5.git')
    # os.system('https://github.com/ZQPei/deep_sort_pytorch')
    os.system('pip freeze > requirements.txt')
    # os.system('pip install -r requirements.txt')

    # Path ###
    video_path = 'sources/TestVideos/vid3.mp4'
    txt_file_name = "outputs/textfiles/demofile5.txt"
    field_image = cv2.imread('sources/TestImages/maracana_homemade.png')

    # Track video and write to file ###
    # frames = track.start_tracking(video_path)
    # outputs.write_frames_to_file(frames, txt_file_name)

    # Convert file to frames ###
    frames = outputs.read_from_file_to_frame(txt_file_name)

    # Game cleaning and filtering ###
    maracana_game = separate_players_and_ball(frames)
    change_perspective(maracana_game)

    remove_irrelevant_players(maracana_game)
    fill_empty_frames(maracana_game)
    connect_players_to_one(maracana_game)
    fill_empty_frames(maracana_game)
    maracana_game.players = delete_out_of_field_players(maracana_game.players, field_image)
    connect_ball_to_players(maracana_game)
    set_video_frame_size(maracana_game, video_path)
    # fix_ball_zig_zags(maracana_game, 2)
    # fix_players_zig_zags(maracana_game, 2)

    # playerteam.identify_players_team(maracana_game, video_path)

    # Outputs ###
    # outputs.create_vid_only_ball(maracana_game.ball, field_img)
    # outputs.create_vid_one_player(maracana_game.players[21], field_img)
    # outputs.create_vid_all_player(maracana_game, field_image)
    # outputs.two_vids_to_one("outputs/videos/game.avi", "sources/TestVideos/vid3.mp4", "outputs/videos/final.mp4")

    # Statistics ###
    # statistics.stats_players_distance_covered(maracana_game.players)
    # statistics.get_avg_distance_in_frame_of_all_players(maracana_game)
    statistics.heat_map(maracana_game, field_image)
    # User Interface ###
    # userInterface.start_ui(maracana_game)
