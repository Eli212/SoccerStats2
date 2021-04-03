# import track
# from old import tracking
import logic.outputs as outputs
from logic import helpful, playerteam, statistics, track

import logic.Classes as Classes
import logic.CONSTS as CONSTS
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


def change_perspective(game):
    points = ["B", "Y", "Y2", "C", "P", "Q", "V2", "W2", "VW2"]

    field_points = CONSTS.MARACANA_FIELD_POINTS2

    pts1 = np.float32([field_points[points[0]],
                       field_points[points[1]],
                       field_points[points[2]],
                       field_points[points[3]],
                       field_points[points[4]],
                       field_points[points[5]],
                       field_points[points[6]],
                       field_points[points[7]],
                       field_points[points[8]]])
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
            for idx_frame in range(CONSTS.MAX_FRAMES - 1):
                if game.players[player_number].location_in_frames_perspective[idx_frame] is not None \
                        and game.players[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                    if helpful.euclidean_distance(game.players[player_number].location_in_frames_perspective[idx_frame],
                                                  game.players[player_number].location_in_frames_perspective[
                                                      idx_frame + 1]) \
                            > CONSTS.MAX_PLAYER_ZIG_ZAGS:
                        # print(midpoint(game.players[player_number].location_in_frames_perspective[idx_frame],
                        #                game.players[player_number].location_in_frames_perspective[idx_frame + 1]))
                        game.players[player_number].location_in_frames_perspective[idx_frame + 1] = \
                            helpful.midpoint(game.players[player_number].location_in_frames_perspective[idx_frame],
                                             game.players[player_number].location_in_frames_perspective[idx_frame + 1])


def fix_ball_zig_zags(game, loop_times=1):
    for _ in range(loop_times):
        for idx_frame in range(CONSTS.MAX_FRAMES - 1):
            if game.ball.location_in_frames_perspective[idx_frame] is not None \
                    and game.ball.location_in_frames_perspective[idx_frame + 1] is not None:
                if helpful.euclidean_distance(game.ball.location_in_frames_perspective[idx_frame],
                                              game.ball.location_in_frames_perspective[idx_frame + 1]) \
                        > CONSTS.MAX_PLAYER_ZIG_ZAGS:
                    # print(midpoint(game.players[player_number].location_in_frames_perspective[idx_frame],
                    #                game.players[player_number].location_in_frames_perspective[idx_frame + 1]))
                    game.ball.location_in_frames_perspective[idx_frame + 1] = \
                        helpful.midpoint(game.ball.location_in_frames_perspective[idx_frame],
                                         game.ball.location_in_frames_perspective[idx_frame + 1])


def identify_players_team(game, vid_path):
    cap = cv2.VideoCapture(vid_path)
    dominant_color_arr = []
    for player_number in game.players:
        print(player_number)
        player_avg_color = []
        # do this for avg in case the random frame hits two players in box
        for frame_number in game.players[player_number].player_box:
            current_box = game.players[player_number].player_box[frame_number]
            current_box = [int(x) for x in current_box]
            # print(current_box)
            cap.set(1, frame_number)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            player_frame = frame[current_box[1]:current_box[1] + current_box[2],
                                 current_box[0]:int(current_box[0] + current_box[3] / 2), :]
            player_dominant_color = playerteam.get_dominant_color(player_frame)
            player_avg_color.append(player_dominant_color)
        # for i in range(3):
        # player_box_frames_keys = game.players[player_number].player_box.keys()
        # player_box_random_len = random.randint(0, len(player_box_frames_keys)-1)
        # player_random_box_frame = list(player_box_frames_keys)[player_box_random_len]
        # current_box = game.players[player_number].player_box[player_random_box_frame]
        # current_box = [int(x) for x in current_box]
        # cap.set(1, player_random_box_frame)
        # ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # player_frame = frame[current_box[1]:current_box[1]+current_box[2], current_box[0]:int(current_box[0]+current_box[3]/2), :]
        # player_dominant_color = playerteam.get_dominant_color(player_frame)
        # player_avg_color.append(player_dominant_color)
        dominant_color_arr.append(playerteam.get_avg_color(player_avg_color))
    teams_arr = playerteam.get_kmeans_teams_arr(dominant_color_arr)
    print(list(game.players.keys()))
    print(teams_arr)
    for idx, player_number in enumerate(game.players):
        game.players[player_number].team = teams_arr[idx]


def fill_empty_frames(game):
    # PLAYERS
    for player_number in game.players:
        count_frames = 0
        player_none = 0
        valid_fill_frames = False
        while count_frames < CONSTS.MAX_FRAMES:
            if game.players[player_number].location_in_frames_perspective[count_frames] is None:
                player_none += 1
            elif game.players[player_number].location_in_frames_perspective[count_frames] is not\
                    None and player_none != 0 and valid_fill_frames:
                if helpful.euclidean_distance(
                        game.players[player_number].location_in_frames_perspective[count_frames - player_none - 1],
                        game.players[player_number].location_in_frames_perspective[count_frames]) < \
                        CONSTS.MAX_PLAYER_JUMPING:
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
    while count_frames < CONSTS.MAX_FRAMES:
        if game.ball.location_in_frames_perspective[count_frames] is None:
            ball_none += 1
        elif game.ball.location_in_frames_perspective[count_frames] is \
                not None and ball_none != 0 and valid_fill_frames:
            if helpful.euclidean_distance(
                    game.ball.location_in_frames_perspective[count_frames - ball_none - 1],
                    game.ball.location_in_frames_perspective[count_frames]) < CONSTS.MAX_BALL_JUMPING:
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


if __name__ == '__main__':
    # max_frames = 178
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
    fill_empty_frames(maracana_game)
    fix_ball_zig_zags(maracana_game, 2)
    maracana_game.players = delete_out_of_field_players(maracana_game.players, field_image)
    fix_players_zig_zags(maracana_game, 2)
    # identify_players_team(maracana_game, video_path)

    # Outputs ###
    # outputs.create_vid_only_ball(maracana_game.ball, field_img)
    # outputs.create_vid_one_player(maracana_game.players[21], field_img)
    outputs.create_vid_all_player(maracana_game, field_image)

    # Statistics ###
    # statistics.stats_players_distance_covered(maracana_game.players)
    # statistics.get_avg_distance_in_frame_of_all_players(maracana_game)
