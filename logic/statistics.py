import logic.consts as consts
from logic import utils
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
import logic.consts as consts
import cv2


def stats_players_distance_covered(players_arr):
    for player_number in players_arr:
        for idx_frame in range(consts.MAX_FRAMES - 1):
            if players_arr[player_number].location_in_frames_perspective[idx_frame] is not None \
                    and players_arr[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                # if player_number == 21:
                #     print(euclidean_distance(players_arr[player_number].location_in_frames_perspective[idx_frame],
                #                        players_arr[player_number].location_in_frames_perspective[idx_frame + 1]))
                players_arr[player_number].distance_covered += \
                    utils.euclidean_distance(players_arr[player_number].location_in_frames_perspective[idx_frame],
                                             players_arr[player_number].location_in_frames_perspective[idx_frame + 1])

    # print stats
    # for player_number in players_arr:
    #     print("player - ", players_arr[player_number].number, "  player_distance  - ",
    #           players_arr[player_number].distance_covered)


def get_avg_distance_in_frame_of_all_players(game):
    sum_distance = 0
    count_of_sums = 0
    for player_number in game.players:
        for idx_frame in range(consts.MAX_FRAMES - 1):
            if game.players[player_number].location_in_frames_perspective[idx_frame] is not None \
                    and game.players[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                sum_distance += \
                    utils.euclidean_distance(game.players[player_number].location_in_frames_perspective[idx_frame],
                                             game.players[player_number].location_in_frames_perspective[
                                                 idx_frame + 1])
                count_of_sums += 1

    # print("Avg distance of all players between each frame: " + str(sum_distance / count_of_sums))
    return sum_distance / count_of_sums


def increase_player_heat_map_radius(game, player_number, frame_idx):
    x = np.arange(0, game.frame_width)
    y = np.arange(0, game.frame_height)

    cx = game.players[player_number].location_in_frames_perspective[frame_idx][0]
    cy = game.players[player_number].location_in_frames_perspective[frame_idx][1]
    r = consts.HEAT_MAP_RADIUS

    circle_mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    game.players[player_number].heat_map[circle_mask] += 1


def calculate_heat_map(game):
    print("Starting setting players heat map")
    for player_number in game.players:
        if game.players[player_number].is_active:
            game.players[player_number].heat_map = np.zeros((game.frame_height, game.frame_width))
            for frame_idx in range(consts.MAX_FRAMES):
                if game.players[player_number].location_in_frames_perspective[frame_idx] is not None:
                    increase_player_heat_map_radius(game, player_number, frame_idx)
    print("Finished setting players heat map")
    return game.players[11].heat_map


def heat_map(game, field_image):
    player_heat_map = calculate_heat_map(game)

    # Plt image ###
    # plt.imshow(player_heat_map, cmap='hot', interpolation='nearest', aspect='auto')
    # plt.axis('off')
    # plt.savefig('heatmap.png')
    # img = cv2.imread('heatmap.png')
    # blur = cv2.GaussianBlur(img, (5, 5), 0)
    # img_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_rgb)
    # plt.show()
