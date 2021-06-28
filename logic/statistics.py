import logic.consts as consts
from logic import utils
import logic.classes as classes
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
import logic.consts as consts
import logic.outputs as outputs
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
    for player_number in players_arr:
        print("player - ", players_arr[player_number].number, "  player_distance  - ",
              str(utils.pixels_to_meters(players_arr[player_number].distance_covered)) + " meters")


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

    print("Avg distance of all players between each frame: " + str(utils.pixels_to_meters(sum_distance / count_of_sums))
          + " meters")
    return utils.pixels_to_meters(sum_distance / count_of_sums)


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
    # return game.players[35].heat_map


def heat_map(game, field_image):
    calculate_heat_map(game)

    for player_number in game.players:
        if game.players[player_number].is_active and player_number == 15:
            # Plt image ###
            cmap = plt.get_cmap('hot')
            cmap.set_under('white', alpha=0)
            plt.imshow(game.players[player_number].heat_map, cmap=cmap, vmin=1)
            plt.axis('off')
            plt.savefig(f'outputs/heatmaps/{player_number}.png', transparent=True, bbox_inches='tight')
            img = cv2.imread(f'outputs/heatmaps/{player_number}.png')
            img = cv2.resize(img, (field_image.shape[1], field_image.shape[0]))
            img = cv2.GaussianBlur(img, (5, 5), 0)
            for i in range(len(img)):
                for j in range(len(img[0])):
                    if img[i][j][0] == 255 and img[i][j][1] == 255 and img[i][j][2] == 255:
                        img[i][j] = field_image[i][j]

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.putText(img, f'Player ID: {player_number}', (50, 50), consts.TEXT_FONT,
                              consts.TEXT_FONTSCALE, consts.TEXT_COLOR, consts.TEXT_THICKNESS, cv2.LINE_AA)

            plt.imshow(img)
            plt.savefig(f'outputs/heatmaps/{player_number}.png', transparent=True, bbox_inches='tight')


def calculate_heat_lines(game):
    for player_number in game.players:
        if game.players[player_number].is_active:
            max_value = 0
            for idx_frame in range(consts.MAX_FRAMES - 1):
                if game.players[player_number].location_in_frames_perspective[idx_frame] is not None \
                        and game.players[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                    game.players[player_number].heat_lines.append(
                        classes.HeatLine(game.players[player_number].location_in_frames_perspective[idx_frame],
                                         game.players[player_number].location_in_frames_perspective[idx_frame + 1]))
                    if game.players[player_number].heat_lines[-1].dis > max_value:
                        max_value = game.players[player_number].heat_lines[-1].dis
            game.players[player_number].compute_heat_lines_color(max_value)


def heat_line(game, field_image):
    calculate_heat_lines(game)
    outputs.create_player_heat_line(game.players[15], field_image)
