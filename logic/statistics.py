import logic.CONSTS as CONSTS
from logic import helpful


def stats_players_distance_covered(players_arr):
    for player_number in players_arr:
        for idx_frame in range(CONSTS.MAX_FRAMES - 1):
            if players_arr[player_number].location_in_frames_perspective[idx_frame] is not None \
                    and players_arr[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                # if player_number == 21:
                #     print(euclidean_distance(players_arr[player_number].location_in_frames_perspective[idx_frame],
                #                        players_arr[player_number].location_in_frames_perspective[idx_frame + 1]))
                players_arr[player_number].distance_covered += \
                    helpful.euclidean_distance(players_arr[player_number].location_in_frames_perspective[idx_frame],
                                               players_arr[player_number].location_in_frames_perspective[idx_frame + 1])

    # print stats
    # for player_number in players_arr:
    #     print("player - ", players_arr[player_number].number, "  player_distance  - ",
    #           players_arr[player_number].distance_covered)


def get_avg_distance_in_frame_of_all_players(game):
    sum_distance = 0
    count_of_sums = 0
    for player_number in game.players:
        for idx_frame in range(CONSTS.MAX_FRAMES - 1):
            if game.players[player_number].location_in_frames_perspective[idx_frame] is not None \
                    and game.players[player_number].location_in_frames_perspective[idx_frame + 1] is not None:
                sum_distance += \
                    helpful.euclidean_distance(game.players[player_number].location_in_frames_perspective[idx_frame],
                                               game.players[player_number].location_in_frames_perspective[
                                                   idx_frame + 1])
                count_of_sums += 1

    # print("Avg distance of all players between each frame: " + str(sum_distance / count_of_sums))
    return sum_distance / count_of_sums
