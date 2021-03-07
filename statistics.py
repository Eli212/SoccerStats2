import logic.CONSTS as CONSTS
import helpful


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