import logic.consts as consts
import logic.classes as classes
from moviepy.editor import VideoFileClip, concatenate_videoclips

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
            players.append(classes.Player(int(player_info[0]), float(player_info[1]), float(player_info[2]),
                                          float(player_info[3]), float(player_info[4]), float(player_info[5]),
                                          float(player_info[6])))
        current_frame = classes.Frame(idx_frame, players)
        current_frame.ball = frame_ball
        frames_arr.append(current_frame)

    f.close()
    print("Finished reading from file: " + file_name)
    return frames_arr


def create_vid_only_ball(ball, field_img):
    new_vid_name = "{0}.avi".format("ball")
    cap = cv2.VideoCapture(0)
    count = 0
    frame_jump = 1
    video = cv2.VideoWriter(new_vid_name, 0, 30, (field_img.shape[1], field_img.shape[0]))
    while count < consts.MAX_FRAMES:
        field_img_copy = field_img.copy()
        # if count % 30 == 0:
        #     print(count)
        if ball.location_in_frames_perspective[count] is not None:
            field_img_copy = cv2.circle(field_img_copy, ball.location_in_frames_perspective[count],
                                        consts.CIRCLE_RADIUS, consts.CIRCLE_COLOR2, consts.CIRCLE_THICKNESS)
            field_img_copy = cv2.putText(field_img_copy, str("B"),
                                         ball.location_in_frames_perspective[count],
                                         consts.TEXT_FONT, consts.TEXT_FONTSCALE, consts.TEXT_COLOR,
                                         consts.TEXT_THICKNESS)
            video.write(field_img_copy)
        else:
            video.write(field_img.copy())
        count += frame_jump
        cap.set(1, count)


def create_vid_one_player(player, field_img):
    new_vid_name = "{0}.avi".format(player.number)
    cap = cv2.VideoCapture(0)
    count = 0
    frame_jump = 1
    video = cv2.VideoWriter(new_vid_name, 0, 30, (field_img.shape[1], field_img.shape[0]))
    while count < consts.MAX_FRAMES:
        field_img_copy = field_img.copy()
        # if count % 30 == 0:
        #     print(count)
        if player.location_in_frames_perspective[count] is not None:
            field_img_copy = cv2.circle(field_img_copy, player.location_in_frames_perspective[count],
                                        consts.CIRCLE_RADIUS, consts.CIRCLE_COLOR2, consts.CIRCLE_THICKNESS)
            field_img_copy = cv2.putText(field_img_copy, str(player.number),
                                         player.location_in_frames_perspective[count],
                                         consts.TEXT_FONT, consts.TEXT_FONTSCALE, consts.TEXT_COLOR,
                                         consts.TEXT_THICKNESS)
            video.write(field_img_copy)
        else:
            video.write(field_img.copy())
        count += frame_jump
        cap.set(1, count)


def create_vid_all_player(game, field_img):
    print("Starting to write video")
    new_vid_name = "outputs/videos/game.avi"
    cap = cv2.VideoCapture(0)
    count = 0
    frame_jump = 1
    video = cv2.VideoWriter(new_vid_name, 0, 30, (field_img.shape[1], field_img.shape[0]))
    while count < consts.MAX_FRAMES:
        field_img_copy = field_img.copy()
        # if count % 30 == 0:
        #     print(count)
        for player_number in game.players:
            if not game.players[player_number].is_active:
                continue
            if game.players[player_number].location_in_frames_perspective[count] is not None:
                if game.players[player_number].team == 0:
                    field_img_copy = cv2.circle(field_img_copy,
                                                game.players[player_number].location_in_frames_perspective[count],
                                                consts.CIRCLE_RADIUS, consts.TEAM_A, consts.CIRCLE_THICKNESS)
                else:
                    field_img_copy = cv2.circle(field_img_copy,
                                                game.players[player_number].location_in_frames_perspective[count],
                                                consts.CIRCLE_RADIUS, consts.TEAM_B, consts.CIRCLE_THICKNESS)
                field_img_copy = cv2.putText(field_img_copy, str(player_number),
                                             game.players[player_number].location_in_frames_perspective[count],
                                             consts.TEXT_FONT, consts.TEXT_FONTSCALE, consts.TEXT_COLOR,
                                             consts.TEXT_THICKNESS)

        if game.ball.location_in_frames_perspective[count] is not None:
            field_img_copy = cv2.circle(field_img_copy, game.ball.location_in_frames_perspective[count],
                                        consts.CIRCLE_RADIUS, consts.CIRCLE_COLOR2, consts.CIRCLE_THICKNESS)
            field_img_copy = cv2.putText(field_img_copy, str("B"),
                                         game.ball.location_in_frames_perspective[count],
                                         consts.TEXT_FONT, consts.TEXT_FONTSCALE, consts.TEXT_COLOR,
                                         consts.TEXT_THICKNESS)

        video.write(field_img_copy)
        count += frame_jump
        cap.set(1, count)
    print("Finished to write video")


def two_vids_to_one(vid1_location, vid2_location, final_vid_location):
    print("Starting to concatenate 2 videos to one")
    clip_1 = VideoFileClip(vid1_location)
    clip_2 = VideoFileClip(vid2_location)
    final_clip = concatenate_videoclips([clip_1, clip_2], method="compose")
    final_clip.write_videofile(final_vid_location)
    print("Finished to concatenate 2 videos to one")
