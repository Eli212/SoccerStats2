import cv2
import numpy as np

# import threading
# import time
# import logging
# import concurrent.futures

import CONSTS
import Ori


def start(vid_name, field_img, save_first_frame_only=True):
    new_vid_name = "new_vid.avi"
    cap = cv2.VideoCapture(vid_name)
    count = 0
    frame_jump = 5

    video = cv2.VideoWriter(new_vid_name, 0, 30, (field_img.shape[1], field_img.shape[0]))
    while cap.isOpened():
        ret, current_frame = cap.read()
        if ret:
            print(count)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            # plt.imshow(frame)
            # plt.show()

            # cv2.imwrite('VideoInImages/vid2/frame{:d}.jpg'.format(count), frame)
            if save_first_frame_only:
                break
            players_from_frame = Ori.detect_humans(current_frame)
            current_field_with_players_2d = testing_changing_perspective(players_from_frame, field_img, current_frame)
            # current_frame_resized = cv2.resize(current_frame, (field_img.shape[1], field_img.shape[0]))
            # concated_frame = np.concatenate((current_field_with_players_2d, current_frame_resized), axis=1)
            video.write(current_field_with_players_2d)
            # plt.imshow(current_frame)
            # plt.show()
            count += frame_jump
            cap.set(1, count)

        else:
            cap.release()
            break


def testing_changing_perspective(players_arr, field_img, current_frame):
    field_img_copy = field_img.copy()

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

    # M = cv2.getAffineTransform(pts2, pts1)
    h, status = cv2.findHomography(pts1, pts2)
    im_dst = cv2.warpPerspective(current_frame, h, (field_img.shape[1], field_img.shape[0]))
    #
    # initCamera = cv2.initCameraMatrix2D(pts1, pts2, (1090, 1080))
    # print(initCamera)
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)

    new_players_arr = []
    for player in players_arr:
        px = (h[0][0] * player[0] + h[0][1] * player[1] + h[0][2]) / (
            (h[2][0] * player[0] + h[2][1] * player[1] + h[2][2]))
        py = (h[1][0] * player[0] + h[1][1] * player[1] + h[1][2]) / (
            (h[2][0] * player[0] + h[2][1] * player[1] + h[2][2]))
        new_players_arr.append((int(px), int(py)))

    # dst = cv2.warpPerspective(frame0, matrix, (field_img.shape[1], field_img.shape[0]))
    # dst2 = cv2.warpAffine(frame0, M, (field_img.shape[1], field_img.shape[0]))

    # add player in 2d images
    for player in new_players_arr:
        field_img_copy = cv2.circle(field_img_copy, player, CONSTS.CIRCLE_RADIUS, CONSTS.CIRCLE_COLOR,
                                    CONSTS.CIRCLE_THICKNESS)

    return field_img_copy
    # plt.imshow(field_img_copy)
    # plt.show()

    # field_with_players = cv2.circle(field_img, p_new, CONSTS.CIRCLE_RADIUS, CONSTS.CIRCLE_COLOR,
    #                                 CONSTS.CIRCLE_THICKNESS)

    # plt.imshow(field_with_players)
    # plt.show()


if __name__ == '__main__':
    field_img_2d = cv2.imread('../sources/TestImages/maracana_homemade.png')
    field_img_2d = cv2.cvtColor(field_img_2d, cv2.COLOR_BGR2RGB)
    start("../sources/TestVideos/vid2.mp4", field_img_2d.copy(), False)
    # frame0 = cv2.imread('VideoInImages/vid2/frame0.jpg')
    # players_from_frame = Ori.detect_humans(frame0)
    # testing_changing_perspective(players_from_frame)
