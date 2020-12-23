# from utils.detect_and_track import utils
# from utils.detect_and_track.models import *
# from ..utils.detect_and_track.sort import *
import logic.Classes as Classes
import logic.CONSTS as CONSTS

import torch
from torchvision import transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image
import numpy
import cv2

from IPython.display import clear_output


# config_path = 'config/detect_and_track/yolov3.cfg'
# weights_path = 'config/detect_and_track/yolov3.weights'
# class_path = 'config/detect_and_track/coco.names'
# img_size = 416
# conf_thres = 0.3
# nms_thres = 0.3
#
# # Load model and weights
# model = Darknet(config_path, img_size=img_size)
# model.load_weights(weights_path)
# model.cuda()
# model.eval()
# classes = utils.load_classes(class_path)
# Tensor = torch.cuda.FloatTensor


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


def run_each_frame(video_path):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in numpy.linspace(0, 1, 20)]
    # initialize Sort object and video capture
    vid = cv2.VideoCapture(video_path)
    mot_tracker_2 = Sort()

    #while(True):
    frames_arr = []
    for ii in range(120):
        ret, current_frame = vid.read()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # current_frame[:, :, 2] -= 150
        # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        # current_frame = cv2.filter2D(current_frame, -1, kernel)

        # brightness = 20
        # contrast = 10
        # current_frame = np.int16(current_frame)
        # current_frame = current_frame * (contrast / 127 + 1) - contrast + brightness
        # current_frame = np.clip(current_frame, 0, 255)
        # current_frame = np.uint8(current_frame)

        pilimg = Image.fromarray(current_frame)
        detections = detect_image(pilimg)
        current_frame_copy = current_frame.copy()

        img = numpy.array(pilimg)

        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if detections is not None:
            tracked_objects = mot_tracker_2.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            players = []
            # print(tracked_objects[0])
            ball_point = None
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                # box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                # box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                # y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                # x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                # ball_point = (int(x1 + box_w / 2), int(y1 + box_h / 2))
                #
                # color = colors[int(obj_id) % len(colors)]
                # color = [i * 255 for i in color]
                # cls = classes[int(cls_pred)]
                #
                # cv2.rectangle(current_frame_copy, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                # cv2.rectangle(current_frame_copy, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1)
                # cv2.putText(current_frame_copy, cls + "-" + str(int(obj_id)), (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                if classes[int(cls_pred)] == "person":
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

                    player = Classes.Player(int(obj_id), x1, y1, box_h, box_w, int(x1 + box_w/2), int(y1 + box_h/2))
                    players.append(player)

                    color = colors[int(obj_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cls = classes[int(cls_pred)]

                    cv2.rectangle(current_frame_copy, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                    cv2.rectangle(current_frame_copy, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                    cv2.putText(current_frame_copy, cls + "-" + str(int(obj_id)), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

                if classes[int(cls_pred)] == "sports ball":
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    ball_point = (int(x1 + box_w / 2), int(y1 + box_h / 2))

                    color = colors[int(obj_id) % len(colors)]
                    color = [i * 255 for i in color]
                    cls = classes[int(cls_pred)]

                    cv2.rectangle(current_frame_copy, (x1, y1), (x1 + box_w, y1 + box_h), color, 4)
                    cv2.rectangle(current_frame_copy, (x1, y1 - 35), (x1 + len(cls) * 19 + 60, y1), color, -1)
                    cv2.putText(current_frame_copy, cls + "-" + str(int(obj_id)), (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            if ii == 50 or ii == 90:
                plt.imshow(current_frame_copy)
                plt.show()

        current_frame = Classes.Frame(ii, players)
        current_frame.ball = ball_point
        frames_arr.append(current_frame)

    return frames_arr


def testing_changing_perspective(the_frame, field_img):
    field_img_copy = field_img.copy()

    points = ["B", "Y", "Y2", "C", "P", "Q", "V2", "W2", "VW2"]

    pts1 = numpy.float32([CONSTS.MARACANA_FIELD_POINTS[points[0]],
                       CONSTS.MARACANA_FIELD_POINTS[points[1]],
                       CONSTS.MARACANA_FIELD_POINTS[points[2]],
                       CONSTS.MARACANA_FIELD_POINTS[points[3]],
                       CONSTS.MARACANA_FIELD_POINTS[points[4]],
                       CONSTS.MARACANA_FIELD_POINTS[points[5]],
                       CONSTS.MARACANA_FIELD_POINTS[points[6]],
                       CONSTS.MARACANA_FIELD_POINTS[points[7]],
                       CONSTS.MARACANA_FIELD_POINTS[points[8]]])
    pts2 = numpy.float32([CONSTS.MARACANA_HOMEMADE_FIELD_POINTS[points[0]],
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
    # im_dst = cv2.warpPerspective(current_frame, h, (field_img.shape[1], field_img.shape[0]))
    #
    # initCamera = cv2.initCameraMatrix2D(pts1, pts2, (1090, 1080))
    # print(initCamera)
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)

    for player in the_frame.players:
        px = (h[0][0] * player.point_x + h[0][1] * player.point_y + h[0][2]) / (
            (h[2][0] * player.point_x + h[2][1] * player.point_y + h[2][2]))
        py = (h[1][0] * player.point_x + h[1][1] * player.point_y + h[1][2]) / (
            (h[2][0] * player.point_x + h[2][1] * player.point_y + h[2][2]))
        player_point = (int(px), int(py))
        field_img_copy = cv2.circle(field_img_copy, player_point, CONSTS.CIRCLE_RADIUS, CONSTS.CIRCLE_COLOR,
                                    CONSTS.CIRCLE_THICKNESS)
        field_img_copy = cv2.putText(field_img_copy, str(player.number), player_point,
                                     CONSTS.TEXT_FONT, CONSTS.TEXT_FONTSCALE, CONSTS.TEXT_COLOR, CONSTS.TEXT_THICKNESS)

    if the_frame.ball is not None:
        px = (h[0][0] * the_frame.ball[0] + h[0][1] * the_frame.ball[1] + h[0][2]) / (
            (h[2][0] * the_frame.ball[0] + h[2][1] * the_frame.ball[1] + h[2][2]))
        py = (h[1][0] * the_frame.ball[0] + h[1][1] * the_frame.ball[1] + h[1][2]) / (
            (h[2][0] * the_frame.ball[0] + h[2][1] * the_frame.ball[1] + h[2][2]))
        ball_point = (int(px), int(py))
        field_img_copy = cv2.circle(field_img_copy, ball_point, CONSTS.CIRCLE_RADIUS, CONSTS.CIRCLE_COLOR2,
                                    CONSTS.CIRCLE_THICKNESS)
    # dst = cv2.warpPerspective(frame0, matrix, (field_img.shape[1], field_img.shape[0]))
    # dst2 = cv2.warpAffine(frame0, M, (field_img.shape[1], field_img.shape[0]))

    # add player in 2d images
    # for player in new_players_arr:
    #     field_img_copy = cv2.circle(field_img_copy, player, CONSTS.CIRCLE_RADIUS, CONSTS.CIRCLE_COLOR,
    #                                 CONSTS.CIRCLE_THICKNESS)


    return field_img_copy


def start_vid(vid_name, field_img, frames_arr):
    new_vid_name = "new_vid.avi"
    cap = cv2.VideoCapture(vid_name)
    count = 0
    frame_jump = 1
    max_frames = 178
    video = cv2.VideoWriter(new_vid_name, 0, 30, (field_img.shape[1], field_img.shape[0]))
    while cap.isOpened() and count < max_frames:
        ret, current_frame = cap.read()
        if ret:
            # print(count)
            # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            # plt.imshow(frame)
            # plt.show()

            # cv2.imwrite('VideoInImages/vid2/frame{:d}.jpg'.format(count), frame)
            if count % 30 == 0:
                print(count)
            current_field_with_players_2d = testing_changing_perspective(frames_arr[count], field_img)
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
    # fig = plt.figure(figsize=(12, 8))
    # plt.title("Video Stream")
    # plt.imshow(frame)
    # plt.show()
    # clear_output(wait=True)


# vid_path = 'sources/TestVideos/vid4.mp4'
# frames = run_each_frame(vid_path)
# field_imgg = cv2.imread('sources/TestImages/maracana_homemade.png')
# start_vid(vid_path, field_imgg, frames)
