import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def detect_humans(image):
    classes = None
    with open('../config/detect_and_track/coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(classes)
    Width = image.shape[1]
    Height = image.shape[0]


    # read pre-trained model and config file
    net = cv2.dnn.readNet('../config/detect_and_track/yolov3.cfg','../config/detect_and_track/yolov3.weights')
    # create input blob
    # set input blob for the network
    net.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False))

    # run inference through the network
    # and gather predictions from output layers

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    point_of_players = []

    # create bounding box
    count = 1
    for out in outs:
        if count == 1:
            print(out)
        for detection in out:
            if count == 1:
                print(detection)
                count += 1
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

    # check if is people detection
    for i in indices:
        i = i[0]
        box = boxes[i]
        print(class_id, classes[class_id], str(classes[class_id]))
        label = str(classes[class_id])

        point_of_players.append([round(box[0] + (box[2] / 2)), round(box[1] + (box[3]))])

        cv2.circle(image, (round(box[0] + (box[2] / 2)), round(box[1] + (box[3]))), 5, (0, 0, 0), 2)
        cv2.putText(image, label, (round(box[0]) - 10, round(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    plt.imshow(image)
    plt.show()
    return point_of_players


image = cv2.imread('../sources/TestImages/figure1.jpg')
points_middle = detect_humans(image)
# print(points_middle)