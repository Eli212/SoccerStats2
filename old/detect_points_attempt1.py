import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance as scipy_distance
from collections import Counter
import random
import locale
import os
import timeit

# import threading
# import time
# import logging
# import concurrent.futures

import CONSTS
import Classes


def close_colors(color1, color2, color_threshold):
    r = color1[0] - color2[0]
    if abs(r) > color_threshold:
        return False
    g = color1[1] - color2[1]
    if abs(g) > color_threshold:
        return False
    b = color1[2] - color2[2]
    if abs(b) > color_threshold:
        return False
    return True


def check_inter_point_area_dominant_color(the_og_img, the_point, og_img_dom_color, area_threshold, color_threshold):
    y_min = max(0, the_point[0][0] - area_threshold)
    y_max = min(the_point[0][0] + area_threshold, the_og_img.shape[1])
    x_min = max(0, the_point[0][1] - area_threshold)
    x_max = min(the_point[0][1] + area_threshold, the_og_img.shape[0])

    intersection_point_area = the_og_img[x_min:x_max, y_min:y_max, :]
    intersection_point_area_dominant_color = get_dominant_color(intersection_point_area)
    return close_colors(intersection_point_area_dominant_color, og_img_dom_color, color_threshold)


def get_dominant_color(the_image):
    start_time = timeit.default_timer()

    z = the_image.reshape((-1, 3))
    # convert to np.float32
    z = np.float32(z)
    # define criteria, number of clusters(K) and apply k-means()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 1
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into u-int8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(the_image.shape)

    stop_time = timeit.default_timer()
    print('Time - Get Dominant Color: ', stop_time - start_time)
    return res2[0][0]


def delete_crowd_in_field(the_img):
    start_time = timeit.default_timer()

    b, g, r = cv2.split(the_img)
    final = the_img.copy()

    color_threshold = 0.4
    for i in range(len(b)):
        for j in range(len(b[i])):
            # find greenness
            if g[i][j] > 0:
                a = float(g[i][j]) / (float(r[i][j]) + float(b[i][j]) + float(g[i][j]))
            else:
                a = 0
            if a > color_threshold:
                final[i][j] = 255
            else:
                final[i][j] = 0

    # open op
    opening = cv2.morphologyEx(final, cv2.MORPH_OPEN, CONSTS.KERNEL_LIGHT)

    # clos op
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, CONSTS.KERNEL_LIGHT, iterations=5)

    final_image = the_img.copy()

    # merge with original image
    for i in range(len(closing)):
        for j in range(len(closing[i])):
            if not (closing[i][j][0] == 0 and closing[i][j][1] == 0 and closing[i][j][2] == 0):
                final_image[i][j] = the_img[i][j]
            else:
                final_image[i][j] = closing[i][j]

    stop_time = timeit.default_timer()
    print('Time - Delete crowd in the field: ', stop_time - start_time)

    return final_image


def get_edges_image(the_img, dominant_color_in_img):
    threshold_img_dominant_color_tuple = 40
    dominant_color_in_img_tuple_low = tuple([int(dominant_color_in_img[0]) - threshold_img_dominant_color_tuple,
                                             int(dominant_color_in_img[1]) - threshold_img_dominant_color_tuple,
                                             int(dominant_color_in_img[2]) - threshold_img_dominant_color_tuple])

    dominant_color_in_img_tuple_high = tuple([int(dominant_color_in_img[0]) + threshold_img_dominant_color_tuple,
                                              int(dominant_color_in_img[1]) + threshold_img_dominant_color_tuple,
                                              int(dominant_color_in_img[2]) + threshold_img_dominant_color_tuple])

    # print(dominant_color_in_img)

    start_time = timeit.default_timer()

    mask = cv2.inRange(the_img, dominant_color_in_img_tuple_low, dominant_color_in_img_tuple_high)
    mask = cv2.bitwise_not(mask)
    # plt.imshow(mask)
    # plt.show()

    # tophat = cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, CONSTS.KERNEL_STRONG)

    # plt.imshow(tophat)
    # plt.show()

    edges = cv2.Canny(mask, 50, 150, 3)

    stop_time = timeit.default_timer()
    print('Time - Get Edges Image: ', stop_time - start_time)

    return edges


def get_lines(the_img, og_img=None):
    start_time = timeit.default_timer()

    # Standard Hough Line Transform
    lines = cv2.HoughLines(the_img, 1, np.pi / 180, 120)
    # lines = cv2.HoughLines(the_img, 1, 0.001, 150, None, 0, 0)
    # lines1 = cv2.HoughLines(the_img, 1, 0.001, 150, None, 1.7, 1.9)
    # lines2 = cv2.HoughLines(the_img, 1, 0.001, 150, None, 1.3, 1.5)
    # lines = np.concatenate((lines1, lines2))

    # Draw the lines
    main_lines = []
    degrees_distance = 0.1
    MAX_DISTANCE = 10

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
            # pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
            pt1 = (int(x0 + the_img.shape[0] * (-b)), int(y0 + the_img.shape[1] * a))
            pt2 = (int(x0 - the_img.shape[0] * (-b)), int(y0 - the_img.shape[1] * a))

            myradians_current = math.atan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
            mydegrees_current = math.degrees(myradians_current)

            add_line = True
            for main_line in main_lines:
                # distance = math.sqrt(
                # ((center_current[0]-main_line.center[0])**2)+((center_current[1]-main_line.center[1])**2))
                # if abs(mydegrees_current - main_line.degrees) <= degrees_distance:

                # Eliminate 'duplicated' lines
                if abs(lines[i][0][0] - main_line.the_line[0][0]) < MAX_DISTANCE:
                    add_line = False
                    break
            if add_line:
                cv2.line(og_img, pt1, pt2, (0, 0, 255), 1)
                main_lines.append(Classes.Line(lines[i], mydegrees_current))

    # plt.imshow(og_img)
    # plt.show()

    stop_time = timeit.default_timer()
    print('Time - Get lines: ', stop_time - start_time)
    return main_lines


# Find intersections between 2 lines
def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    if line1.degrees - line2.degrees > 10:
        rho1, theta1 = line1.the_line[0]
        rho2, theta2 = line2.the_line[0]

        # print(rho1, rho2, theta1, theta2)
        a = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        try:
            x0, y0 = np.linalg.solve(a, b)
        except:
            return -1
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        if x0 < 0 or y0 < 0:
            return -1
        return [[x0, y0], theta1 * theta2]
    return -1


def get_intersection_points(lines, the_img):
    start_time = timeit.default_timer()

    THETAS_DISTANCE = 10
    intersections = []
    count_circles = 0
    sum_thetas = 0
    for i in range(len(lines)):
        for j in range(len(lines)):
            if i != j:
                intersection_point = intersection(lines[i], lines[j])
                if intersection_point != -1:
                    if intersection_point[0][0] <= the_img.shape[1] and intersection_point[0][1] <= the_img.shape[0]:
                        # if check_inter_point_area_dominant_color(
                        # img, intersection_point, og_img_dominant_color, 300, 40):

                        intersections.append(
                            Classes.IntersectionPoint(intersection_point, intersection_point[1], [lines[i], lines[j]]))
                        inter_point = (intersection_point[0][0], intersection_point[0][1])
                        sum_thetas += intersection_point[1]
                        # the_img = cv2.circle(the_img, inter_point, radius, circle_color, circle_thickness)
                        # the_img = cv2.putText(the_img, str(count_circles), inter_point, font, fontScale, text_color,
                        # text_thickness, cv2.LINE_AA, False)
                        count_circles += 1

    avg_thetas = sum_thetas / count_circles

    # Split intersection points into close groups
    MAX_DISTANCE_BETWEEN_POINTS = 10
    intersections_groups = []
    for i in range(len(intersections)):
        if not intersections[i].in_group:
            current_point_group = [intersections[i]]
            for j in range(len(intersections)):
                if i != j:
                    # distance = math.sqrt((abs(intersections[i].point[0][0] - intersections[j].point[0][0]) ** 2) + (
                    #         abs(intersections[i].point[0][1] - intersections[i].point[0][1]) ** 2))
                    distance = scipy_distance.euclidean(tuple(intersections[i].point[0]),
                                                        tuple(intersections[j].point[0]))
                    if distance < MAX_DISTANCE_BETWEEN_POINTS:
                        current_point_group.append(intersections[j])
                        intersections[j].in_group = True
            intersections_groups.append(current_point_group)

    intersection_points = []
    # Show only 1 intersection point from close group
    count_circles = 0
    for group in intersections_groups:
        if avg_thetas - THETAS_DISTANCE < group[0].thetas < avg_thetas + THETAS_DISTANCE:
            # the_img = cv2.circle(the_img, (group[0].point[0][0], group[0].point[0][1]), CONSTS.CIRCLE_RADIUS,
            #                      CONSTS.CIRCLE_COLOR, CONSTS.CIRCLE_THICKNESS)
            # the_img = cv2.putText(the_img, str(count_circles), (group[0].point[0][0], group[0].point[0][1]),
            #                       CONSTS.TEXT_FONT, CONSTS.TEXT_FONTSCALE, CONSTS.TEXT_COLOR,
            #                       CONSTS.TEXT_THICKNESS, cv2.LINE_AA, False)
            count_circles += 1
            # print(group[0].lines[0].the_line, group[0].lines[1].the_line)
            intersection_points.append(group[0])

    # print(intersections_groups[3][0].point)
    # print(intersections_groups[3])
    # cv2_imshow(the_img)

    # final_intersection_points = []
    for i in range(len(intersection_points)):
        if intersection_points[i].separated:
            for j in range(len(intersection_points)):
                if i != j and intersection_points[j].separated:
                    distance_between_points = scipy_distance.euclidean(tuple(intersection_points[i].point[0]),
                                                                       tuple(intersection_points[j].point[0]))
                    if distance_between_points < 50:
                        intersection_points[i].separated = False
                        break
            # if add_point:
            #     final_intersection_points.append(intersection_points[i])

            # the_img = cv2.circle(the_img, (intersection_points[i].point[0][0], intersection_points[i].point[0][1]),
            #                      CONSTS.CIRCLE_RADIUS,
            #                      CONSTS.CIRCLE_COLOR, CONSTS.CIRCLE_THICKNESS)
            # the_img = cv2.putText(the_img, str(i), (intersection_points[i].point[0][0],
            #                                         intersection_points[i].point[0][1]), CONSTS.TEXT_FONT,
            #                       CONSTS.TEXT_FONTSCALE, CONSTS.TEXT_COLOR,
            #                       CONSTS.TEXT_THICKNESS, cv2.LINE_AA, False)

    # for i in range(len(intersection_points)):
    #     if intersection_points[i].separated:
    #         the_img = cv2.circle(the_img, (intersection_points[i].point[0][0], intersection_points[i].point[0][1]),
    #                              CONSTS.CIRCLE_RADIUS,
    #                              CONSTS.CIRCLE_COLOR, CONSTS.CIRCLE_THICKNESS)
    #         the_img = cv2.putText(the_img, str(i),
    #                               (intersection_points[i].point[0][0], intersection_points[i].point[0][1]),
    #                               CONSTS.TEXT_FONT, CONSTS.TEXT_FONTSCALE, CONSTS.TEXT_COLOR,
    #                               CONSTS.TEXT_THICKNESS, cv2.LINE_AA, False)
    # plt.imshow(the_img)
    # plt.show()

    stop_time = timeit.default_timer()
    print('Time - Get intersection points: ', stop_time - start_time)

    return intersection_points


def split_video_to_frames(vid_name):
    cap = cv2.VideoCapture(vid_name)
    count = 0
    frame_jump = 5
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # slice the border, which is colored in black.
            # TODO - CHANGE THE POSITION OF THIS CODE LINE
            frame = frame[5: -5, 5: -5, :]
            # plt.imshow(frame)
            # plt.show()

            # Threading attempt
            # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            #     executor.map(start(frame, count), range(3))
            # x = threading.Thread(target=start, args=(frame,count), daemon=True)
            # x.start()
            # x.join()

            start(frame)
            # cv2.imwrite('VideoInImages/frame{:d}.jpg'.format(count), frame)
            count += frame_jump  # i.e. at 30 fps, this advances one second
            cap.set(1, count)

        else:
            cap.release()
            break

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()


def get_most_common(arr):
    lst = Counter(arr).most_common()
    highest_count = max([i[1] for i in lst])
    values = [i[0] for i in lst if i[1] == highest_count]
    random.shuffle(values)
    return values[0]


def find_field_point(the_img):
    start_time = timeit.default_timer()

    resize_div_by = 1.5

    field_points_arr = []
    locale.setlocale(locale.LC_ALL, '')

    the_img = cv2.resize(the_img, (int(the_img.shape[1] / resize_div_by), int(the_img.shape[0] / resize_div_by)))
    the_img2 = the_img.copy()

    field_points_directory = "FieldPoints"
    for point_folder in os.listdir(field_points_directory):
        new_field_point = Classes.FieldPoint(point_folder)
        for point_image in os.listdir(field_points_directory + "/" + point_folder):
            current_point = os.path.join(field_points_directory, point_folder, point_image)

            template = cv2.imread(current_point)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
            template = cv2.resize(template,
                                  (int(template.shape[1] / resize_div_by), int(template.shape[0] / resize_div_by)))

            # All the 6 methods for comparison in a list
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
            # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            #            'cv2.TM_CCORR_NORMED']

            # methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

            methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED']

            locs = []
            for meth in methods:
                the_img = the_img2.copy()
                method = eval(meth)

                # Apply template Matching
                res = cv2.matchTemplate(the_img, template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                locs.append(min_loc)
                locs.append(max_loc)

            get_most_common_locs = get_most_common(locs)
            resize_back = (get_most_common_locs[0] * resize_div_by, get_most_common_locs[1] * resize_div_by)
            new_field_point.add_point(Classes.FieldPointImageInfo(template.shape[0],
                                                                  template.shape[1],
                                                                  resize_back))
        field_points_arr.append(new_field_point)

    stop_time = timeit.default_timer()
    print('Time - Find field point: ', stop_time - start_time)
    return field_points_arr


def transfer_intersection_point_to_field_point(intersection_points, field_points):
    start_time = timeit.default_timer()

    for intersection_point in intersection_points:
        for field_point in field_points:
            current_score = 0
            for field_point_arr in field_point.points:
                current_score += field_point_arr.point_in_detected_field_point(intersection_point.point[0])
            if current_score > intersection_point.best_match_score:
                intersection_point.best_match_score = current_score
                intersection_point.best_match_name = field_point.point_name

    stop_time = timeit.default_timer()
    print('Time - Transfer intersection point to field point: ', stop_time - start_time)


def get_detected_field_points(intersection_points_arr, the_img=None):
    detected_field_points = []
    # field_points_names is for not showing duplicates
    selected_field_points_names = []
    for inter_point in intersection_points_arr:
        if inter_point.best_match_score >= 0 and inter_point.best_match_name not in selected_field_points_names:
            detected_field_points.append(inter_point)
            selected_field_points_names.append(inter_point.best_match_name)
            the_img = cv2.circle(the_img, (inter_point.point[0][0], inter_point.point[0][1]),
                                 CONSTS.CIRCLE_RADIUS,
                                 CONSTS.CIRCLE_COLOR, CONSTS.CIRCLE_THICKNESS)
            the_img = cv2.putText(the_img, inter_point.best_match_name,
                                  (inter_point.point[0][0], inter_point.point[0][1]),
                                  CONSTS.TEXT_FONT, CONSTS.TEXT_FONTSCALE, CONSTS.TEXT_COLOR,
                                  CONSTS.TEXT_THICKNESS, cv2.LINE_AA, False)
            # print(inter_point.best_match_name + " " + str(inter_point.best_match_score))
    plt.imshow(the_img)
    plt.show()

    return detected_field_points


def change_point_perspective(field_points_arr_3d, players_points_arr_3d, field_img=None, the_img=None):
    p = (1166, 281)
    # field_img = cv2.resize(field_img, (the_img.shape[1], the_img.shape[0]))
    # cv2.imwrite('field_soccer_resized.png', field_img)
    if len(field_points_arr_3d) >= 4:
        pts1 = []
        pts2 = []
        for i in range(4):
            print(tuple(field_points_arr_3d[i].point[0]))
            print(field_points_arr_3d[i].best_match_name)
            pts1.append(tuple(field_points_arr_3d[i].point[0]))
            pts2.append(CONSTS.FIELD_POINTS[field_points_arr_3d[i].best_match_name])
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
            (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
        py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
            (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))

        p_new = (int(px), int(py))
        print(p_new)
        dst = cv2.warpPerspective(the_img, matrix, (field_img.shape[1], field_img.shape[0]))
        field_with_players = cv2.circle(field_img, p_new, CONSTS.CIRCLE_RADIUS, CONSTS.CIRCLE_COLOR,
                                        CONSTS.CIRCLE_THICKNESS)
        # plt.imshow(field_with_players)
        # plt.show()

        plt.imshow(dst)
        plt.show()
    else:
        return -1


def testing_changing_perspective():
    p = (1630, 376)

    field_img = cv2.imread('../sources/TestImages/field_soccer_resized.png')
    field_img = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)

    frame0 = cv2.imread('../sources/VideoInImages/vid1/frame0.jpg')
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
# O2 B U
    pts1 = np.float32([[1346, 139, 0], [1611, 281, 0], [1235, 314, 0], [1060, 236, 0]])
    pts2 = np.float32([[1763, 70, 0], [1658, 380, 0], [1445, 413, 0], [1445, 200, 0]])

    # M = cv2.getAffineTransform(pts2, pts1)
    h, status = cv2.findHomography(pts1, pts2)
    im_dst = cv2.warpPerspective(frame0, h, (field_img.shape[1], field_img.shape[0]))

    initCamera = cv2.initCameraMatrix2D(pts1, pts2, (1090, 1080))
    print(initCamera)
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
    #     (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))
    # py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
    #     (matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2]))

    # p_new = (int(px), int(py))

    # dst = cv2.warpPerspective(frame0, matrix, (field_img.shape[1], field_img.shape[0]))
    # dst2 = cv2.warpAffine(frame0, M, (field_img.shape[1], field_img.shape[0]))

    plt.imshow(im_dst)
    plt.show()

    # field_with_players = cv2.circle(field_img, p_new, CONSTS.CIRCLE_RADIUS, CONSTS.CIRCLE_COLOR,
    #                                 CONSTS.CIRCLE_THICKNESS)

    # plt.imshow(field_with_players)
    # plt.show()


def test_match_template(the_img):
    start_time = timeit.default_timer()

    the_img = cv2.cvtColor(the_img, cv2.COLOR_BGR2RGB)
    the_img2 = the_img.copy()

    template = cv2.imread('../sources/FieldPoints/B/pointB1.png')
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    # All the 6 methods for comparison in a list
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED']
    methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    locs = []
    for meth in methods:
        the_img = the_img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(the_img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        locs.append(min_loc)
        locs.append(max_loc)

    stop_time = timeit.default_timer()
    print('Time - Testing matching template: ', stop_time - start_time)
    print('Locs', end=" ")
    print(get_most_common(locs))


def test_match_template_small_img(the_img):
    resize_div_by = 10
    start_time = timeit.default_timer()

    the_img = cv2.resize(the_img, (int(the_img.shape[1] / resize_div_by), int(the_img.shape[0] / resize_div_by)))
    the_img = cv2.cvtColor(the_img, cv2.COLOR_BGR2RGB)
    the_img2 = the_img.copy()

    template = cv2.imread('../sources/FieldPoints/B/pointB3.png')
    template = cv2.resize(template, (int(template.shape[1] / resize_div_by), int(template.shape[0] / resize_div_by)))
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    # All the 6 methods for comparison in a list
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED']
    methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    locs = []
    for meth in methods:
        the_img = the_img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(the_img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        locs.append(min_loc)
        locs.append(max_loc)

    get_most_common_locs = get_most_common(locs)
    resize_back = (get_most_common_locs[0] * resize_div_by, get_most_common_locs[1] * resize_div_by)
    stop_time = timeit.default_timer()
    print('Time - Testing matching template: ', stop_time - start_time)
    print('Locs:', end=" ")
    print(get_most_common_locs)
    print('Back to resized Locs:', end=" ")
    print(resize_back)


def testing_find_corners(the_img, detected_field_points=None):
    # current_lines = detected_field_points[0].lines[0].the_line
    current_lines = [[209, 1.5184364]]
    rho = current_lines[0][0]
    theta = current_lines[0][1]

    a = math.cos(theta)
    b = math.sin(theta)

    # x0 = a * rho
    # y0 = b * rho

    x0 = 1372 * 2 * the_img.shape[1]
    y0 = 144 * 2 * the_img.shape[0]

    dup_size_img = np.zeros_like(the_img)
    combine_2_images = np.concatenate((dup_size_img, the_img), axis=1)
    combine_2_images = np.concatenate((dup_size_img, combine_2_images), axis=1)
    print(the_img.shape)
    pt1 = (int(x0 + the_img.shape[1] * (-b)) + the_img.shape[1]*2, int(y0 + the_img.shape[0] * a))
    pt2 = (int(x0 - the_img.shape[1] * (-b)) + the_img.shape[1]*2, int(y0 - the_img.shape[0] * a))

    print(pt1, pt2)
    cv2.line(combine_2_images, pt1, pt2, (0, 0, 255), 1)
    plt.imshow(combine_2_images)
    plt.show()


def start(the_img):
    # field_img = cv2.imread('TestImages/soccerfield2d.jpg')
    field_img = cv2.imread('../sources/TestImages/field_soccer_resized.png')
    og_img_dominant_color = get_dominant_color(the_img.copy())

    # deleted_crowd_image = delete_crowd_in_field(the_img.copy())
    # tophat_image = get_tophat_image(deleted_crowd_image, og_img_dominant_color)

    edges_image = get_edges_image(the_img.copy(), og_img_dominant_color)

    the_lines = get_lines(edges_image, the_img.copy())
    intersection_points_arr = get_intersection_points(the_lines, the_img.copy())
    field_points = find_field_point(the_img.copy())
    transfer_intersection_point_to_field_point(intersection_points_arr, field_points)
    # intersection_points_arr = delete_duplicate_points(intersection_points_arr)
    detected_field_points = get_detected_field_points(intersection_points_arr, the_img.copy())
    # testing_find_corners(detected_field_points, the_img.copy())
    change_point_perspective(detected_field_points, [], field_img.copy(), the_img.copy())


if __name__ == '__main__':
    # Main start is this:
    # split_video_to_frames('TestVideos/vid1.mp4')



    testing_changing_perspective()
    # frame0 = cv2.imread('VideoInImages/frame0.jpg')
    # frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    # testing_find_corners(frame0.copy())
    #
    # test_match_template(frame0)
    # test_match_template_small_img(frame0)
    # frame0 = frame0[5:-5, 5:-5, :]
    # start(frame0)

    # field_points = find_best_field_point(frame0)
    # for field_point in field_points:
    #     print(field_point.point_name)
    #     for specific_field_point in field_point.points:
    #         print(specific_field_point, end=" ")
    #         if specific_field_point.point_in_detected_field_point((240, 310)):
    #             print("true")
    #         else:
    #             print("false")
    #         bottom_right = (specific_field_point.point[0] + specific_field_point.image_width,
    #         specific_field_point.point[1] + specific_field_point.image_height)
    #         cv2.rectangle(frame0, specific_field_point.point, bottom_right, 255, 2)
    # plt.imshow(frame0)
    # plt.show()

    # img = cv2.imread('TestImages/soccer_image7.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # field = cv2.imread('TestImages/soccerfield2d.jpg')
    #
    # og_img_dominant_color = get_dominant_color(img.copy())
    # deleted_crowd_image = delete_crowd_in_field(img.copy())
    # tophat_image = get_tophat_image(deleted_crowd_image, og_img_dominant_color)
    # the_lines = get_lines(tophat_image.copy(), img.copy())
    # print(get_intersection_points(the_lines, img))
