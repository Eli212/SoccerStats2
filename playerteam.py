import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_dominant_color(the_image):
    Z = the_image.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 1
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(the_image.shape)
    return res2[0][0]


def get_kmeans_teams_arr(colors_arr):
    from sklearn.cluster import KMeans
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=0).fit(colors_arr)
    idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
    lut = np.zeros_like(idx)

    # print(kmeans.cluster_centers_.sum(axis=1))
    return kmeans.labels_


def get_avg_color(colors_arr):
    r = 0
    g = 0
    b = 0
    colors_arr_len = len(colors_arr)
    for color in colors_arr:
        r += color[0]
        g += color[1]
        b += color[2]
    new_rgb = [int(r/colors_arr_len), int(g/colors_arr_len), int(b/colors_arr_len)]
    return new_rgb

# player_img_green1 = cv2.imread("sources/TestImages/green1.png")
# player_img_green1 = cv2.cvtColor(player_img_green1, cv2.COLOR_BGR2RGB)
# dom_color_green1 = get_dominant_color(player_img_green1)
#
# player_img_red1 = cv2.imread("sources/TestImages/red1.png")
# player_img_red1 = cv2.cvtColor(player_img_red1, cv2.COLOR_BGR2RGB)
# dom_color_red1 = get_dominant_color(player_img_red1)
#
# player_img_red2 = cv2.imread("sources/TestImages/red2.png")
# player_img_red2 = cv2.cvtColor(player_img_red2, cv2.COLOR_BGR2RGB)
# dom_color_red2 = get_dominant_color(player_img_red2)

# X = np.vstack((dom_color_green1, player_img_red1, dom_color_red2))
# X_morning = np.random.uniform(low=.02, high=.18, size=38)
# X_afternoon = np.random.uniform(low=.05, high=.20, size=38)
# X_night = np.random.uniform(low=.025, high=.175, size=38)
# X = np.vstack([X_morning, X_afternoon, X_night]).T
