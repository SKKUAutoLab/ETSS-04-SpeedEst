"""
Copy-Right by: PHAN HONG SON
Univ: Sungkyunkwan University
"""
# import nescessary library
import numpy as np
import math
import cv2

# Init Last Tracking and Curent tracking
area_1 = [(265.48015512, 269.03789474), (1445.48015512, 269.03789474), (1445.48015512, 458.03789474), (265.48015512, 458.03789474)]
area_2 = [(355.2531856, 617.27887719), (1906, 617.27887719), (1906, 1058.27887719), (355.2531856, 1058.27887719)]
area_3 = [(337.61311911, 237.74363066), (834.39104709, 190.72425485), (1777.91043, 689.098578), (870.39184488, 807.86437673)]
centroids = {}
entering_car = {}
total_time = {}
NUM_FRAME = 50 # the number of frame using for calculating speed

# function to calculate the distance between two point1, point2
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

#function to get match point from image to 3D coordinate.
def get_match_point(point):
    src_point = np.append(point, 1)
    # This is homography constant was calculated by Camera calibration by PHAN HONG SON in matlab.
    H = np.array([[2.22510852e-02, -2.07936738e-02, -2.56869477e+00],
                  [1.36458639e-02, 1.44173841e-01, -3.88834350e+01],
                  [1.27450462e-04, 2.59918447e-03, 1.00000000e+00]])
    cal_point = np.dot(H, src_point)
    # Convert point from 3D to 2D detail view https://towardsdatascience.com/understanding-homography-a-k-a-perspective-transformation-cacaed5ca17
    des_point = np.array([cal_point[0]/cal_point[2], cal_point[1]/cal_point[2]])
    return des_point

# function for drawing the trajectory of poses
def draw_traject(image, centroids):
    for id in centroids.keys():
        for c in range(1, len(centroids[id])):
            if centroids[id][c - 1] is None or centroids[id][c] is None:
                continue
            cv2.line(image, centroids[id][c - 1], centroids[id][c], (0, 0, 255), 2)

# function to calculate speed
def speed_estimation(frame_idx, online_im, online_tlwhs, online_ids):
    for j, tlwh in enumerate(online_tlwhs):
        x, y, w, h = tlwh
        box = tuple(map(int, (x, y, x + w, y + h)))
        centroid = (int(x + w / 2), int(y + h / 2))
        obj_id = online_ids[j]
        # Save centroid to distionary
        if obj_id in centroids.keys():
            centroids[obj_id].append(centroid)
        else:
            centroids[obj_id] = [centroid]
        flag = cv2.pointPolygonTest(np.array(area_1, np.int32), centroid, False)
        if flag >= 0:
            entering_car[obj_id] = frame_idx  # time.time()
        if obj_id in entering_car:
            flag = cv2.pointPolygonTest(np.array(area_2, np.int32), centroid, False)
            if flag >= 0:
                spend_time = frame_idx - entering_car[obj_id]  # time.time() - entering_car[obj_id]
                if obj_id not in total_time:
                    total_time[obj_id] = spend_time
                if obj_id in total_time:
                    spend_time = total_time[obj_id]
                    print('This is spending time: ', total_time)
                # Caculate speed
                # # son test homorgraphy
                # point = np.array([973.39000554, 264.19696399])
                # des_point = get_match_point(point)
                # print('=====>  This is mathpoint: ', des_point)


                distance = 6.93  # meters
                speed_ms = distance / (spend_time * 0.02)
                speed_kh = speed_ms * 3.6
                speed_kh = "{0:,.2f}".format(speed_kh)
                # Drawing result
                cv2.rectangle(online_im, box[0:2], box[2:4], color=(0, 255, 0), thickness=2)
                cv2.putText(online_im, str(speed_kh) + ' km/h', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), thickness=2)
                cv2.circle(online_im, centroid, 5, (255, 0, 0), -1)

        # Son Plot bounding box
        # cv2.polylines(online_im, [np.array(area_1, np.int32)], True, (0, 0, 255), 3)
        # cv2.polylines(online_im, [np.array(area_2, np.int32)], True, (0, 0, 255), 3)
        cv2.polylines(online_im, [np.array(area_3, np.int32)], True, (0, 255, 0), 3)

    # Plot Trajectory of centroid
    draw_traject(online_im, centroids)