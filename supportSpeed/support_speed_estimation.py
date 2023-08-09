"""
Copy-Right by: PHAN HONG SON
Univ: Sungkyunkwan University
"""
# import nescessary library
import numpy as np
import math
import cv2
import sys
import json
import os.path as osp
import time
# sys.path.insert(0, './mm')


# Init Last Tracking and Curent tracking
area_1 = [(265.48015512, 269.03789474), (1445.48015512, 269.03789474), (1445.48015512, 458.03789474), (265.48015512, 458.03789474)]
area_2 = [(355.2531856, 617.27887719), (1906, 617.27887719), (1906, 1058.27887719), (355.2531856, 1058.27887719)]
area_3 = [(337.61311911, 237.74363066), (834.39104709, 190.72425485), (1777.91043, 689.098578), (870.39184488, 807.86437673)]

# area_4 = [(229, 132), (820, 73), (1912, 571), (900, 856)]# ===> session4_left
# area_4 = [(636, 37), (1170, 69), (1914, 961), (612, 985)]# ===> session4_center
# area_4 = [(1357, 270), (1908, 322), (1444, 1003), (406, 787)]# ===> session4_right
# area_4 = [(114, 246), (441, 189), (1693, 634), (991, 976)] # ===> session5_left
# area_4 = [(1140, 76), (1546, 82), (1429, 1036), (228, 838)]# ===> session5_right
# area_4 = [(447, 342), (1071, 315), (1857, 753), (517, 868)]# ===> session6_center

# area_4 = [(220, 126), (751, 19), (1919, 627), (1107, 999)]# ===> session1_left
# area_4 = [(850, 60), (1332, 63), (1636, 1047), (382, 1011)]# ===> session1_center
area_4 = [(1212, 84), (1734, 120), (1393, 1068), (322, 823)]# ===> session1_right

# area_4 = [(348, 157), (922, 111), (1701, 756), (519, 928)]# ===> session3_left
# area_4 = [(1200, 72), (1903, 64), (1900, 969), (228, 846)]# ===> session2_right
# area_4 = [(387, 472), (786, 438), (1516, 670), (723, 829)]# ===> session6_left


centroids = {}
entering_car = {}
total_time = {}
position = {}
speed_result = {}
eff = {}
list_id_in_result = []
vp1 = [672.9, -509.72]
go_z = 0
# Json file to export result to evaluate speed estimation in my Paper
result = dict(
        camera_calibration = dict(
            pp = [960.5, 540.5],
            scale = 0.0153809509,
            vp1 = [2295.38, -247.55],
            vp2 = [-6030.86093393578, -138.44094778385198]
        ),
    cars = [
        # list [{frames, id, posX, posY}]
    ]
)

NUM_FRAME = 35 # the number of frame using for calculating speed

# function to calculate the distance between two point1, point2
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

#function to get match point from image to 3D coordinate.
def get_match_point(point):
    src_point = np.append(point, 1)
    # This is homography constant was calculated by Camera calibration by PHAN HONG SON in matlab.
    # H = np.array([[2.22510852e-02, -2.07936738e-02, -2.56869477e+00],
    #               [1.36458639e-02, 1.44173841e-01, -3.88834350e+01],
    #               [1.27450462e-04, 2.59918447e-03, 1.00000000e+00]])
    H = np.array([[ 2.23248225e-02, -2.08625814e-02, -2.57720710e+00],
                [ 1.37239783e-02,  1.44999149e-01, -3.91060193e+01],
                 [ 1.28992762e-04,  2.62027346e-03,  1.00000000e+00]])

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

# function for mapping ids and bbox and with pose
def mapping_pose(list_pose,ids,bbox):
    mapped_pose = {}
    for i, tlwh in enumerate(bbox):
        object_id = ids[i]
        x, y, w, h = tlwh
        are_box = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        # print('this is box: ', are_box)
        for j, p in enumerate(list_pose):
            # print("====> this is pose {}th : {}".format(j,p))
            # check whatever this pose include in box
            flag = cv2.pointPolygonTest(np.array(are_box, np.int32), p, False)
            if flag>=0:
                mapped_pose[object_id] = p
                # print(('Mapped at {} th: {}'.format(i,mapped_pose)))
    return mapped_pose


#funtion to fine mapped point in road lane
def find_eff(point,vp):
    point_n= [point[0], point[1]+go_z]
    k = (vp[1]-point_n[1])/(vp[0]-point_n[0])
    b = point_n[1] - k*point_n[0]
    return k, b
def find_point_in_road(point,k,b):
    y = k*point[0]+b
    return y

# function to save resutl to evaluation onto Brnocompspeed
def export_resutl(frame, id, pose):
    print(" Call export_resutl")
    # dict to put in result
    detection = dict(
        frames=[],
        id = 0,
        posX=[],
        posY=[],
    )
    detection["frames"].append(frame)
    detection["id"] = id
    detection["posX"].append(np.float(pose[0]))
    detection["posY"].append(np.float(pose[1] + go_z))
    print(" Finish ==>Call export_resutl")
    print(detection)
    return detection

# Mapping detection function
def map_detection(image, frame_idx, online_tlwhs, online_ids, mapped_pose):
    start_time = time.time()
    print("Call map_detection")
    for i, tlwh in enumerate(online_tlwhs):
        print("this is bbox: {}th".format(i))
        obj_id = online_ids[i]
        # # Don't update information of id 1. Because it is not moving.
        # if obj_id == 1: # or obj_id == 6:
        #     continue

        # condition to check whatever have bbox but haven't pose:
        if obj_id in mapped_pose:
            pose = mapped_pose[obj_id]
        else:
            continue

        # #DEBUG
        # if obj_id <=2000:
        #     x, y, w, h = tlwh
        #     pose = (np.float32(x + w / 2), np.float32(y + h))


        flag = cv2.pointPolygonTest(np.array(area_4, np.int32), pose, False)
        if flag >= 0:
            print('This is id: {}th in list '.format(obj_id))
            print('This is len of  result["car"]: {} '.format(len(result["cars"])))
            # if len(result["cars"]) == 0:
            #     next_id = export_resutl(frame_idx, obj_id, pose)
            #     result["cars"].append(next_id)
            #     print(" =====> This is len result: ", len(result["cars"]))
            #     continue
            if obj_id in list_id_in_result:
                for car in result["cars"]:
                    print("++++ object id: {}, and Car: {}".format(obj_id,car["id"]))
                    if obj_id == car["id"]and len(car["frames"])<=200:                        #and  len(car["frames"])<=54
                        print("==============================Pass here===========================")

                        # #DEBUG
                        # [k1,k2] = eff[obj_id]
                        # y_z = find_point_in_road(pose,k1,k2)
                        # #END

                        car["frames"].append(frame_idx)
                        car["posX"].append(np.float(pose[0]))
                        car["posY"].append(np.float(pose[1] + go_z))
                        # print("==============> This is result: ", result["cars"])
                        break # because car["id"] is unique
            else:
                # #DEBUG
                # k, b = find_eff(pose,vp1)
                # eff[obj_id] = [k,b]
                # #end

                next_id = export_resutl(frame_idx, obj_id, pose)
                result["cars"].append(next_id)
        # save object id was saved
        list_id_in_result.append(obj_id)
        # Draw pose in image
        # #Debug
        # print("===> pose: ", pose)
        # cv2.circle(image, (int(pose[0]), int(pose[1])), 5, (0, 0, 255), -1)
        cv2.circle(image, pose, 5, (0, 0, 255), -1)

        if obj_id>=1:
            flag = False
            write_result(flag)
        total_time = time.time() - start_time
        print("====>This is total time: ", total_time)


# function to save result to  json file
def write_result(flag):
    if not flag:
        print("Call write_result to json")
        output_dir = "result_detection/"
        out_file = osp.join(output_dir, "system_PSP.json")
        with open(out_file, "w") as f:
            json.dump(result, f)
        # sys.exit()


# function to calculate speed
def speed_estimation(frame_idx, online_im, online_tlwhs, online_ids, mapped_pose):
    # print('+++++++> Call speed_estimation +++at frame: +',frame_idx)
    for j, tlwh in enumerate(online_tlwhs):
        x, y, w, h = tlwh
        box = tuple(map(int, (x, y, x + w, y + h)))
        # centroid = (np.float32(x + w / 2), np.float32(y + h / 2))
        obj_id = online_ids[j]
        # condition to check whatever have bbox but haven't pose:
        if obj_id in mapped_pose:
            pose = mapped_pose[obj_id]
        else:
            continue
        # Save centroid to distionary
        if obj_id in centroids.keys():
            centroids[obj_id].append(pose)
        else:
            centroids[obj_id] = [pose]
        flag = cv2.pointPolygonTest(np.array(area_3, np.int32), pose, False)
        if flag >= 0 and (obj_id not in entering_car):
            print("=========> Pass here<===================")
            entering_car[obj_id] = frame_idx  # time.time()
            position[obj_id] = pose

        if obj_id in entering_car:
            if frame_idx==(entering_car[obj_id]+NUM_FRAME):
                last_pos = get_match_point(position[obj_id])
                cur_pos = get_match_point(pose)
                distance = calculate_distance(last_pos, cur_pos)
                print('This is distance of first car: {} of ID: {} '.format(distance,  obj_id))
                cv2.circle(online_im, pose, 30, (0, 255, 255), -1)

                speed_ms = distance / (NUM_FRAME * 0.02356)
                speed_kh = speed_ms * 3.6
                speed_kh = "{0:,.2f}".format(speed_kh)
                print('======> This is speed of ID: {} with speed: {}'.format(obj_id,speed_kh))
                speed_result[obj_id]  = speed_kh
                print('=====> This is speed result: ', speed_result)
                # Drawing result
                cv2.rectangle(online_im, box[0:2], box[2:4], color=(0, 255, 0), thickness=2)
                cv2.putText(online_im, str(speed_kh) + ' km/h', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), thickness=2)
            # flag = cv2.pointPolygonTest(np.array(area_2, np.int32), centroid, False)
            # if flag >= 0:
            #     spend_time = frame_idx - entering_car[obj_id]  # time.time() - entering_car[obj_id]
            #     if obj_id not in total_time:
            #         total_time[obj_id] = spend_time
            #     if obj_id in total_time:
            #         spend_time = total_time[obj_id]
            #         print('This is spending time: ', total_time)
            #     # Caculate speed
            #     # # son test homorgraphy
            #     # point = np.array([973.39000554, 264.19696399])
            #     # des_point = get_match_point(point)
            #     # print('=====>  This is mathpoint: ', des_point)
            #     distance = 6.93  # meters
            #     speed_ms = distance / (spend_time * 0.02)
            #     speed_kh = speed_ms * 3.6
            #     speed_kh = "{0:,.2f}".format(speed_kh)
            #     # Drawing result
            #     cv2.rectangle(online_im, box[0:2], box[2:4], color=(0, 255, 0), thickness=2)
            #     cv2.putText(online_im, str(speed_kh) + ' km/h', (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), thickness=2)
        cv2.circle(online_im, pose, 5, (0, 0, 255), -1)

        # Son Plot bounding box
        # cv2.polylines(online_im, [np.array(area_1, np.int32)], True, (0, 0, 255), 3)
        # cv2.polylines(online_im, [np.array(area_2, np.int32)], True, (0, 0, 255), 3)
        cv2.polylines(online_im, [np.array(area_3, np.int32)], True, (0, 255, 0), 3)

    # Plot Trajectory of centroid
    # draw_traject(online_im, centroids)