"""
Copy-Right by: PHAN HONG SON
Univ: Sungkyunkwan University
"""
import cv2
from ultralytics import YOLO
import numpy as np
import sys
# join path
sys.path.insert(0, '.')
import time
import math
import torch
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
import argparse
import os.path as osp
import json
import os
from support.position_confix import Vps, Hmatrixs

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    # Json args
    parser.add_argument("--json_path", type=str, default="result_detection/result", help="json file path")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


# Define a zone to traking
dict_area_track = {
"session1_left": [(220, 126), (751, 19), (1919, 627), (1107, 999)], # ===> session1_left
"session1_center":  [(850, 60), (1332, 63), (1636, 1047), (382, 1011)],# ===> session1_center
"session1_right":  [(1212, 84), (1734, 120), (1393, 1068), (322, 823)], # ===> session1_right

"session2_left": [(264, 94), (769, 61), (1790, 670), (622, 846)],# ===> session2_left
"session2_center": [(591, 144), (1239, 152), (1911, 773), (380, 875)],# ===> session2_center
"session2_right": [(1200, 72), (1903, 64), (1900, 969), (228, 846)],# ===> session2_right

"session3_left": [(348, 157), (922, 111), (1701, 756), (519, 928)],# ===> session3_left
"session3_center": [(813, 46), (1455, 49), (1902, 945), (414, 951)],# ===> session3_center
"session3_right": [(1053, 138), (1572, 136), (1665, 822), (480, 790)],# ===> session3_right

"session4_left": [(229, 132), (820, 73), (1912, 571), (900, 856)],# ===> session4_left
"session4_center": [(636, 37), (1170, 69), (1914, 961), (612, 985)],# ===> session4_center
"session4_right": [(1357, 270), (1908, 322), (1444, 1003), (406, 787)],# ===> session4_right

"session5_left": [(114, 246), (441, 189), (1693, 634), (991, 976)],# ===> session5_left
"session5_center": [(882, 60), (1306, 54), (1803, 1014), (316, 950)],# ===> session5_center
"session5_right": [(1140, 76), (1546, 82), (1429, 1036), (228, 838)],# ===> session5_right

"session6_left": [(387, 472), (786, 438), (1516, 670), (723, 829)],# ===> session6_left
"session6_center": [(447, 342), (1071, 315), (1857, 753), (517, 868)],# ===> session6_center
"session6_right": [(1098, 98), (1801, 95), (1723, 779), (168, 678)],# ===> session6_right
}


# Define class Track for support Witer data to evaluate in Brnocompspeed
class Track:
    def __init__(self, box, pose, id, frame):
        # print("Call Initialize")
        self.boxs = [box]
        self.poses = [pose]
        self.id = id
        self.frames = [frame]
        self.missing = -1
        self.speed = 0

    def assign_info(self, box, frame, pose):
        self.boxs.append(box)
        self.frames.append(frame)
        self.poses.append(pose)
        self.missing = -1

    def assign_speed(self, speed):
        self.speed = speed

    def check_misses(self, keeping_time):
        self.missing += 1
        # print("===> Call check miss of id: {} with missing number {}: ".format(self.id,  self.missing))
        return self.missing > keeping_time



# Load the model (ensure it supports car detection)
model = YOLO("pretrain/bests.pt")
# model = YOLO("yolo11s.pt")


def main(opt, filename):
    # Open the video file
    video_path = filename
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    #get area coords from dict_area_track
    area_coords = np.array(dict_area_track[filename.split("/")[-2]], dtype=np.int32)

    # Define json structure
    json_structure = {'cars': [], 'camera_calibration': {}}
    json_structure["camera_calibration"] = Vps[filename.split("/")[-2]]['camera_calibration']
    homo_matrix = Hmatrixs[filename.split("/")[-2]]['H']


    def addEntry(track):
        if len(track.frames) < 5:
            return
        frames = []
        posX = []
        posY = []
        # print("====>Call  addRecord")
        for frame, pose, box in zip(track.frames, track.poses, track.boxs):
            posX.append(float(pose[0]))
            posY.append(float(pose[1]))
            frames.append(frame)

        if len(frames) < 5:
            return

        dist = math.sqrt(math.pow(posX[0] - posX[-1], 2) + math.pow(posY[0] - posY[-1], 2))
        if dist > 30:
            entry = {'frames': track.frames, 'id': track.id, 'posX': posX, 'posY': posY}
            # print("=======> entry: ", entry)
            json_structure['cars'].append(entry)

    def remove(list_track, list_ids):
        for i in reversed([i for (i, track) in enumerate(list_track) if track.check_misses(5)]):
            # print("========> Call remove")
            addEntry(list_track[i])
            del list_track[i]
            del list_ids[i]

    def speed_estimation_inside(track, H):
        startFrame = track.frames[0]
        endFrame = track.frames[-1]
        fps = 50
        elapsedTime = abs(endFrame - startFrame) / fps
        startPoint = np.array([track.poses[0][0], track.poses[0][1], 1])
        endPoint = np.array([track.poses[-1][0], track.poses[-1][1], 1])
        distance = math.sqrt(np.sum(np.square(getWorldPoint(endPoint, H)[0:2]/getWorldPoint(endPoint, H)[2]- getWorldPoint(startPoint, H)[0:2]/getWorldPoint(startPoint, H)[2])))
        # print('This is world point {} and {} '.format(getWorldPoint(endPoint, H)[0:2], getWorldPoint(startPoint, H)[0:2]))
        # print('This is distance: ', distance)
        return distance/ elapsedTime * 3.6
    
    def getWorldPoint(point, H):
        return np.dot(H, point)
    
    # Some Define intialize
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = BYTETracker(opt,frame_rate=50)
    results = []
    track_list = []
    list_ids = []
    frame_id = 0
    info_img = [height, width]
    img_size = (height, width)
    zone_mask = None

     # === Add this block to set up VideoWriter ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_dir = "Output_result/{}".format(filename.split("/")[-2])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # out_video_path = osp.join(output_dir, "result.mp4")
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        start_time = time.time()

        if zone_mask is None:
            frame_h, frame_w = frame.shape[:2]
            zone_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            cv2.fillPoly(zone_mask, [area_coords], 1)

        # Perform tracking on the frame, filtering for cars (class index 2)
        results = model(frame,verbose=False)
        # print("Results: ", results)


        # filter boxes in the area_coords
        in_boxs = {}
        filtered_boxes = []
        filtered_cls = []
        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.shape[0] == 0:
                continue

            boxes_xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            centers_x = ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5).astype(np.int32)
            centers_y = ((boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5).astype(np.int32)
            centers_x = np.clip(centers_x, 0, zone_mask.shape[1] - 1)
            centers_y = np.clip(centers_y, 0, zone_mask.shape[0] - 1)

            inside_mask = zone_mask[centers_y, centers_x] > 0
            if not np.any(inside_mask):
                continue

            xyxy_inside = boxes_xyxy[inside_mask]
            conf_inside = confidences[inside_mask][:, None]
            filtered_boxes.extend(np.hstack((xyxy_inside, conf_inside)))
            filtered_cls.extend(classes[inside_mask].tolist())
            
                    
        filtered_boxes = np.array(filtered_boxes)
        in_boxs['filtered_boxes'] = filtered_boxes
        in_boxs['class'] = filtered_cls
        # print("in_boxs: ", in_boxs)
        
        # Apply ByteTrack Tracking
        if len(filtered_boxes) > 0:
            online_targets = tracker.update(filtered_boxes, info_img, img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > opt.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            # Plot traking result
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=fps)

            # DEBUG Deploy for write data to json
            for i, id in enumerate(online_ids):
                if len(track_list) == 0:
                    # Calculate the midpoint of the bottom edge of the bounding box
                    x, y, w, h = online_tlwhs[i]
                    midpoint = (x + w / 2, y + h)
                    new_track = Track(online_tlwhs[i], midpoint, id, frame_id)
                    track_list.append(new_track)
                    list_ids.append(id)
                    continue
                for track in track_list:
                    if id in list_ids:
                        if id == track.id:
                            # Calculate the midpoint of the bottom edge of the bounding box
                            x, y, w, h = online_tlwhs[i]
                            midpoint = (x + w / 2, y + h)
                            track.assign_info(online_tlwhs[i], frame_id, midpoint)
                        else:
                            continue
                    else:
                        # print("=========> pass here: ", id)
                        list_ids.append(id)
                        # Calculate the midpoint of the bottom edge of the bounding box
                        x, y, w, h = online_tlwhs[i]
                        midpoint = (x + w / 2, y + h)
                        new_track = Track(online_tlwhs[i], midpoint, id, frame_id)
                        track_list.append(new_track)
                        break  # prevent the loop from hitting the newly-added track and calling assign_info again

                #  #DEBUG Call calculate speed for each track
                # for track in track_list:
                #     # print('This is track: ', track.id)
                #     if track.frames[-1] - track.frames[0] == 20:
                #         # print('==> This is track: ', track.id)
                #         speed = speed_estimation_inside(track, homo_matrix)
                #         # print('This is  speed of car: ', speed)
                #         track.assign_speed(speed)
                for id in list_ids:
                    # print('This is list id: ', list_ids)
                    # cv2.circle(online_im, (int(mapped_pose[id][0]), int(mapped_pose[id][1])), 5, (0, 0, 255), -1) #Draw pose
                    #draw the midpoint of the bottom edge of the bounding box
                    cv2.circle(online_im, (int(midpoint[0]), int(midpoint[1])), 6, (0, 0, 255), -1)
                    
                    # for track in track_list:
                    #     if track.id == id and track.speed > 0:
                    #         mapped_pose = {track.id: track.poses[-1] for track in track_list}
                    #         cv2.putText(online_im, str("{0:,.2f}".format(track.speed)) + 'km/h', (int(mapped_pose[id][0]), int(mapped_pose[id][1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)


                remove(track_list, list_ids)
        else:
            online_im = frame # insert for case output_track = [None]
        
        # if frame_id % 3000 == 0:
        #     print("============================= Call write to Json at frame: {} =============================".format(frame_id))
        #     output_dir = "Output_result/{}".format(filename.split("/")[-2])
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #     out_file = osp.join(output_dir, "system_PSE.json")
        #     with open(out_file, "w") as f:
        #         json.dump(json_structure, f)


        # draw area_coords on the frame
        result_im = cv2.polylines(online_im, [area_coords], isClosed=True, color=(0, 255, 0), thickness=2)
        # # draw line count on the frame
        # cv2.line(result_im, line[0], line[1], (0, 0, 255), 2)
        # show  the result frame
        cv2.imshow("Result Frame", result_im)
         # === Write the frame to the output video ===
        # video_writer.write(result_im)    
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        frame_id += 1

    print("============================= Call write to Json =============================")
    output_dir = "Output_result/{}".format(filename.split("/")[-2])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_file = osp.join(output_dir, "system_PSE.json")
    with open(out_file, "w") as f:
        json.dump(json_structure, f)

    cap.release()
    cv2.destroyAllWindows()

#main function
if __name__ == "__main__":
    opt = make_parser()
    print(opt)
    vid_list = []
    # vid_path = '/home/sonskku/Documents/Datasets/brnocompspeed/data/2016-ITS-BrnoCompSpeed/dataset'
    vid_path = '/home/sonskku/Documents/Datasets/smallSetofBronocomspeed/dataset'
    for i in range(1, 7):
        dir_list = ['session{}_left'.format(i), 'session{}_right'.format(i), 'session{}_center'.format(i)]
        vid_list.extend([os.path.join(vid_path, d, 'video.mp4') for d in dir_list])
        
    for filename in vid_list:
            main(opt, filename)

