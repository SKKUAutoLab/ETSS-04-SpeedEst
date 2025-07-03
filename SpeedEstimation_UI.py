"""
Copy-Right by: PHAN HONG SON
Univ: Sungkyunkwan University
"""
# limit the number of cpus used by high performance libraries
import os
import numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import math
import sys
import os.path as osp
import argparse
import shutil
import time
import cv2
import torch
import json
import torch.backends.cudnn as cudnn
from supportSpeed.support_speed_estimation import draw_traject, speed_estimation, mapping_pose, map_detection, write_result
from supportSpeed.support_multithread_online import *
from loguru import logger
from pathlib import Path
# Son deploy multithreading for  speed up Performent
import threading
from PyQt6 import QtCore

# join path
sys.path.insert(0, './yolov5')
# sys.path.insert(0, './mmposeLink/mmpose')

from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

# Son Implement for YoloV5
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box

from config_pose_for_speed.inferpose import inference_top_down_pose_model, init_pose_model
# from mmposeLink.mmpose.mmpose.apis import (inference_top_down_pose_model, init_pose_model,
#                          vis_pose_result, process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from position_confix import Vps

local_runtime = False




FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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

# Config file to load model to infer Pose
pose_config = 'config_pose_for_speed/custom_config.py' #'mmposeLink/mmpose/configs/CarCornerConfig/my_custom_config.py'#'config_pose_for_speed/custom_config.py'
pose_checkpoint = 'mmposeLink/mmpose/BackupPre-Train/License_Us/epoch_33.pth' #'mmposeLink/mmpose/work_dirs/my_custom_config_60_Epochs/epoch_52.pth' #../../../../media/sonskku/DATA_2/BackupPre-Train/my_custom_config_center/epoch_30.pth' #'mmposeLink/mmpose/BackupPre-Train/License_Us/epoch_33.pth'#mmposeLink/mmpose/work_dirs/my_custom_config_60_Epochs/epoch_60.pth' #' '


json_structure = {'cars': [], 'camera_calibration': {}}

# Dist for draw trajectory
centroids = {}

# initialize pose model
pose_model = init_pose_model(pose_config, pose_checkpoint)
class PoseThread(threading.Thread):
    def __init__(self, image, boxs):
        threading.Thread.__init__(self)
        self.list_pose = None
        self.img = image
        self.boxs = boxs

    def run(self):
        self.list_pose = self.inferpose(self.img, self.boxs)

    def inferpose(self, image, boxs):
        pose_list = []
        result_boxs = []
        img = image
        for box in boxs:
            result_boxs.append({'bbox': box})
        # inference pose
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            result_boxs,
            bbox_thr=0.3,
            format='xyxy',
            dataset=pose_model.cfg.data.test.type)
        # Save result in a tuple
        for i in range(len(pose_results)):
            pose_list.append((pose_results[i]['keypoints'][0][0], pose_results[i]['keypoints'][0][1]))
        return pose_list
# Define class Track for support Witer data to evaluate in Brnocompspeed
class Track:
    def __init__(self, box, pose, id, frame):
        print("Call Initialize")
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
        print("===> Call check miss of id: {} with missing number {}: ".format(self.id,  self.missing))
        return self.missing > keeping_time
def addEntry(track):
    if len(track.frames) < 5:
        return
    frames = []
    posX = []
    posY = []
    print("====>Call  addRecord")
    for frame, pose, box in zip(track.frames, track.poses, track.boxs):
        posX.append(float(pose[0]))
        posY.append(float(pose[1]))
        frames.append(frame)

    if len(frames) < 5:
        return

    dist = math.sqrt(math.pow(posX[0] - posX[-1], 2) + math.pow(posY[0] - posY[-1], 2))
    if dist > 30:
        entry = {'frames': track.frames, 'id': track.id, 'posX': posX, 'posY': posY}
        print("=======> entry: ", entry)
        json_structure['cars'].append(entry)

def remove(list_track, list_ids):
    for i in reversed([i for (i, track) in enumerate(list_track) if track.check_misses(5)]):
        print("========> Call remove")
        addEntry(list_track[i])
        del list_track[i]
        del list_ids[i]

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
    parser.add_argument('--show-vid', action='store_true', default=[True], help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, default=[2], help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
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

def main(opt, filename, h_matrix, update_video):
    out, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
    project, exist_ok, update, save_crop = \
        opt.output, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, [1920,1080], opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    #SONDEBUG

    json_structure["camera_calibration"] = Vps[filename.split("/")[-2]]['camera_calibration']
    homo_matrix = h_matrix
    source = filename #"../Datasets/brnocompspeed/data/2016-ITS-BrnoCompSpeed/dataset/session1_left/video.avi"
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    #get area to tracking
    area_track = dict_area_track[filename.split("/")[-2]]

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    print("=========This is image size Before=======: ", imgsz)
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    print("This is image size: ", imgsz)

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    # if show_vid:
    #     show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        print('++++ Pass video')
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    print("================Phan Hong Son 9999 ===============")
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    # Son Define Init Bye track
    tracker = BYTETracker(opt, frame_rate=30)
    # timer = Timer()
    frame_id = 0
    results = []
    list_ids = []
    track_list = [] # this value was use for write data to json

    # with open('result_detection/results/Asave/session4_left/system_PSE.json') as json_file:
    #     data = json.load(json_file) # File to get calibrate info

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        startTime = time.time()
        # t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # t2 = time_sync()
        # dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        # t3 = time_sync()
        # dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        # dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                output = det[:, 0:5]
                output = output.cpu().numpy()
                confs = det[:, 4]
                clss = det[:, 5]

                # DEBUG filter box in a zone_track  ==> Track it
                output_track = []
                for box in output:
                    x_min,y_min,x_max,y_max = box[0],box[1],box[2],box[3]
                    centroid = (np.float32(x_min/2 + x_max/2), np.float32(y_min/2 + y_max/2))
                    flag = cv2.pointPolygonTest(np.array(area_track, np.int32), centroid, False)
                    if flag>=0:
                        #DEBUG Extended box and add to output_track
                        box1 = np.array([x_min-10,y_min-10,x_max+10,y_max+10,box[4]])
                        # print('This is box: ', box)
                        output_track.append(box1)

                output_track = np.array(output_track)


                # print("===========This is output_track=======>: ", output_track)
                # Create the inference pose thread
                pose_thread = PoseThread(im0, output_track)
                pose_thread.start()

                info_img = [im0.shape[0], im0.shape[1]]
                img_size =(im0.shape[0], im0.shape[1])
                if len(output_track) > 0:
                    # Son Start apply Bye Track
                    online_targets = tracker.update(output_track, info_img, img_size)
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
                    # timer.toc()
                    # Plot traking result
                    online_im = plot_tracking(
                        im0, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=30
                    )
                    print("Pass tracking ===========> ")
                    # # # # if frame_idx%2==0:
                    # start = time.time()
                    # # Call infer pose
                    # list_pose = inferpose(im0, output_track)
                    # total = time.time() - start
                    # print("================total time: ", total)
                    pose_thread.join()
                    list_pose = pose_thread.list_pose

                    # Call map pose and idx and it's bbox
                    mapped_pose = mapping_pose(list_pose, online_ids, online_tlwhs)
                    print("=========>> mapped_pose: ", mapped_pose)
                    print("=========>> online_ids: ", online_ids)

                    # DEBUG Deploy for write data to json
                    for i, id in enumerate(online_ids):
                        # condition to check whatever have bbox but haven't pose:
                        if id not in mapped_pose:
                            continue
                        print("====> Id in add Track: ", id)
                        print("====> len track list: ", len(track_list))

                        if len(track_list) == 0:
                            new_track = Track(online_tlwhs[i], mapped_pose[id], id, frame_id)
                            track_list.append(new_track)
                            list_ids.append(id)
                            continue
                        for track in track_list:
                            if id in list_ids:
                                if id == track.id:
                                    track.assign_info(online_tlwhs[i], frame_id, mapped_pose[id])
                                else:
                                    continue
                            else:
                                # print("=========> pass here: ", id)
                                list_ids.append(id)
                                new_track = Track(online_tlwhs[i], mapped_pose[id], id, frame_id)
                                track_list.append(new_track)
                    print("===> List id: ", list_ids)
                    # print("=========>> track_list: ", track_list)
                    #
                    #DEBUG Call calculate speed for each track
                    for track in track_list:
                        print('This is track: ', track.id)
                        if track.frames[-1] - track.frames[0] == 20:
                            print('==> This is track: ', track.id)
                            # speed = calculateSpeeds(data, track)
                            #DEBUG
                            speed = speed_estimation_inside(track, homo_matrix)
                            print('This is  speed of car: ', speed)
                            track.assign_speed(speed)
                    for id in list_ids:
                        print('This is list id: ', list_ids)
                        if id not in mapped_pose:
                            continue
                        print('this is pose: {} of id: {} '.format(mapped_pose[id], id))
                        # cv2.circle(online_im, mapped_pose[id], 5, (0, 0, 255), -1) #Draw pose
                        for track in track_list:
                            if track.id == id and track.speed > 0:
                                cv2.putText(online_im, str("{0:,.2f}".format(track.speed)) + 'km/h', (int(mapped_pose[id][0]), int(mapped_pose[id][1])), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)


                        # #DEBUG for draw trajectory
                        #     # Save centroid to distionary
                        #     if id in centroids.keys():
                        #         centroids[id].append(mapped_pose[id])
                        #     else:
                        #         centroids[id] = [mapped_pose[id]]

                    # # Draw the trajectory of centroid
                    # for id in centroids.keys():
                    #     for c in range(1, len(centroids[id])):
                    #         if centroids[id][c - 1] is None or centroids[id][c] is None:
                    #             continue
                    #         cv2.line(online_im, centroids[id][c - 1], centroids[id][c], (0, 0, 255), 2)


                    # Call to write json file when accepted
                    remove(track_list, list_ids)


                else:
                    online_im = im0  #  insert in case output_track = [None]
            else:
                online_im = im0 ##  insert in case det = [None]

            # Debug
            totalTime = time.time() - startTime
            fps = 1 / totalTime
            print("FPS: {:.2f}".format(fps))

            #DEBUG draw zone_track area
            cv2.polylines(online_im, [np.array(area_track, np.int32)], True, (0, 255, 0), 3)


            frame_id += 1
            # Son End Bye Track
            # Stream results
            im0 = annotator.result()
            # Update image to signal
            update_video.emit({
                "image": online_im
            })

            # if show_vid:
            #     cv2.imshow(str(p), online_im)
            #     cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                print('This is save path',save_path)
                print('This is vid path',len(vid_path))
                if vid_path[0] != save_path:  # new video
                    print("Phan Hong Son")
                    vid_path[0] = save_path
                    if isinstance(vid_writer[0], cv2.VideoWriter):
                        vid_writer[0].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 50, online_im.shape[1], online_im.shape[0]
                    save_path = str(Path(save_path).with_suffix('.avi'))  # force *.mp4 suffix on results videos
                    # vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[0] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))
                vid_writer[0].write(online_im)

        # # Debug
        # totalTime = time.time() - startTime
        # fps = 1 / totalTime
        # print("FPS: {:.2f}".format(fps))
    print("============================= FINISH =============================")
    output_dir = "Output_result/{}".format(filename.split("/")[-2])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_file = osp.join(output_dir, "system_PSE.json")
    with open(out_file, "w") as f:
        json.dump(json_structure, f)

    # # write result to json
    # flag_cap = dataset.get_cap()
    # print("this is flag cap: ", flag_cap)
    # write_result(flag_cap)

# Fuction subport speed
def speed_estimation_inside(track, H):
    startFrame = track.frames[0]
    endFrame = track.frames[-1]
    elapsedTime = abs(endFrame - startFrame) / fps
    startPoint = np.array([track.poses[0][0], track.poses[0][1], 1])
    endPoint = np.array([track.poses[-1][0], track.poses[-1][1], 1])
    distance = math.sqrt(np.sum(np.square(getWorldPoint(endPoint, H)[0:2]/getWorldPoint(endPoint, H)[2]- getWorldPoint(startPoint, H)[0:2]/getWorldPoint(startPoint, H)[2])))
    print('This is world point {} and {} '.format(getWorldPoint(endPoint, H)[0:2], getWorldPoint(startPoint, H)[0:2]))
    print('This is distance: ', distance)
    return distance*0.94656/ elapsedTime * 3.6

def getWorldPoint(point, H):
    return np.dot(H, point)

def speedEstimation(filename, matrix, update_video):
    opt = make_parser()
    main(opt, filename, matrix, update_video)
