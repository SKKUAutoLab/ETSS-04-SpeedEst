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

import sys
import os.path as osp
import argparse
import shutil
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
from supportSpeed.support_speed_estimation import draw_traject, speed_estimation, mapping_pose, map_detection, write_result
from loguru import logger
from pathlib import Path
import sys

# Son deploy multithreading for  speed up performent
import threading

# join path
sys.path.insert(0, './yolov5')
sys.path.insert(0, './mmposeLinkV2/mmpose')

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



import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
# print(inference_topdown.path)
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False



local_runtime = False




FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Define a zone to traking
# area_track = [(220, 126), (751, 19), (1919, 627), (1107, 999)]# ===> session1_left
# area_track = [(850, 60), (1332, 63), (1636, 1047), (382, 1011)]# ===> session1_center
# area_track = [(1212, 84), (1734, 120), (1393, 1068), (322, 823)]# ===> session1_right

# area_track = [(348, 157), (922, 111), (1701, 756), (519, 928)]# ===> session3_left

# area_track = [(1200, 72), (1903, 64), (1900, 969), (228, 846)]# ===> session2_right



area_track = [(229, 132), (820, 73), (1912, 571), (900, 856)]# ===> session4_left
# area_track = [(636, 37), (1170, 69), (1914, 961), (612, 985)]# ===> session4_center
# area_track = [(1357, 270), (1908, 322), (1444, 1003), (406, 787)]# ===> session4_right
# area_track = [(114, 246), (441, 189), (1693, 634), (991, 976)]# ===> session5_left
# area_track = [(1140, 76), (1546, 82), (1429, 1036), (228, 838)]# ===> session5_right
# area_track = [(447, 342), (1071, 315), (1857, 753), (517, 868)]# ===> session6_center


# Config file to load model to infer Pose
pose_config = 'mmposeLinkV2/mmpose/configs/car_poses/car_pose_config.py'
pose_checkpoint = 'mmposeLinkV2/mmpose/work_dirs/car_pose_config/best_PCK_epoch_10.pth'
det_config = 'mmposeLinkV2/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'


# # initialize detector
# detector = init_detector(
#     det_config,
#     det_checkpoint,
#     device=device
# )

# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=None
)
def inferpose(image,boxs):
    pose_list = []
    result_boxs = []
    img = image

    # # extract Car (COCO_ID=3) bounding boxes from the detection results
    # person_results = process_mmdet_results(mmdet_results, cat_id=3)
    for box in boxs:
        print("=========>> box: ", box)
        result_boxs.append(box[0:4])
    # Debug
    print("==============> boxs result: ", result_boxs)

    # inference pose
    pose_results = inference_topdown(pose_estimator, img, result_boxs)
    data_samples = merge_data_samples(pose_results)
    pose_infer = data_samples.pred_instances.keypoints
    # Save result in a tuple
    for pose in pose_infer:
        print("=========>> pose: ", pose[0])
        pose_list.append((pose[0][0], pose[0][1]))
    print("==============> pose_list: ", pose_list)

    return pose_list


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
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
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
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
    project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, [1920,1080], opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

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
    if show_vid:
        show_vid = check_imshow()

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
    timer = Timer()
    frame_id = 0
    results = []
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        startTime = time.time()
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3



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
                # print("===========This is output=======>: ", output)


                # DEBUG filter box in a zone_track  ==> Track it
                output_track = []
                for box in output:
                    # print('========> Box', box)
                    x_min,y_min,x_max,y_max = box[0],box[1],box[2],box[3]
                    # print('+++++++++++ x: {}, y: {}, w: {}, h: {}'.format(x,y,w,h))
                    centroid = (np.float32(x_min/2 + x_max/2), np.float32(y_min/2 + y_max/2))
                    flag = cv2.pointPolygonTest(np.array(area_track, np.int32), centroid, False)
                    if flag>=0:
                        output_track.append(box)

                output_track = np.array(output_track)
                # print("===========This is output_track=======>: ", output_track)


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
                    timer.toc()
                    # Plot traking result
                    online_im = plot_tracking(
                        im0, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )

                    # # # if frame_idx%2==0:
                    start = time.time()
                    # Call infer pose
                    list_pose = inferpose(im0, output_track)
                    total = time.time() - start
                    print("================total time: ", total)

                    # Call map pose and idx and it's bbox
                    # print('======> Call mapping pose at frame: {} and list pose: {}'.format(frame_idx, list_pose))

                    mapped_pose = mapping_pose(list_pose, online_ids, online_tlwhs)
                    # print('====> This is mapped pose: ', mapped_pose)
                    # sys.exit()
                    # print("===================================> frame idx: ", frame_idx)
                    # print("===================================> frame id: ", frame_id)
                    # print("===================================> bbox: ", online_tlwhs)
                    #
                    # # DEBUG
                    # mapped_pose = []

                    # Call map detection to evaluate Brnocomspeed
                    map_detection(online_im, int(frame_idx), online_tlwhs, online_ids, mapped_pose)
                    print('====> finish is map_detection: ')
                    # sys.exit()
                    # # Call speed estimation function
                    # speed_estimation(frame_idx, online_im, online_tlwhs, online_ids, mapped_pose)


                else:
                    online_im = im0  #  insert in case output_track = [None]
            else:
                online_im = im0 ##  insert in case det = [None]

            #DEBUG draw zone_track area
            cv2.polylines(online_im, [np.array(area_track, np.int32)], True, (0, 255, 0), 3)
            # for box in output_track:
            #     x_min, y_min, x_max, y_max = box[0], box[1], box[2], box[3]
            #     point = (np.float32(x_min/2 + x_max / 2), np.float32(y_min/2 + y_max/2))
            #     cv2.circle(online_im, point, 5, (0, 0, 255), -1)

            frame_id += 1
            # Son End Bye Track
            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), online_im)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                print(save_path)
                if vid_path[i] != save_path:  # new video
                    print("Phan Hong Son")
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 50, online_im.shape[1], online_im.shape[0]
                    save_path = str(Path(save_path).with_suffix('.avi'))  # force *.mp4 suffix on results videos
                    # vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (w, h))
                vid_writer[i].write(online_im)

        # Debug
        totalTime = time.time() - startTime
        fps = 1 / totalTime
        print("FPS: {:.2f}".format(fps))

    # write result to json
    flag_cap = dataset.get_cap()
    print("this is flag cap: ", flag_cap)
    write_result(flag_cap)

if __name__ == "__main__":
    opt = make_parser()
    main(opt)
