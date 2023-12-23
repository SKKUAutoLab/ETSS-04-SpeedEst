"""
Copyright  by PHAN HONG SON
Univ: SungKyunKwan University
"""
from config_pose_for_speed.datasets.builder import DATASETS
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
import numpy as np
from mmcv import Config, deprecated_api_warning
import json
from collections import OrderedDict
import tempfile

from config_pose_for_speed.core.evaluation.top_down_eval import (keypoint_nme, keypoint_pck_accuracy)
from config_pose_for_speed.datasets.builder import DATASETS
from config_pose_for_speed.datasets.datasets.base import Kpt2dSviewRgbImgTopDownDataset
# from configs._base_.datasets.custom import dataset_info
# DEBUG
#import pose evaluation
from config_pose_for_speed.Pose_evaluation import pose_evaluation

@DATASETS.register_module()
class CarCorner(Kpt2dSviewRgbImgTopDownDataset):
    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. ', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/custom.py')
            dataset_info = cfg._cfg_dict['dataset_info']
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            coco_style=True,
            test_mode=test_mode)

        # flip_pairs, upper_body_ids and lower_body_ids will be used
        # in some data augmentations like random flip

        # self.ann_info['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
        #                                [11, 12], [13, 14], [15, 16]]
        # self.ann_info['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        # self.ann_info['lower_body_ids'] = (11, 12, 13, 14, 15, 16)
        #
        # self.ann_info['joint_weights'] = None
        # self.ann_info['use_different_joint_weights'] = False

        self.dataset_name = 'CarCorner'

        # print('======> Print  link image: ', self.img_prefix)
        self.db = self._get_db()
        # print(f'=> num_images: {self.num_images}')
        # print(f'=> load {len(self.db)} samples')

    # Load dataset
    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        num_joints = self.ann_info['num_joints']
        # print('=======> Son print num_joint: ', num_joints)
        for img_id in self.img_ids:
            # print('====> Son print image list ids: ', self.img_ids)
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False) # ID of annotation
            # print('====> Son print annotations ids: ', ann_ids)
            objs = self.coco.loadAnns(ann_ids) #annotation field
            # print('====> Son print objs ids: ', objs)

            for obj in objs:
                if True:
                    joints_3d = np.zeros((num_joints, 3), dtype=np.float32)   # joints_3d = corner for training
                    joints_3d_visible = np.zeros((num_joints, 3),
                                                 dtype=np.float32)
                    keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                    # print('====> Son print keypoint: ', keypoints)
                    joints_3d[:, :2] = keypoints[:, :2]
                    joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

                    image_file = osp.join(self.img_prefix,
                                          self.id2name[img_id])
                    gt_db.append({
                        'image_file': image_file,
                        'rotation': 0,
                        'joints_3d': joints_3d,
                        'joints_3d_visible': joints_3d,
                        'dataset': self.dataset_name,
                        'bbox': obj['bbox'],
                        'bbox_score': 1,
                        'bbox_id': bbox_id
                    })

                    # print('====> Son print a gt_db: ', gt_db)
                    bbox_id = bbox_id + 1

                    # print('====> Son print pass here <=============: ', bbox_id)
        # print('====> Son print gt_db: ', gt_db)
        gt_db = sorted(gt_db, key=lambda x: x['bbox_id'])
        # print('====> Son print gt_db: ', gt_db)

        return gt_db

    def evaluate(self, results, res_folder=None, metric='PCK', **kwargs):
        """Evaluate keypoint detection results. The pose prediction results will
        be saved in `${res_folder}/result_keypoints.json`.

        Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

        Args:
        results (list(preds, boxes, image_path, output_heatmap))
            :preds (np.ndarray[N,K,3]): The first two dimensions are
                coordinates, score is the third dimension of the array.
            :boxes (np.ndarray[N,6]): [center[0], center[1], scale[0]
                , scale[1],area, score]
            :image_paths (list[str]): For example, ['Test/source/0.jpg']
            :output_heatmap (np.ndarray[N, K, H, W]): model outputs.

        res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
        metric (str | list[str]): Metric to be performed.
            Options: 'PCK', 'NME'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['PCK', 'NME']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        kpts = []
        for result in results:
            preds = result['preds']
            boxes = result['boxes']
            image_paths = result['image_paths']
            bbox_ids = result['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'bbox_id': bbox_ids[i]
                })
        kpts = self._sort_and_unique_bboxes(kpts)

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file, metrics)
        name_value = OrderedDict(info_str)

        if tmp_folder is not None:
            tmp_folder.cleanup()

        return name_value

    def _report_metric(self, res_file, metrics, pck_thr=0.3):
        """Keypoint evaluation.

        Args:
        res_file (str): Json file stored prediction results.
        metrics (str | list[str]): Metric to be performed.
            Options: 'PCK', 'NME'.
        pck_thr (float): PCK threshold, default: 0.3.

        Returns:
        dict: Evaluation results for evaluation metric.
        """
        info_str = []

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        outputs = []
        gts = []
        masks = []

        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :-1])
            gts.append(np.array(item['joints_3d'])[:, :-1])
            masks.append((np.array(item['joints_3d_visible'])[:, 0]) > 0)

        outputs = np.array(outputs)
        # print("this is outputs: ", outputs)
        gts = np.array(gts)
        #DEBUG
        pose_eval = pose_evaluation(outputs, gts)
        print("===========> This is result in pose evaluation: ", pose_eval)
        masks = np.array(masks)

        normalize_factor = self._get_normalize_factor(gts)

        if 'PCK' in metrics:
            _, pck, _ = keypoint_pck_accuracy(outputs, gts, masks, pck_thr,
                                              normalize_factor)
            info_str.append(('PCK', pck))

        if 'NME' in metrics:
            info_str.append(
                ('NME', keypoint_nme(outputs, gts, masks, normalize_factor)))

        return info_str

    @staticmethod
    def _write_keypoint_results(keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    @staticmethod
    def _sort_and_unique_bboxes(kpts, key='bbox_id'):
        """sort kpts and remove the repeated ones."""
        kpts = sorted(kpts, key=lambda x: x[key])
        num = len(kpts)
        for i in range(num - 1, 0, -1):
            if kpts[i][key] == kpts[i - 1][key]:
                del kpts[i]

        return kpts

    @staticmethod
    def _get_normalize_factor(gts):
        # print("_get_normalize_factor ==> Gts: ",gts)
        """Get inter-ocular distance as the normalize factor, measured as the
        Euclidean distance between the outer corners of the eyes.

        Args:
            gts (np.ndarray[N, K, 2]): Groundtruth keypoint location.

        Return:
            np.ndarray[N, 2]: normalized factor
        """
        # print("====> dim0: ", gts[:, 0, :])

        # interocular = np.linalg.norm(
        #     gts[:, 0, :] - gts[:, 1, :], axis=1, keepdims=True)
        #SON  DEBUG
        interocular = np.linalg.norm(
            gts[:, 0, :] - gts[:, 0, :], axis=1, keepdims=True)
        return  gts[:, 0, :]
        # return np.tile(interocular, [1, 2])

