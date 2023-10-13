import mmcv
import cv2
import numpy as np

class VideoProcessingConfig:
    def __init__(self):

        video_path = '../../../Thingy-Detector/hats1.mp4'
        video_reader = mmcv.VideoReader(video_path)
        self.video_reader = video_reader


        self.video_path = video_path
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.bbox_thr = 0.3
        self.det_cat_id = 0
        self.nms_thr = 0.3
        self.detected_people = {}
        self.vertices_polygon = []
        self.frame_rate = video_reader.fps  # Update with the actual FPS
        self.time_period = 1
        self.frame_interval = self.frame_rate * self.time_period  # Update with the calculated frame interval
        self.safe_threshold = 0.8
        self.unsafe_count_threshold = self.frame_interval * 0.2  # Update with the calculated threshold
        self.start_time = 0
        self.frame_count = 0
        self.misconduct_count = 0
        self.min_person_area = 10000
        self.safety_intervals = []
        self.device = 'cpu'
        self.deepsort_model = 'weights/mars-small128.pb'
        self.yolo_path = 'weights/best_yolo8_100epochs.pt'
        self.pose_model_cfg = '../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
        self.pose_ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'
        self.detection_config_file = '../mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_1x_coco.py'
        self.detection_checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'


        # The interval of show (ms), 0 is block
        self.wait_time = 1
        self.vertices_polygon = np.array([[611, 241],
                                        [573, 555],
                                        [1266, 596],
                                        [1101, 214]]) # manually adjusted
        