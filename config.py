import mmcv
import cv2
import numpy as np


video_path = 'test.mp4'
video_reader = mmcv.VideoReader(video_path)
video_reader = video_reader


video_path = video_path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
bbox_thr = 0.3
det_cat_id = 0
nms_thr = 0.3
detected_people = {}
vertices_polygon = []
frame_rate = video_reader.fps  # Update with the actual FPS
time_period = 1
frame_interval = frame_rate * time_period  # Update with the calculated frame interval
safe_threshold = 0.8
unsafe_count_threshold = frame_interval * 0.2  # Update with the calculated threshold
start_time = 0
frame_count = 0
misconduct_count = 0
min_person_area = 10000
safety_intervals = []
device = 'cpu'
deepsort_model = 'weights/mars-small128.pb'
yolo_path = 'weights/best_yolo8_100epochs.pt'
pose_model_cfg = '../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
pose_ckpt = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'
detection_config_file = '../mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_1x_coco.py'
detection_checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_1x_coco/mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.38__segm_mAP-0.344_20200504_231812-0ebd1859.pth'


# The interval of show (ms), 0 is block
wait_time = 1
vertices_polygon = np.array([[611, 241],
                            [573, 555],
                            [1266, 596],
                            [1101, 214]]) # manually adjusted
        