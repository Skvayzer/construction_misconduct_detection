from deepsort import nn_matching
from deepsort.detection import Detection
from deepsort.tracker import Tracker


from tools import generate_detections as gdet
from tools.utils import find_closest_color, crop_minAreaRect, angle_between_vectors, iscc_nbs_color_dict


from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS as POSE_VISUALIZERS
from mmpose.structures import merge_data_samples

import cv2
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS as DET_VISUALIZERS

# from mmdet.apis import init_detector, inference_detector
import numpy as np 
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline

from mmdet.apis import inference_detector, init_detector

from collections import Counter
from tools.processing import *
from ultralytics import YOLO
import pandas as pd
from config import VideoProcessingConfig


import os
print(os.getcwd())

config = VideoProcessingConfig()



# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# Deep SORT
encoder = gdet.create_box_encoder(config.deepsort_model, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

# Initializing the tracker with given metrics.
tracker = Tracker(metric)


uniform_yolo = YOLO(config.yolo_path)


device = config.device

# init model
pose_model = init_model(config.pose_model_cfg, config.pose_ckpt, device=device)


# build the visualizer
pose_visualizer = POSE_VISUALIZERS.build(pose_model.cfg.visualizer)

# # set skeleton, colormap and joint connection rule
pose_visualizer.set_dataset_meta(pose_model.dataset_meta)





# Build the model from a config file and a checkpoint file
detection_model = init_detector(config.detection_config_file, config.detection_checkpoint_file, device=device)
detection_model.cfg = adapt_mmdet_pipeline(detection_model.cfg)

# Init visualizer
det_visualizer_cfg = detection_model.cfg.visualizer
det_visualizer_cfg['alpha'] = 0.1
det_visualizer = DET_VISUALIZERS.build(det_visualizer_cfg)
# print(detection_model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
meta = detection_model.dataset_meta
meta['classes'] = meta['classes'][:1]
meta['palette'] = meta['palette'][:1]

det_visualizer.dataset_meta = detection_model.dataset_meta



def process_frame(frame, frame_idx):
    # if frame_idx < 250:
    #     continue

    # if frame_idx % 5 !=0:
    #     continue
    # frame = cv2.imread('Thingy-Detector/people_lying4.jpg', 1) 

    # Update the video time (elapsed time in seconds)
    config.frame_count += 1
    video_time = frame_idx / config.frame_rate



    
    # vertices_polygon = np.array(select_polygon_points(frame)) #np.array([[600,0], [600,600], [1200,1200], [1200,0]]) # manually adjusted

    
    draw_frame = frame
    
    det_result = inference_detector(detection_model, frame)
    pred_instance = det_result.pred_instances.cpu().numpy()

    indexes = np.logical_and(pred_instance.labels == config.det_cat_id,
                                    pred_instance.scores > config.bbox_thr)
    filtered_indexes = np.logical_not(indexes)
    # boxes = det_result.pred_instances.bboxes[indexes]
    # det_result.pred_instances.bboxes = boxes
    # det_result.pred_instances.labels = det_result.pred_instances.labels[indexes]
    det_result.pred_instances.scores[filtered_indexes] = 0


    # Initialize an empty dictionary to store the mapping
    mapping_dict = {}

    # Iterate through the boolean array and track the ordinal number of True elements
    ordinal_number = 0
    for index, value in enumerate(indexes):
        if value:
            mapping_dict[ordinal_number] = index
            ordinal_number += 1

    boxes = pred_instance.bboxes[indexes]
    labels = pred_instance.labels[indexes]
    scores = pred_instance.scores[indexes]

    for i, box in enumerate(boxes):
        x=int(box[0])
        y=int(box[1])
        w=int(box[2]-x)
        h=int(box[3]-y)
        boxes[i] = [x,y,w,h]



    features = encoder(frame, boxes)


    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    
    
    bboxes = bboxes[indexes]
    bboxes = bboxes[nms(bboxes, config.nms_thr), :4]

    found_people = bboxes.shape[0] != 0

    if found_people:
        # predict keypoints
        pose_results = inference_topdown(pose_model, frame, bboxes)
        data_samples = merge_data_samples(pose_results)
        person_dict = convert_to_dict(data_samples)

        


    # score to 1.0 here.
    detections = [Detection(bbox, score, {'class': label, 'score': score}, feature) for bbox, feature, score, label in zip(boxes, features, scores, labels)]
    # Run non-maxima suppression.
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)

    

    det_result_copy = det_result.clone()
    det_indexes = np.array([False] * len(det_result_copy.pred_instances.labels))

    

    if found_people:
        data_samples_copy = data_samples.clone()
        pose_indexes = np.array([False] * len(data_samples.pred_instances.bbox_scores))

        visible_person_dict = {}
        for person, value in person_dict.items():
            visible_person_dict[person] = {part: coords for part, coords in value.items() if np.any(coords != -1)}

        spines = {}
        for person in person_dict.keys():
            spines[person] = [(-1, -1), (-1, -1)]
            nose, left_hip, right_hip = person_dict[person]['nose'], person_dict[person]['left_hip'], person_dict[person]['right_hip']
            if np.any(nose == -1) or np.any(left_hip == -1) and np.any(right_hip == -1):
                continue
            if np.all(left_hip != -1) and np.all(right_hip != -1):
                bottom = (left_hip + right_hip) / 2
            else:
                bottom = left_hip + right_hip + [1, 1]
            
            spines[person] = [nose, bottom]
        
        legs = {}
        for person in person_dict.keys():
            legs[person] = [-1, -1]
            left_hip, left_knee, right_hip, right_knee = person_dict[person]['left_hip'], person_dict[person]['left_knee'], person_dict[person]['right_hip'], person_dict[person]['right_knee']
            left_leg, right_leg = None, None
            if np.all(left_hip != -1) and np.all(left_knee != -1):
                left_leg = left_hip - left_knee
            if np.all(right_hip != -1) and np.all(right_knee != -1):
                right_leg = right_hip - right_knee

            visible_medium_leg = None
            if left_leg is not None and right_leg is not None:
                legs[person] = (left_leg + right_leg)/2
            elif left_leg is None and right_leg is not None:
                legs[person] = right_leg
            elif left_leg is not None and right_leg is None:
                legs[person] = left_leg




        spine_leg_angles = {}
        poses = {}
        wears_uniform = {}
        for person_idx, person in enumerate(person_dict.keys()):

            person_width = bboxes[person_idx][2] - bboxes[person_idx][0]
            person_height = bboxes[person_idx][3] - bboxes[person_idx][1]

            wears_uniform[person] = '?'
            if -1 in spines[person][0] or -1 in spines[person][1]:
                continue

            spine, leg = spines[person][0] - spines[person][1], legs[person]

            ground_vector = np.array([1, 0])

            spine_legs_angle = angle_between_vectors(spine, leg)[0]

            spine_ground_angle = angle_between_vectors(spine, ground_vector)[0]

            leg_ground_angle = angle_between_vectors(leg, ground_vector)[0]

            spine_leg_angles[person] = spine_legs_angle

            leg_delta = 30

            if 90 - leg_delta <= leg_ground_angle <= 90 + leg_delta:
                if person_width > person_height:
                    poses[person] = 'лежит'
                else:
                    poses[person] = 'стоит'
                # delta = 45
                # if spine_legs_angle >= 90 + delta:
                #     if spine_ground_angle > 45:
                #         poses[person] = 'стоит'
                #     else:
                #         poses[person] = 'сидит'
                
                # else:
                #     poses[person] = 'стоит'
            elif spine_ground_angle <= 30 or spine_ground_angle >= 150:
                poses[person] = 'лежит'
            else:
                poses[person] = 'сидит'

            perpendicular_vector = np.array([-spine[1], spine[0]])
            norm_perpendicular = perpendicular_vector / np.linalg.norm(perpendicular_vector)
            
            
            spine_coords = spines[person]
            shoulder = person_dict[person]['right_shoulder'] if 'right_shoulder' in visible_person_dict.keys() else person_dict[person]['left_shoulder']
            shoulder_length = np.linalg.norm(person_dict[person]['nose'] - shoulder)
            width = norm_perpendicular
            width = width * shoulder_length
            if np.linalg.norm(width) < 5:
                continue

            rect = [spine_coords[0] - width, spine_coords[0] + width, spine_coords[1] + width, spine_coords[1] - width]


            clothing = crop_minAreaRect(frame, rect)
            # cv2.imwrite('cloth.jpg', clothing)
            # mmcv.imshow(clothing, 'video', wait_time)
            clothing = clothing.astype("float32") / 255
            lab_image = cv2.cvtColor(clothing, cv2.COLOR_BGR2LAB)

            # Reshape the Lab image to a 2D array of pixels
            lab_pixels = lab_image.reshape((-1, 3))

            # Convert the Lab pixel values to float32
            lab_pixels = np.float32(lab_pixels)

            # Define the criteria and apply k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            k = 3  # Number of clusters (colors)
            # print('K MEANS CALC')
            _, labels, centers = cv2.kmeans(lab_pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # print('K MEANS DONE')
            # Convert the Lab centers to integer values
            # centers = np.uint8(centers)

            # Count the occurrences of each label
            label_counts = Counter(labels.flatten())

            # Get the top 3 most common colors
            top_colors = [centers[i] for i, _ in label_counts.most_common(7)]

            # similarity = compare_color_histograms(clothing, uniform_reference)
            closest_colors = [find_closest_color(color) for color in top_colors]
            closest_base_colors = [color.split('_')[-1] for color in closest_colors]
            if 'Orange' in closest_base_colors or 'Green' in closest_base_colors or 'Yellow' in closest_base_colors:
                wears_uniform[person] = 1 #'Одет'
            else:
                wears_uniform[person] = 0 #'Не одет'

                            
            # max_iou = 0
            # matched_track_id = -1
            # for track_id, track in enumerate(tracker.tracks):
            #     iou = get_iou(track.to_tlbr(), bboxes[person])
            #     if iou > max_iou:
            #         matched_track_id = track_id
            #         max_iou = iou
            # if matched_track_id != -1:
            #     track = tracker.tracks[matched_track_id]
            #     track.filters['vest'].add_prediction(wears_uniform[person])
            #     wears_uniform[person] = track.filters['vest'].get_filtered_prediction()

    # print(tracker.tracks)

    draw_frame = draw_polygon_area(draw_frame, config.vertices_polygon)


    people_in_red_zone, people_out_red_zone, total_people = 0, 0, 0
    
    if len(person_dict) > 0:
        count_people_in_red_zone(person_dict, config.vertices_polygon)
        total_people = len(person_dict.keys())

    for track in tracker.tracks:
        # print('TRACK', track)
        if not track.is_confirmed() or track.time_since_update > 5:
            continue
        bbox = track.to_tlbr()
        class_id = int(track.get_data()['class'])
        class_name = detection_model.dataset_meta['classes'][class_id]
        score = track.get_data()['score']


        max_iou = 0
        matched_detection_idx = -1
        for idx, detection in enumerate(detections):
            iou = get_iou(detection.to_tlbr(), bbox)
            if detection.get_data()['class'] == config.det_cat_id and iou > max_iou:
                matched_detection_idx = idx
                if bboxes.shape[0] != 0 and idx < len(pose_indexes):
                    pose_indexes[idx] = True
                max_iou = iou
        
        if matched_detection_idx != -1:
            indexes[mapping_dict[matched_detection_idx]] = True



    det_result_copy.pred_instances.scores[np.logical_not(indexes)] = 0


    det_visualizer.add_datasample(
        name='video',
        image=draw_frame,
        data_sample=det_result_copy,
        draw_gt=False,
        show=False)
    draw_frame = det_visualizer.get_image()




    if found_people:
        data_samples_copy.pred_instances.bbox_scores[np.logical_not(pose_indexes)] = 0

        pose_visualizer.add_datasample(
            name='video',
            image=draw_frame,
            data_sample=data_samples_copy,
            draw_gt=False,
            show=False)
        draw_frame = pose_visualizer.get_image()

        misconduct_detected = False
        for i, bbox in enumerate(bboxes):
            #padding
            padding = 5
            person_box = {
                'ymin': int(bbox[1]) - padding if int(bbox[1]) - padding >= 0 else int(bbox[1]), 
                'ymax': int(bbox[3]) + padding if int(bbox[3]) + padding < config.video_reader.width else int(bbox[3]), 
                'xmin': int(bbox[0]) - padding if int(bbox[0]) - padding >= 0 else int(bbox[0]), 
                'xmax': int(bbox[2]) + padding if int(bbox[2]) + padding < config.video_reader.height else int(bbox[2])
                }
            
            person_width = bbox[2] - bbox[0]
            person_height = bbox[3] - bbox[1]
            person_area = person_width * person_height
            

            original_person = frame[person_box['ymin']:person_box['ymax'], person_box['xmin']:person_box['xmax']]
            draw_person = draw_frame[person_box['ymin']:person_box['ymax'], person_box['xmin']:person_box['xmax']]
            findings = uniform_yolo(original_person)
            
            
            predictions = findings[0].boxes.data
            predictions = pd.DataFrame(predictions.cpu().numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'])            

            hat_boxes = predictions[predictions['class'].isin([0, 2])].reset_index()
            vest_boxes = predictions[predictions['class'].isin([4, 7])].reset_index()

            vest_boxes = vest_boxes.replace({'class': {4: 0, 7: 1}})
            hat_boxes = hat_boxes.replace({'class': {0: 1, 2: 0}})

            # print("BOXES", boxs)
            # hat_boxes = predictions[:, :4]
            # print(hat_boxes)
            
            # scores = predictions[:, 4]
            # categories = predictions[:, 5]

            max_iou = 0
            matched_track_id = -1
            for track_id, track in enumerate(tracker.tracks):
                iou = get_iou(track.to_tlbr(), bbox)
                if iou > max_iou:
                    matched_track_id = track_id
                    max_iou = iou

            if len(vest_boxes) > 0:
                vest_box = vest_boxes.iloc[0]
                if vest_box['class'] == 1:
                    wears_uniform[i] = 1
                    
                vest_coords =  ((int(vest_box['xmin']), int(vest_box['ymin'])), (int(vest_box['xmax']), int(vest_box['ymax'])))



                if matched_track_id != -1:
                    track = tracker.tracks[matched_track_id]
                    wears_uniform[i] = int(wears_uniform[i]) if type(wears_uniform[i]) is not str else wears_uniform[i]
                    track.filters['vest'].add_prediction(wears_uniform[i])
                    print('vest')
                    wears_uniform[i] = track.filters['vest'].get_filtered_prediction()

                    color = (0, 128, 255) if wears_uniform[i] == 1 else (255, 128, 0)
                    cv2.rectangle(draw_person, *vest_coords, color, 2)

            hat_detected = False
            if len(hat_boxes) > 0:
                # print(hat_boxes['ymin'].idxmin())
                hat_box = hat_boxes.iloc[hat_boxes['ymin'].idxmin()]
                print('\n\n\n', hat_box, '\n\n\n')
                coords = ((int(hat_box['xmin']), int(hat_box['ymin'])), (int(hat_box['xmax']), int(hat_box['ymax'])))

                if matched_track_id != -1:
                    track = tracker.tracks[matched_track_id]
                    track.filters['hat'].add_prediction(int(hat_box["class"]))
                    print('hat')
                    filtered_hat_class = track.filters['hat'].get_filtered_prediction()

                    # track.filters['vest'].add_prediction(int(wears_uniform[i]))
                    # print('vest')
                    # wears_uniform[i] = track.filters['vest'].get_filtered_prediction()

                else:
                    filtered_hat_class = int(hat_box["class"])

                # # MATCHING WITH PEOPLE
                # for track in tracker.tracks:
                #     # print('TRACK', track)
                #     if not track.is_confirmed() or track.time_since_update > 5:
                #         continue
                #     person_box = track.to_tlbr()
                #     class_id = int(track.get_data()['class'])
                #     class_name = detection_model.dataset_meta['classes'][class_id]
                #     score = track.get_data()['score']

                #     if get_iou(hat_box, person_box) > 0.5:
                #         matched_id = track.track_id
                                        
                if filtered_hat_class:
                    hat_detected = True
                    box_color = (0, 255, 0)
                    text_color = (255, 255, 255)
                    class_name='Каска'
                    confidence = round(hat_box["confidence"], 2)
                else:
                    box_color = (0, 0, 255)
                    text_color = (180, 180, 255)
                    class_name='Нет Каски'
                    confidence = ''
                cv2.rectangle(draw_person, *coords, box_color, 2)
                # mmcv.imshow(draw_person, 'video', 3*60)
                # mmcv.imshow(original_person, 'video', 3*60)

                # text_coords = ((int(hat_box['xmin'] + person_box['xmin']), int(hat_box['ymin'] + person_box['ymin'])), (int(hat_box['xmax'] + person_box['xmin']), int(hat_box['ymax'] + person_box['ymin'])))
                # cv2.rectangle(draw_frame, *coords, color, 2)
                # Annotate with class name
                # text_coords = (person_box['ymin'] - 10, person_box['xmin'])
                text_coords = (int(hat_box['xmin'] + person_box['xmin']) - 10, int(hat_box['ymax'] + person_box['ymin']))

                # cv2.circle(draw_frame, text_coords, 5, color, 5)
                
                cv2.putText(draw_frame, f'{class_name}', text_coords, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(draw_frame, f'{class_name}', text_coords, cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 1, cv2.LINE_AA)
            else:
                box_color = (0, 0, 255)
                text_color = (180, 180, 255)
                keypoints = data_samples.pred_instances.keypoints[i]
                # cv2.circle(draw_frame, (int(keypoints[0][0]), int(keypoints[0][1])), 5, box_color, 5)
                # coords = (int(keypoints[0][0]), int(keypoints[0][1]))
                cv2.circle(draw_frame, (int(keypoints[0][0]), int(keypoints[0][1])), 5, box_color, 1)

                cv2.putText(draw_frame, f'{"Нет Каски"}', (int(keypoints[0][0]), int(keypoints[0][1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(draw_frame, f'{"Нет Каски"}', (int(keypoints[0][0]), int(keypoints[0][1]) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 1, cv2.LINE_AA)

            body_coords = (person_box['xmax'], person_box['ymin'] + int(0.5*(person_box['ymax'] - person_box['ymin'])))
            dress = 'Одет' if wears_uniform[i] else 'Не одет'
            cv2.putText(draw_frame, f'{dress}', body_coords, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(draw_frame, f'{dress}', body_coords, cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            pose_coords = (person_box['xmax'] + 10, person_box['ymax'] + 10)
            cv2.putText(draw_frame, f'{poses[i]}', pose_coords, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(draw_frame, f'{poses[i]}', pose_coords, cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            if person_area > config.min_person_area and (not wears_uniform[i] or poses[i] in ['сидит, лежит'] or not hat_detected): #or people_in_red_zone > 0 :
                misconduct_detected = True

        if misconduct_detected:
            config.misconduct_count +=1
        cv2.putText(draw_frame, f'В зоне: {people_in_red_zone}/{total_people}', (30, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)


    # Check if a 5-minute interval has passed
    if video_time - config.start_time >= config.time_period:
        # Determine if this interval is safe or unsafe
        misconduct_detection_ratio = config.misconduct_count / config.frame_count
        if misconduct_detection_ratio > config.safe_threshold or misconduct_count > config.unsafe_count_threshold:
            config.safety_intervals.append((start_time, video_time, 0))
        else:
            config.safety_intervals.append((start_time, video_time, 1))

        # Reset variables for the next interval
        misconduct_count = 0
        start_time = video_time
        config.frame_count = 0



    return draw_frame

def save_intervals():
    # Save the results to a file
    with open('safety_results.txt', 'w') as file:
        for interval in config.safety_intervals:
            file.write(f'{interval[0]} - {interval[1]} : {interval[2]}\n')

