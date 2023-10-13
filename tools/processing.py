
import numpy as np
import cv2


COCO_BODY_PARTS=["nose", "left_eye", "right_eye", "left_ear", 
                 "right_ear", "left_shoulder", "right_shoulder",
                   "left_elbow", "right_elbow", "left_wrist", 
                   "right_wrist", "left_hip", "right_hip", 
                   "left_knee", "right_knee", "left_ankle", "right_ankle"
                   ]


def convert_to_dict(pose_data_samples):
    pred_instances = pose_data_samples.pred_instances
    person_dict = {}
    for i, (bbox, bbox_score, keypoints, keypoints_scores) in enumerate(zip(pred_instances.bboxes, pred_instances.bbox_scores, 
                                                             pred_instances.keypoints, pred_instances.keypoint_scores)):
        keypoint_dict = {key: value for key, value in zip(COCO_BODY_PARTS, keypoints)}
        person_dict[i] = keypoint_dict

    return person_dict


def classify_pose(spine_vector, leg_vector):
    # Define the ground vector (horizontal).
    ground_vector = np.array([1, 0])  # Assuming a 2D space.

    # Calculate angles between vectors and the ground vector.
    angle_spine = np.degrees(np.arccos(np.dot(spine_vector, ground_vector)))
    angle_legs = np.degrees(np.arccos(np.dot(leg_vector, ground_vector)))

    # Classify the pose based on the angles.
    if angle_spine < 45 and angle_legs < 45:
        return "Standing"
    elif angle_spine > 45:
        return "Sitting"
    else:
        return "Lying"

def draw_polygon_area(image, vertices_polygon):

    cv2.polylines(image, [vertices_polygon.reshape(-1, 1, 2)], True, (0, 0, 128), 4)

    mod = image.copy()
    overlay = cv2.fillPoly(mod, pts = [vertices_polygon], color=(0, 0, 128))
    background = image.copy()

    image = cv2.addWeighted(
        src1 = background, # fisrt image
        alpha = 0.6, # first image weight
        src2 = overlay, # second image
        beta = 0.4, # second image weight
        gamma = 0.1, # scalar factor
        dst = overlay # output array shape
    )
    return image

def count_people_in_red_zone(people_skeletons, vertices=None):
    '''
    For a given frame (img) of the video, count how many people are in and out of the Red Zone.
    Check position in/out of the Red Zone only for Person class. Don't check other classes.
    '''

    people_in_red_zone = 0
    people_out_red_zone = 0

    for i, person_dict in people_skeletons.items():
        # inside = 0
        # for keypoint in person_dict.values():
        #     # Check if the foot is inside the polygon
        #     inside = cv2.pointPolygonTest(np.array([vertices]), keypoint, False)

        #     # 1: inside; 0: on border; -1: outside
        #     if inside == 1 or inside == 0:
        #         people_in_red_zone += 1
        #         break

        lower_visible_parts = list(person_dict.values())[-2:]

        # Check if the foot is inside the polygon
        inside1 = cv2.pointPolygonTest(np.array([vertices]), lower_visible_parts[0], False)
        inside2 = cv2.pointPolygonTest(np.array([vertices]), lower_visible_parts[1], False)


        # 1: inside; 0: on border; -1: outside
        if inside1 == 1 or inside2 == 1:
            people_in_red_zone += 1
        else:
            people_out_red_zone += 1


    return people_in_red_zone, people_out_red_zone



def select_polygon_points(image):
    global points
    points = []
    def mouse_callback(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow("Image", image)

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Escape key to exit
            break
        elif key == 13:  # Enter key to finish selecting points
            break

    cv2.destroyAllWindows()

    return points
            
            

    


def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou

def non_max_suppression(boxes,max_bbox_overlap,scores=None):
 
    if len(boxes)==0:
        return []

    boxes=boxes.astype(float)
    pick=[]

    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]+boxes[:,0]
    y2=boxes[:,3]+boxes[:,1]

    area=(x2-x1+1)*(y2-y1+1)
    if scores is not None:
        idxs=np.argsort(scores)
    else:
        idxs=np.argsort(y2)

    while len(idxs)>0:
        last=len(idxs)-1
        i=idxs[last]
        pick.append(i)

        xx1=np.maximum(x1[i],x1[idxs[:last]])
        yy1=np.maximum(y1[i],y1[idxs[:last]])
        xx2=np.minimum(x2[i],x2[idxs[:last]])
        yy2=np.minimum(y2[i],y2[idxs[:last]])

        w=np.maximum(0,xx2-xx1+1)
        h=np.maximum(0,yy2-yy1+1)

        overlap=(w*h)/area[idxs[:last]]

        idxs=np.delete(
            idxs,np.concatenate(
                ([last],np.where(overlap>max_bbox_overlap)[0])))

    return pick