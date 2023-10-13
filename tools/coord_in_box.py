import numpy as np

def coordinates_in_box(box,coord):     #box & values of ith-person dict
	count=0
	total=len(coord)
	if total>0:
		for i in coord:
			# print(box,i)
			if i[0]==-1 and i[1]==-1:
				total-=1
				continue
			
			if box[0]<i[0]<box[2] and box[1]<i[1]<box[3]:
				count+=1
		if count/total>0.7:
			return True
		
	return False

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (list): [x1, y1, x2, y2] format representing the first bounding box.
        box2 (list): [x1, y1, x2, y2] format representing the second bounding box.

    Returns:
        float: IoU value between 0 and 1.
    """
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def bbox_to_fig_ratio(bbox,coord):
    coord=np.array(coord)
    fig_max_y=coord.max(axis=0)[1]
    fig_min_y=coord.min(axis=0)[1]
    dst_fig=fig_max_y-fig_min_y
    
    bbox_max_y=bbox[3]
    bbox_min_y=bbox[1]
    dst_bbox=bbox_max_y-bbox_min_y
    
    if dst_fig/dst_bbox>0.7:
        return True
    
    return False
    