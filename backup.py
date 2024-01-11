from flask import Flask, abort, request, jsonify
import cv2
import copy
import time
import random
import numpy as np
from openpose import util
from openpose.body import Body
from openpose.hand import Hand
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from ultralytics import YOLO
import numpy as np
import cv2
import math

count = 0

body_estimation = Body('model/body_pose_model.pth')

# nearest human
nearest_cfg = get_cfg()
nearest_cfg.MODEL.DEVICE = "cuda"
nearest_cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
nearest_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
nearest_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
nearest_predictor = DefaultPredictor(nearest_cfg)

#segment
segment_cfg = get_cfg()
segment_cfg.MODEL.DEVICE = "cuda"
segment_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
segment_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
segment_predictor = DefaultPredictor(segment_cfg)

model = YOLO('model/yolov8n-pose.pt')

def analyse_image(img, draw = True):
    candidate, subset = body_estimation(img)
    canvas = copy.deepcopy(img)
    canvas, body_angles = util.draw_bodypose(canvas, candidate, subset, draw)
    # Offset joints
    if body_angles:
        body_angles[0] = body_angles[0] - 90
        if body_angles[1] <0:
            body_angles.append(0)
            body_angles[1] = -1*body_angles[1]
        else:
            body_angles.append(100)
        if body_angles[3] <0:
            body_angles.append(0)
            body_angles[3] = -1*body_angles[3]
        else:
            body_angles.append(100)
        body_angles[1] = - (180 - body_angles[1])
        body_angles[2] = - (body_angles[2] - 90)
        body_angles[3] = 180 - body_angles[3]
        if draw:
            global count
            cv2.imwrite('test' + str(count) + '.jpg', canvas)
            count += 1
    if body_angles:
        return body_angles
    return None

def nearestHuman(image):
    outputs = nearest_predictor(image)
    instances = outputs["instances"]
    labels = instances.pred_classes
    human_label = 0
    human_instances = instances[labels == human_label]
    human_boxes = human_instances.pred_boxes
    height, width = image.shape[:2]
    human_mask = np.zeros((height, width), dtype=np.uint8)
    max_width = -1
    x1_max, x2_max, y1_max, y2_max = -1, -1, -1, -1
    for box in human_boxes:
        x1, y1, x2, y2 = map(int, box)
        if abs(x1 - x2) > max_width:
            max_width = abs(x1 - x2)
            x1_max, x2_max, y1_max, y2_max = x1, x2, y1, y2

    if x1_max == -1:
        return None
    human_mask[y1_max:y2_max, x1_max:x2_max] = 1
    image_with_humans = image.copy()
    image_with_humans[human_mask == 0] = 0
    return image_with_humans

def segmentHuman(image):
    outputs = segment_predictor(image)
    instances = outputs["instances"]

    # Get the labels for each instance
    labels = instances.pred_classes

    # Define the label for humans (usually 0 for 'person' in COCO dataset)
    human_label = 0

    # Filter the instances to keep only humans
    human_instances = instances[labels == human_label]
    human_masks = human_instances.pred_masks.cpu().numpy()

    # Create an empty mask for humans
    height, width = image.shape[:2]
    human_mask = np.zeros((height, width), dtype=np.uint8)

    # Combine human masks into a single mask
    for mask in human_masks:
        human_mask = np.logical_or(human_mask, mask)

    # Apply the human mask to the original image
    image_with_humans = image.copy()
    image_with_humans[human_mask == 0] = 0

    return image_with_humans

def pose_est(image):
    results = model(image) 
    print(results)

def get_angle_in_radians(angle):
    if angle is not None:
        body_angle_in_radians = [math.radians(x) for x in angle[:4]]
        # pitch is the rotation around the shoulder socket
        direction = [] # -119.5 to 119.5
        direction.append(math.radians(0)) if int(angle[-2]) > 50 else direction.append(math.radians(180))
        direction.append(math.radians(0)) if int(angle[-1]) > 50 else direction.append(math.radians(180))
        body_angle_in_radians = direction + body_angle_in_radians
        if body_angle_in_radians[0] < -2.0857: body_angle_in_radians[0] = -2.0857
        if body_angle_in_radians[0] > 2.0857: body_angle_in_radians[0] = 2.0857
        if body_angle_in_radians[1] < -2.0857: body_angle_in_radians[1] = -2.0857
        if body_angle_in_radians[1] > 2.0857: body_angle_in_radians[1] = 2.0857
        if body_angle_in_radians[2] < 0.0087: body_angle_in_radians[2] = 0.0087
        if body_angle_in_radians[2] > 1.5620: body_angle_in_radians[2] = 1.5620
        if body_angle_in_radians[3] < -1.5620: body_angle_in_radians[3] = -1.5620
        if body_angle_in_radians[3] > -0.0087: body_angle_in_radians[3] = -0.0087
        if body_angle_in_radians[4] < -1.5620: body_angle_in_radians[4] = -1.5620
        if body_angle_in_radians[4] > -0.0087: body_angle_in_radians[4] = -0.0087
        if body_angle_in_radians[5] < 0.0087: body_angle_in_radians[5] = 0.0087
        if body_angle_in_radians[5] > 1.5620: body_angle_in_radians[5] = 1.5620
        return body_angle_in_radians
    return None

app = Flask(__name__)
@app.route("/imitate", methods=['POST'])
async def imitate():
    if not request.files:
        abort(400)

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    image = nearestHuman(image)
    if image is None:
        return {'angles': None}

    image = segmentHuman(image)

    angles = analyse_image(image)
    print(angles)
    angles = get_angle_in_radians(angles)
    data = {'angles': angles}

    return data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)
