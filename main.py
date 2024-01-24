# from flask import Flask, abort, request, jsonify
import cv2
import copy
import time
import random
import numpy as np
from ultralytics import YOLO
import numpy as np
import cv2
import math
import base64
import json
import angle_detection
from tempfile import NamedTemporaryFile
import pika
from collections import defaultdict

count = 0
model = YOLO('model/yolov8n-pose.pt')

params = pika.URLParameters('amqp://zftppdhz:i4bn6ElyHC-AGgswO3czf3pulF6hpOjy@albatross.rmq.cloudamqp.com/zftppdhz')
params.socket_timeout = 5

connection = pika.BlockingConnection(params)
rabbit_channel = connection.channel()
rabbit_channel.queue_declare(queue='image_queue')

def offset_angles(body_angles):
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
    return body_angles

def pose_keypoints(image):
    max_width = -1
    human_box = None
    index = -1
    height, width = image.shape[:2]
    results = model(image)
    human_mask = np.zeros((height, width), dtype=np.uint8)
    for i, box in enumerate(results[0].boxes.xywh):
        if max_width < box[2]:
            max_width = box[2]
            human_box = box
            index = i
    if human_box is None:
        return None
    return(results[0].keypoints[index].xy[0])

def get_angle_in_radians(angle, leg_angles):
    if angle is not None:
        body_angle_in_radians = [math.radians(x) for x in angle[:4]]
        for i, el in enumerate(body_angle_in_radians):
            if math.isnan(el):
                body_angle_in_radians[i] = 0
        # pitch is the rotation around the shoulder socket
        direction = [] # -119.5 to 119.5
        direction.append(math.radians(0)) if int(angle[-2]) > 50 else direction.append(math.radians(180))
        direction.append(math.radians(0)) if int(angle[-1]) > 50 else direction.append(math.radians(180))
        body_angle_in_radians = direction + body_angle_in_radians
        if body_angle_in_radians[0] < -2.0857: body_angle_in_radians[0] = -2.0857
        if body_angle_in_radians[0] > 1.8: body_angle_in_radians[0] = 1.8
        if body_angle_in_radians[1] < -2.0857: body_angle_in_radians[1] = -2.0857
        if body_angle_in_radians[1] > 1.8: body_angle_in_radians[1] = 1.8
        if body_angle_in_radians[2] < 0.01: body_angle_in_radians[2] = 0.0087
        if body_angle_in_radians[2] > 1.5620: body_angle_in_radians[2] = 1.5620
        if body_angle_in_radians[3] < -1.5620: body_angle_in_radians[3] = -1.5620
        if body_angle_in_radians[3] > -0.0087: body_angle_in_radians[3] = -0.0087
        if body_angle_in_radians[4] < -1.5620: body_angle_in_radians[4] = -1.5620
        if body_angle_in_radians[4] > -0.0087: body_angle_in_radians[4] = -0.0087
        if body_angle_in_radians[5] < 0.0087: body_angle_in_radians[5] = 0.0087
        if body_angle_in_radians[5] > 1.5620: body_angle_in_radians[5] = 1.5620
        for angle in leg_angles:
            body_angle_in_radians.append(angle)
        return body_angle_in_radians
    return None

# app = Flask(__name__)
# @app.route("/imitate", methods=['POST'])
# async def imitate():
#     if not request.files:
#         abort(400)

#     image_file = request.files['image']
#     image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

#     keypoints = pose_keypoints(image)
#     if keypoints is None:
#         _, buffer = cv2.imencode('.jpg', image)
#         img_str = base64.b64encode(buffer).decode('utf-8')
#         return {'angles': None, 'image': img_str}
#     # print(keypoints)
#     keypoints = keypoints.cpu()
#     print(keypoints)
#     body_angles = angle_detection.get_body_angles(keypoints)
#     leg_angles = angle_detection.get_leg_angles(keypoints)
#     # print(leg_angles)
#     # print(body_angles)
#     body_angles = offset_angles(body_angles)
#     angles = get_angle_in_radians(body_angles, leg_angles)
#     print(angles)

#     annotated_image = image.copy()
#     if angles:
#         # Draw keypoints
#         for x, y in keypoints:
#             cv2.circle(annotated_image, (int(x), int(y)), 5, (0, 0, 255), -1)

#         # Draw angles
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         for i, angle in enumerate(angles):
#             angle_text = f'Angle {i + 1}: {math.degrees(angle):.2f} degrees'
#             cv2.putText(annotated_image, angle_text, (10, 30 * (i + 1)), font, 0.7, (0, 255, 0), 2)

#         # Define the path to save the annotated image
#         global count
#         save_path = "annotated_image" + str(count) + ".jpg"
#         count += 1

#         # Save the annotated image to the specified path
#         cv2.imwrite(save_path, annotated_image)
#     _, buffer = cv2.imencode('.jpg', annotated_image)
#     img_str = base64.b64encode(buffer).decode('utf-8')
#     data = {'angles': angles, 'image': img_str}

#     return data

def imitate_callback(ch, method, properties, body):
    body = json.loads(body)
    image = cv2.imdecode(np.frombuffer(base64.b64decode(body["image"]), np.uint8), cv2.IMREAD_UNCHANGED)

    keypoints = pose_keypoints(image)
    if keypoints is None:
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return {'angles': None, 'image': img_str}
    # print(keypoints)
    keypoints = keypoints.cpu()
    # print(keypoints)
    body_angles = angle_detection.get_body_angles(keypoints)
    leg_angles = angle_detection.get_leg_angles(keypoints)
    # # print(leg_angles)
    # # print(body_angles)
    body_angles = offset_angles(body_angles)
    angles = get_angle_in_radians(body_angles, leg_angles)
    # print(angles)

    annotated_image = image.copy()
    if angles:
        for x, y in keypoints:
            cv2.circle(annotated_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # for i, angle in enumerate(angles):
        #     angle_text = f'Angle {i + 1}: {math.degrees(angle):.2f} degrees'
        #     cv2.putText(annotated_image, angle_text, (10, 30 * (i + 1)), font, 0.7, (0, 255, 0), 2)
        # global count
        # save_path = "annotated_image" + str(count) + ".jpg"
        # count += 1
        # cv2.imwrite(save_path, annotated_image)

    _, buffer = cv2.imencode('.jpg', annotated_image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    data = {"angles": angles, "image": img_str}
    rabbit_channel.basic_publish(exchange='', routing_key='angle_queue', body=json.dumps(data))
    print(angles)

rabbit_channel.basic_consume('image_queue',
  imitate_callback,
  auto_ack=True)
rabbit_channel.start_consuming()
connection.close()

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5006)
