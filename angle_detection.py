import cv2, math
import numpy as np

def get_body_angles(body_peaks):
    '''Returns arm angles as a List in this order [LShoulderRoll, LElbowRoll, RShoulderRoll, RElbowRoll] '''

    angles = []
    body_edge_pairs = [
        [[5, 6], [6, 8]],
        [[6, 8], [8, 10]],
        [[6, 5], [5, 7]],
        [[5, 7], [7, 9]],
    ]

    body_peaks_as_array = np.array(body_peaks)
    for e, e2 in body_edge_pairs:
        angle = get_angle(e, e2, body_peaks_as_array)

        if angle != -1:
            mid_point = set(e).intersection(set(e2)).pop()
            x, y = body_peaks[mid_point]
            if e == [6,8]:
                if body_peaks_as_array[8][1] - body_peaks_as_array[10][1]  < 0:
                  angle = -1*angle
            if e == [5,7]:
                if body_peaks_as_array[7][1] - body_peaks_as_array[9][1]  < 0:
                  angle = -1*angle
            angles.append(angle)
        else:
            angles.append(0)

    if len(angles) < 4:
        return [0, 0, 0, 0]
    return angles

def get_angle(edg1,  edg2, peaks):
    ''' Returns angle between two edges '''

    edge1 = set(edg1)
    edge2 = set(edg2)

    if len(edge1.intersection(edge2)) == 0 or len(edge1.intersection(edge2)) == 2:
        return -1

    mid_point = edge1.intersection(edge2).pop()

    a = (edge1-edge2).pop()
    b = (edge2-edge1).pop()
    if len(peaks) > max(a, b, mid_point):
        v1 = peaks[mid_point]-peaks[a]
        v2 = peaks[mid_point]-peaks[b]

        angle = (math.degrees(np.arccos(np.dot(v1,v2)
                                        /(np.linalg.norm(v1)*np.linalg.norm(v2)))))
        return angle
    
    else:
        return -1

def get_leg_angles(body_peaks):
    # Define the leg keypoints
    if body_peaks[11][0] != 0 and body_peaks[11][1] != 0 and body_peaks[12][0] != 0 and body_peaks[12][1] != 0 and body_peaks[13][0] != 0 and body_peaks[13][1] != 0 and body_peaks[14][0] != 0 and body_peaks[14][1] != 0:
        print(body_peaks[11][1], body_peaks[13][1], body_peaks[12][1], body_peaks[14][1])
        if body_peaks[11][1] >= body_peaks[13][1] and body_peaks[12][1] >= body_peaks[14][1]:
            return [-10, -10]
    leg_keypoints = {
        "left_shoulder": [body_peaks[11][0], 0],
        "left_hip": body_peaks[11],
        "left_knee": body_peaks[15],

        "right_shoulder": [body_peaks[12][0], 0],
        "right_hip": body_peaks[12],
        "right_knee": body_peaks[16]
    }

    if leg_keypoints['left_knee'][0] == 0 and leg_keypoints['left_knee'][1] == 0:
        left_angle = 0
    else:
        p12 = math.sqrt((leg_keypoints["left_shoulder"][0] - leg_keypoints["left_hip"][0]) ** 2 + (leg_keypoints["left_shoulder"][1] - leg_keypoints["left_hip"][1]) ** 2)
        p13 = math.sqrt((leg_keypoints["left_knee"][0] - leg_keypoints["left_hip"][0]) ** 2 + (leg_keypoints["left_knee"][1] - leg_keypoints["left_hip"][1]) ** 2)
        p23 = math.sqrt((leg_keypoints["left_shoulder"][0] - leg_keypoints["left_knee"][0]) ** 2 + (leg_keypoints["left_shoulder"][1] - leg_keypoints["left_knee"][1]) ** 2)

        if p12 == 0 or p13 == 0:
            left_angle = 0
        else:
            angle = np.arccos((p12 ** 2 + p13 ** 2 - p23 ** 2)/(2*p12*p13))
            print(angle)
            if angle >= 3:
                left_angle = 0
            else:
                left_angle = 3.14159 - angle

    if leg_keypoints["right_knee"][0] == 0 and leg_keypoints['right_knee'][1] == 0:
        right_angle = 0
    else:
        p12 = math.sqrt((leg_keypoints["right_shoulder"][0] - leg_keypoints["right_hip"][0]) ** 2 + (leg_keypoints["right_shoulder"][1] - leg_keypoints["right_hip"][1]) ** 2)
        p13 = math.sqrt((leg_keypoints["right_knee"][0] - leg_keypoints["right_hip"][0]) ** 2 + (leg_keypoints["right_knee"][1] - leg_keypoints["right_hip"][1]) ** 2)
        p23 = math.sqrt((leg_keypoints["right_shoulder"][0] - leg_keypoints["right_knee"][0]) ** 2 + (leg_keypoints["right_shoulder"][1] - leg_keypoints["right_knee"][1]) ** 2)

        if p12 == 0 or p13 == 0:
            right_angle = 0
        else:
            angle = np.arccos((p12 ** 2 + p13 ** 2 - p23 ** 2)/(2*p12*p13))
            print(angle)
            if angle >= 3:
                right_angle = 0
            else:
                right_angle = -1 * (3.14159 - angle)

    return [left_angle, right_angle]
