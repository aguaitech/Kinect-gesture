import sys
import argparse
from openni import openni2, nite2, utils
import numpy as np
import cv2
import colorsys
import math
from bresenham import bresenham
import os

GRAY_COLOR = (65, 65, 65)
CAPTURE_SIZE_KINECT = (512, 424)
CAPTURE_SIZE_OTHERS = (640, 480)

handPositionCache = dict()
userColorCache = dict()
handTraceCache = dict()
mode = 'spray'
splashPositionCache = dict()
splashRemainTime = dict()
splashColor = dict()


def parse_arg():
    parser = argparse.ArgumentParser(description='Test OpenNI2 and NiTE2.')
    parser.add_argument('-w', '--window_width', type=int, default=1024,
                        help='Specify the window width.')
    return parser.parse_args()


def refresh_color(user):
    if not userColorCache.get(user.id):
        userColorCache[user.id] = colorsys.hsv_to_rgb(0, 1, 1)
    elbow = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_ELBOW]
    shoulder = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_SHOULDER]
    offx = elbow.position.x - shoulder.position.x
    offy = elbow.position.y - shoulder.position.y
    offz = elbow.position.z - shoulder.position.z
    vec = np.array((offx, offy, offz))
    vec /= np.linalg.norm(vec)
    userColorCache[user.id] = colorsys.hsv_to_rgb(
        math.acos(max(0, -vec[1]))/math.pi*2, 1, 1)
    (r, g, b) = userColorCache[user.id]
    # return (round(b * 255), round(g * 255), round(r * 255))
    return (b, g, r)


def draw_limb(img, ut, j1, j2, col):
    (x1, y1) = ut.convert_joint_coordinates_to_depth(
        j1.position.x, j1.position.y, j1.position.z)
    (x2, y2) = ut.convert_joint_coordinates_to_depth(
        j2.position.x, j2.position.y, j2.position.z)

    if (0.3 < j1.positionConfidence and 0.3 < j2.positionConfidence):
        c = col
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)

        c = GRAY_COLOR if (j1.positionConfidence < 1.0) else col
        cv2.circle(img, (int(x1), int(y1)), 2, c, -1)

        c = GRAY_COLOR if (j2.positionConfidence < 1.0) else col
        cv2.circle(img, (int(x2), int(y2)), 2, c, -1)


def draw_hand(img, curImg, ut, user, color):
    global mode
    elbow = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_ELBOW]
    shoulder = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_SHOULDER]
    hand = user.skeleton.joints[nite2.JointType.NITE_JOINT_RIGHT_HAND]
    offx = elbow.position.x - shoulder.position.x
    offy = elbow.position.y - shoulder.position.y
    offz = elbow.position.z - shoulder.position.z
    vec1 = np.array((offx, offy, offz))
    vec1 /= np.linalg.norm(vec1)
    offx = hand.position.x - elbow.position.x
    offy = hand.position.y - elbow.position.y
    offz = hand.position.z - elbow.position.z
    vec2 = np.array((offx, offy, offz))
    vec2 /= np.linalg.norm(vec2)

    hand = user.skeleton.joints[nite2.JointType.NITE_JOINT_LEFT_HAND]
    (x, y) = ut.convert_joint_coordinates_to_depth(
        hand.position.x, hand.position.y, hand.position.z)
    cv2.circle(curImg, (int(x), int(y)), 4, color, -1)
    if mode == 'splash':
        splash(img, ut, user.id)
    else:
        mode = analyse_hand_mode(ut, user.id, hand)
        if mode == 'spray':
            if np.dot(vec1, vec2) < -0.1:
                cv2.circle(curImg, (int(x), int(y)), 8, color, -1)
                draw_hand_path(
                    img, ut, user.id, user.skeleton.joints[nite2.JointType.NITE_JOINT_LEFT_HAND], color)
            else:
                if np.dot(vec1, vec2) >= -0.1:
                    handPositionCache[user.id] = None
        elif mode == 'drip':
            drip(img, ut, user.id, color)
        else:
            splash(img, ut, user.id, color)


def analyse_hand_mode(ut, hid, hand):
    if not handTraceCache.get(hid):
        handTraceCache[hid] = []
    if len(handTraceCache[hid]) < 10:
        handTraceCache[hid].append(hand.position)
        return 'spray'
    handTraceCache[hid].pop(0)
    handTraceCache[hid].append(hand.position)
    if handTraceCache[hid][-1].z - handTraceCache[hid][0].z < -400:
        return 'splash'
    elif np.linalg.norm(np.array(ut.convert_joint_coordinates_to_depth(handTraceCache[hid][-1].x, handTraceCache[hid][-1].y, handTraceCache[hid][-1].z)) - np.array(ut.convert_joint_coordinates_to_depth(handTraceCache[hid][0].x, handTraceCache[hid][0].y, handTraceCache[hid][0].z))) > 80:
        return 'drip'
    return 'spray'


def draw_hand_path(img, ut: nite2.UserTracker, hid: int, hand: nite2.HandData, col):
    if not handPositionCache.get(hid):
        handPositionCache[hid] = ut.convert_joint_coordinates_to_depth(
            hand.position.x, hand.position.y, hand.position.z)
        return
    (x, y) = ut.convert_joint_coordinates_to_depth(
        hand.position.x, hand.position.y, hand.position.z)
    for point in bresenham(int(handPositionCache[hid][0]), int(
            handPositionCache[hid][1]), int(x), int(y)):
        spray_paint(img, *point, col)
    handPositionCache[hid] = (x, y)


def spray_paint(img, x, y, col):
    spray = np.random.normal(size=(20, 2)) * 10
    for idx in range(20):
        dotx = int(x + spray[idx, 0])
        doty = int(y + spray[idx, 1])
        r = int(max(1, min(3, 3/(spray[idx, 0]+spray[idx, 1]))))
        cv2.circle(img, (dotx, doty), r, col, -1)


def splash(img, ut, uid, col=None):
    global splashColor, splashRemainTime, splashPositionCache, mode
    if not splashRemainTime.get(uid):
        splashRemainTime[uid] = 0
    if splashRemainTime[uid] == 0:
        splashColor[uid] = col
        splashRemainTime[uid] = 1
    (x, y) = ut.convert_joint_coordinates_to_depth(
        handTraceCache[uid][-1].x, handTraceCache[uid][-1].y, handTraceCache[uid][-1].z)
    spray = np.random.normal(size=(20, 2)) * 30
    for idx in range(20):
        dotx = int(x + spray[idx, 0])
        doty = int(y + spray[idx, 1])
        r = int(max(1, min(5, abs(np.random.normal() * 5))))
        cv2.circle(img, (dotx, doty), r, splashColor[uid], -1)
    cv2.circle(img, (int(x), int(y)), 20, splashColor[uid], -1)
    splashRemainTime[uid] -= 1
    if splashRemainTime[uid] <= 0:
        handTraceCache[uid] = []
        mode = 'spray'


def drip(img, ut, uid, col):
    for idx in range(10):
        dot = handTraceCache[uid][idx]
        (x, y) = ut.convert_joint_coordinates_to_depth(dot.x, dot.y, dot.z)
        cv2.circle(img, (int(x), int(y)), 10-idx, col, -1)
    handTraceCache[uid] = []


def draw_skeleton(img, ut, user, col):
    for idx1, idx2 in [(nite2.JointType.NITE_JOINT_HEAD, nite2.JointType.NITE_JOINT_NECK),
                       # upper body
                       (nite2.JointType.NITE_JOINT_NECK,
                        nite2.JointType.NITE_JOINT_LEFT_SHOULDER),
                       (nite2.JointType.NITE_JOINT_LEFT_SHOULDER,
                        nite2.JointType.NITE_JOINT_TORSO),
                       (nite2.JointType.NITE_JOINT_TORSO,
                        nite2.JointType.NITE_JOINT_RIGHT_SHOULDER),
                       (nite2.JointType.NITE_JOINT_RIGHT_SHOULDER,
                        nite2.JointType.NITE_JOINT_NECK),
                       # left hand
                       (nite2.JointType.NITE_JOINT_LEFT_HAND,
                        nite2.JointType.NITE_JOINT_LEFT_ELBOW),
                       (nite2.JointType.NITE_JOINT_LEFT_ELBOW,
                        nite2.JointType.NITE_JOINT_LEFT_SHOULDER),
                       # right hand
                       (nite2.JointType.NITE_JOINT_RIGHT_HAND,
                        nite2.JointType.NITE_JOINT_RIGHT_ELBOW),
                       (nite2.JointType.NITE_JOINT_RIGHT_ELBOW,
                        nite2.JointType.NITE_JOINT_RIGHT_SHOULDER),
                       # lower body
                       (nite2.JointType.NITE_JOINT_TORSO,
                        nite2.JointType.NITE_JOINT_LEFT_HIP),
                       (nite2.JointType.NITE_JOINT_LEFT_HIP,
                        nite2.JointType.NITE_JOINT_RIGHT_HIP),
                       (nite2.JointType.NITE_JOINT_RIGHT_HIP,
                        nite2.JointType.NITE_JOINT_TORSO),
                       # left leg
                       (nite2.JointType.NITE_JOINT_LEFT_FOOT,
                        nite2.JointType.NITE_JOINT_LEFT_KNEE),
                       (nite2.JointType.NITE_JOINT_LEFT_KNEE,
                        nite2.JointType.NITE_JOINT_LEFT_HIP),
                       # right leg
                       (nite2.JointType.NITE_JOINT_RIGHT_FOOT,
                        nite2.JointType.NITE_JOINT_RIGHT_KNEE),
                       (nite2.JointType.NITE_JOINT_RIGHT_KNEE, nite2.JointType.NITE_JOINT_RIGHT_HIP)]:
        draw_limb(
            img, ut, user.skeleton.joints[idx1], user.skeleton.joints[idx2], col)


# -------------------------------------------------------------
# main program from here
# -------------------------------------------------------------


def init_capture_device():

    openni2.initialize(os.path.join(
        os.path.dirname(__file__), 'OpenNI2', 'Redist'))
    nite2.initialize(os.path.join(
        os.path.dirname(__file__), 'NiTE', 'Redist'))
    return openni2.Device.open_any()


def close_capture_device():
    nite2.unload()
    openni2.unload()


def capture_skeleton():
    args = parse_arg()
    dev = init_capture_device()

    dev_name = dev.get_device_info().name.decode('UTF-8')
    print("Device Name: {}".format(dev_name))
    use_kinect = False
    if dev_name == 'Kinect':
        use_kinect = True
        print('using Kinect.')

    try:
        user_tracker = nite2.UserTracker(dev)
        hand_tracker = nite2.HandTracker(None)
    except utils.NiteError:
        print("Unable to start the NiTE human tracker. Check "
              "the error messages in the console. Model data "
              "(s.dat, h.dat...) might be inaccessible.")
        sys.exit(-1)

    (img_w, img_h) = CAPTURE_SIZE_KINECT if use_kinect else CAPTURE_SIZE_OTHERS
    win_w = args.window_width
    win_h = int(img_h * win_w / img_w)

    hand_tracker.start_gesture_detection(
        nite2.GestureType.NITE_GESTURE_WAVE)
    hand_tracker.start_gesture_detection(
        nite2.GestureType.NITE_GESTURE_CLICK)

    ut_frame = user_tracker.read_frame()
    ht_frame = hand_tracker.read_frame()

    u_depth_frame = ut_frame.get_depth_frame()
    h_depth_frame = ht_frame.depth_frame

    sketch = np.ndarray((h_depth_frame.height, h_depth_frame.width),
                        dtype=np.uint16).astype(np.float32)
    if use_kinect:
        sketch = sketch[0:img_h, 0:img_w]
    sketch = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)

    while True:
        ut_frame = user_tracker.read_frame()
        ht_frame = hand_tracker.read_frame()
        h_depth_frame = ht_frame.depth_frame
        depth_frame_data = h_depth_frame.get_buffer_as_uint16()
        depth = np.ndarray((h_depth_frame.height, h_depth_frame.width), dtype=np.uint16,
                           buffer=depth_frame_data).astype(np.float32)
        cursorSketch = sketch.copy()
        if use_kinect:
            depth = depth[0:img_h, 0:img_w]

        (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(depth)
        if (min_val < max_val):
            depth = (depth - min_val) / (max_val - min_val)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

        if ut_frame.users:
            for user in ut_frame.users:
                if user.is_new():
                    print("new human id:{} detected.".format(user.id))
                    user_tracker.start_skeleton_tracking(user.id)
                elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and
                      user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                    color = refresh_color(user)
                    draw_skeleton(depth, user_tracker, user, color)
                    draw_hand(sketch, cursorSketch, user_tracker, user, color)

        # if ht_frame.gestures:
        #     for gesture in ht_frame.gestures:
        #         try:
        #             gesture: nite2.c_api.NiteGestureData = gesture
        #             print(gesture.type, gesture.currentPosition)
        #             hand_tracker.start_hand_tracking(gesture.currentPosition)
        #         except utils.NiteError:
        #             pass

        # if ht_frame.hands:
        #     for hand in ht_frame.hands:
        #         hand: nite2.HandData = hand
        #         if hand.state == 2:
        #             draw_hand_path(sketch, hand_tracker, hand.id, hand)
        #         elif hand.state == 0:
        #             handPositionCache[hand.id] = None

        cv2.imshow("Sketch", cv2.resize(cursorSketch, (win_w, win_h)))
        cv2.imshow("Depth", cv2.resize(depth, (win_w, win_h)))
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    close_capture_device()


if __name__ == '__main__':
    capture_skeleton()
