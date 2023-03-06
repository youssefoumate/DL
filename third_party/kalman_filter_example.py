"""
   Tracking of rotating point.
   Rotation speed is constant.
   Both state and measurements vectors are 1D (a point angle),
   Measurement is the real point angle + gaussian noise.
   The real and the estimated points are connected with yellow line segment,
   the real and the measured points are connected with red line segment.
   (if Kalman filter works correctly,
    the yellow segment should be shorter than the red one).
   Pressing any key (except ESC) will reset the tracking with a different speed.
   Pressing ESC will stop the program.
"""
# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    long = int

import cv2
from math import cos, sin, sqrt
import numpy as np

if __name__ == "__main__":

    img_height = 500
    img_width = 500
    kalman = cv2.KalmanFilter(2, 1, 0)
    bbox_w = 10
    bbox_h = 20

    code = long(-1)

    cv2.namedWindow("Kalman")

    while True:
        state = 0.1 * np.random.randn(4, 1)

        kalman.transitionMatrix = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [0. ,0. ,1., 1.], [0., 0., 0., 1.]])
        kalman.measurementMatrix = 1. * np.ones((1, 4))
        kalman.processNoiseCov = 1e-5 * np.eye(4)
        kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1))
        kalman.errorCovPost = 1. * np.ones((4, 4))
        kalman.statePost = 0.1 * np.random.randn(4, 1)

        while True:
            def calc_point(angle):
                return (np.around(img_width/2 + img_width/3*cos(angle), 0).astype(int),
                        np.around(img_height/2 - img_width/3*sin(angle), 1).astype(int))

            state_angle = state[0, 0]
            state_w = state[1, 0]
            state_h = state[2, 0]
            state_pt = calc_point(state_angle)
            print(state_pt)

            prediction = kalman.predict()
            predict_angle = prediction[0, 0]
            predict_w = prediction[1, 0]
            predict_h = prediction[2, 0]
            predict_pt = calc_point(predict_angle)

            measurement = kalman.measurementNoiseCov * np.random.randn(1, 1)

            # generate measurement
            measurement = np.dot(kalman.measurementMatrix, state) + measurement

            measurement_angle = measurement[0, 0]
            #measurement_w = measurement[1, 0]
            #measurement_h = measurement[2, 0]
            measurement_pt = calc_point(measurement_angle)

            # plot points
            def draw_bbox(center, color, w, h, d):
                top = center[0] - int(w)
                left = center[1] - int(h)
                bottom = center[0] + int(w)
                right = center[1] + int(h)
                cv2.rectangle(img, (top, left), (bottom, right), color, 1, cv2.LINE_AA, 0)

            def draw_cross(center, color, d):
                cv2.line(img,
                         (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
                         color, 1, cv2.LINE_AA, 0)
                cv2.line(img,
                         (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
                         color, 1, cv2.LINE_AA, 0)

            img = np.zeros((img_height, img_width, 3), np.uint8)
            #draw_cross(np.int32(state_pt), (255, 255, 255), 3)
            draw_bbox(np.int32(state_pt), (0, 255, 0), state_w, state_h, 3)
            draw_bbox(np.int32(state_pt), (255, 0, 0), predict_w, predict_h, 3)
            #draw_bbox(np.int32(state_pt), (0, 255, 0), 3)
            #draw_cross(np.int32(measurement_pt), (0, 0, 255), 3)
            #draw_cross(np.int32(predict_pt), (0, 255, 0), 3)

            #cv2.line(img, state_pt, measurement_pt, (0, 0, 255), 3, cv2.LINE_AA, 0)
            #cv2.line(img, state_pt, predict_pt, (0, 255, 255), 3, cv2.LINE_AA, 0)

            kalman.correct(measurement)

            process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(4, 1)
            state = np.dot(kalman.transitionMatrix, state) + process_noise

            cv2.imshow("Kalman", img)

            code = cv2.waitKey(100)
            if code != -1:
                break

        if code in [27, ord('q'), ord('Q')]:
            break

    cv2.destroyWindow("Kalman")