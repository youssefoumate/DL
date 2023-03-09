import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from math import cos, sin, sqrt
import numpy as np


class Sampling():

    def __init__(self):

        self.transitionMatrix = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.],
                                          [0., 0., 1., 1.], [0., 0., 0., 1.]])
        self.measurementMatrix = 1. * np.ones((1, 4))
        self.processNoiseCov = 1e-5 * np.eye(4)
        self.measurementNoiseCov = 1e-1 * np.ones((1, 1))
        self.errorCovPost = 1. * np.ones((4, 4))
        self.statePost = 0.1 * np.random.randn(4, 1)
        img_height = 500
        img_width = 500
        self.kalman = cv2.KalmanFilter(2, 1, 0)
        bbox_w = 10
        bbox_h = 20
        code = -1
        self.gt = np.array([10, 10, 40, 80])
        #cv2.namedWindow("Kalman")

    def kalman_sampling(self):
        #TODO implement soon
        pass

    def norm_sampling(self, img_size, bb, n):
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')
        trans = 1
        scale = 1
        # (center_x, center_y, w, h)
        sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]],
                          dtype='float32')
        samples = np.tile(sample[None, :], (n, 1))
        samples[:, :2] += trans * np.mean(bb[2:]) * np.clip(
            0.5 * np.random.randn(n, 2), -1, 1)
        samples[:, 2:] *= scale**np.clip(0.5 * np.random.randn(n, 1), -1, 1)
        samples[:, 2:] = np.clip(samples[:, 2:], 10, img_size - 10)
        samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2,
                                 img_size - samples[:, 2:] / 2 - 1)
        samples[:, :2] -= samples[:, 2:] / 2
        return samples

    def vis(self):
        for offset in range(25):
            self.gt[:2] = self.gt[:2] + offset
            gt = self.gt
            samples =   self.norm_sampling(512, gt, 100)
            black_img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.rectangle(black_img, 
                        (gt[0], gt[1]),
                        (gt[0]+gt[2], gt[1]+gt[3]),
                        (0,255,0), 1)
            for sample in samples:
                cv2.rectangle(black_img, 
                        (int(sample[0]), int(sample[1])),
                        (int(sample[0]+sample[2]), int(sample[1]+sample[3])),
                        (0,0,255), 1)
            cv2.imshow("img", black_img)
            cv2.waitKey(300)
            


if __name__ == "__main__": 
    sampler = Sampling()
    sampler.vis()
