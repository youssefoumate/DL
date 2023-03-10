import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from math import cos, sin, sqrt
import numpy as np


class Sampling():

    def __init__(self):
        #num. of candidates
        self.num_samples = 50
        #initial box
        self.gt = np.array([10., 10., 40., 80.])
        #kalman params
        self.kalman = cv2.KalmanFilter(self.gt.shape[0],
                                       self.num_samples * self.gt.shape[0], 0)
        self.kalman.transitionMatrix = np.array([
            [1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ]).astype(np.float32)  # np.dot(kalman.transitionMatrix, state)
        self.kalman.measurementMatrix = 1. * np.eye(self.num_samples * self.gt.shape[0]).astype(
                np.float32)  #np.dot(kalman.measurementMatrix, state)
        self.kalman.processNoiseCov = 1e-5 * np.eye(self.gt.shape[0]).astype(np.float32) #sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(50, 1)
        dt = 1
        q = 1e-5
        self.kalman.processNoiseCov = q * np.array(
            [[dt**3 / 3, dt**2 / 2, dt**2 / 2, dt], [dt**2 / 2, dt, dt, q],
             [dt**2 / 2, dt, dt**3 / 3, dt**2 / 2], [dt, q, dt**2 / 2, q]]).astype(
                np.float32
            )
        
        self.kalman.measurementNoiseCov = 1e-1 * np.eye(
            self.num_samples * self.gt.shape[0]).astype(
                np.float32
            )  #kalman.measurementNoiseCov * np.random.randn(1, 1)
        self.kalman.statePre = np.array([self.gt[0], self.gt[1], 0, 0]).astype(np.float32)
        self.kalman.errorCovPre = np.eye(self.gt.shape[0]).astype(np.float32)

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

    def kalman_sampling(self, img_size, bb, n):
        # Initialize state with ground truth
        state = self.norm_sampling(img_size, bb, n)

        # Predict new state
        prediction = self.kalman.predict()
        print(prediction)
        # Generate measurement noise
        print(self.kalman.measurementMatrix)
        measurement = state.reshape(-1,1)
        print(measurement.shape)
        # Correct the prediction based on the measurement
        self.kalman.correct(measurement)
        print(prediction[0][0])
        exit()
        return prediction

    def vis(self):
        for offset in range(25):
            self.gt[:2] = self.gt[:2] + offset
            gt = self.gt
            samples = self.kalman_sampling(512, gt, self.num_samples)
            black_img = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.rectangle(black_img, (gt[0], gt[1]),
                          (gt[0] + gt[2], gt[1] + gt[3]), (0, 255, 0), 1)
            for sample in samples:
                cv2.rectangle(
                    black_img, (int(sample[0]), int(sample[1])),
                    (int(sample[0] + sample[2]), int(sample[1] + sample[3])),
                    (0, 0, 255), 1)
            cv2.imshow("img", black_img)
            cv2.waitKey(300)


if __name__ == "__main__":
    sampler = Sampling()
    sampler.vis()
