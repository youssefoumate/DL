import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from math import cos, sin, sqrt
import numpy as np
from utils.iou import calc_iou


class Sampling():

    def __init__(self):
        #num. of candidates
        self.img_size = 512
        self.num_samples = 50
        #initial box
        self.gt = np.array([10., 10., 40., 80.])
        self.filters = []

    def kalman_setup(self, gt):
        init_state = self.norm_sampling(gt)
        for i in range(self.num_samples):
            kalman = cv2.KalmanFilter(4, 2) # State size: 4 (x, y, dx, dy), Measurement size: 2 (x, y)
            kalman.transitionMatrix = np.array([[1., 0., 1., 0.], # State transition model
                                            [0., 1., 0., 1.],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]]).astype(np.float32)
            kalman.measurementMatrix = np.array([[1., 0., 0., 0.], # Measurement model
                                            [0., 1., 0., 0.]]).astype(np.float32)
            kalman.processNoiseCov = np.eye(4).astype(np.float32) * (10 ** -3) # Process noise covariance
            kalman.measurementNoiseCov = np.eye(2).astype(np.float32) * (10 ** -2) # Measurement noise covariance
            kalman.errorCovPost = np.eye(4).astype(np.float32) * (10 ** -3) # Posteriori error estimate covariance matrix
            kalman.statePost = np.array([init_state[i][0], init_state[i][1], 
                                    np.random.randn(), 
                                    np.random.randn()], dtype=np.float32).reshape(-1,1) # Initial state estimate
            self.filters.append(kalman)

    def norm_sampling(self, bb):
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')
        trans = 1
        scale = 1
        # (center_x, center_y, w, h)
        sample = np.array([bb[0] + bb[2] / 2, bb[1] + bb[3] / 2, bb[2], bb[3]],
                          dtype='float32')
        samples = np.tile(sample[None, :], (self.num_samples, 1))
        samples[:, :2] += trans * np.mean(bb[2:]) * np.clip(
            0.5 * np.random.randn(self.num_samples, 2), -1, 1)
        samples[:, 2:] *= scale**np.clip(0.5 * np.random.randn(self.num_samples, 1), -1, 1)
        samples[:, 2:] = np.clip(samples[:, 2:], 10, self.img_size - 10)
        samples[:, :2] = np.clip(samples[:, :2], samples[:, 2:] / 2,
                                 self.img_size - samples[:, 2:] / 2 - 1)
        samples[:, :2] -= samples[:, 2:] / 2
        return samples

    def kalman_sampling(self, bb):
        # Initialize state with ground truth
        state = self.norm_sampling(bb)
        measurements = []
        for state_idx in range(len(state)):
            measurements.append([state[state_idx][0], state[state_idx][1]])
        measurements = np.array(measurements).astype(np.float32)
        predictions = []
        for sample_idx in range(self.num_samples):
            prediction = self.filters[sample_idx].predict()[:2] # Predict next state of point i
            correction = self.filters[sample_idx].correct(measurements[sample_idx].reshape(-1,1))[:2] # Correct state estimate with measurement of point i
            prediction_x,prediction_y= prediction.flatten().astype(int)
            predictions.append([prediction_x,prediction_y])
            correction_x ,correction_y= correction.flatten().astype(int)
            measured_x ,measured_y= measurements[sample_idx].astype(int)
        return predictions, measurements
    
    def roi_crop(self, image, centers, w, h, size=512, show=False):
        rois = []
        for center in centers:
            crop = image[int(max(0, center[0]-h/2)):int(center[0]+h/2), int(max(0, center[1]-w/2)):int(center[1]+w/2), :]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            crop_resized = cv2.resize(crop, (size, size))
            crop_resized = crop_resized.transpose(2, 0, 1)
            rois.append(crop_resized)
        return rois
    
    def generate_labels(self, boxes, gt, thresh=0.7):
        labels = []
        for box in boxes:
            iou_score = calc_iou(box, gt)
            if iou_score > thresh:
                label = 1
            else:
                label = 0
            labels.append(label)
        return labels
            
    def sample_generator(self, image, gt, show=True):
        self.kalman_setup(gt)
        preds, measures = self.kalman_sampling(gt)
        preds.extend(measures)
        rois = self.roi_crop(image, preds, gt[2], gt[3], show=True)
        labels = self.generate_labels(preds, gt)
        if show:
            cv2.rectangle(
                    image, (int(gt[0] - gt[2]/2), int(gt[1] - gt[3]/2)),
                    (int(gt[0] + gt[2]/2), int(gt[1] + gt[3]/2)),
                    (0, 255, 0), 1)
            for pred_box, roi, label in zip(preds, rois, labels):
                pred_color = (255*label, 0, 255*(1 - label))
                cv2.rectangle(
                    image, (int(pred_box[0] - gt[2]/2), int(pred_box[1] - gt[3]/2)),
                    (int(pred_box[0] + gt[2]/2), int(pred_box[1] + gt[3]/2)),
                    pred_color, 1)
                cv2.imshow("roi", roi)
                cv2.imshow("img", image)
                cv2.waitKey(0)
        return rois, preds, labels


if __name__ == "__main__":
    sampler = Sampling()
    rois = sampler.sample_generator()
