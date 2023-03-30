import sys
from model import ResNet
sys.path.append("..")
from utils.dummy_video import create_dummy_video
from dataloader import Sampling
import cv2
import torch
from losses import bce_loss
from tqdm import tqdm

class Tracker():
    def __init__(self) -> None:
        self.model = ResNet()
        self.num_frames = 1000
        self.sampler = Sampling()
        self.sigmoid = torch.nn.Sigmoid()
        pass
    def init_train(self, init_frame=None, init_gt=None, epochs=10):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for _ in tqdm(range(epochs)):
            rois, _, labels = self.sampler.sample_generator(init_frame, init_gt, show=False)
            sum_loss = 0
            for roi, label in zip(rois, labels):
                roi = roi/255
                roi = torch.tensor(roi, dtype=torch.float32)
                label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
                output = self.model(roi.unsqueeze(0))
                loss = bce_loss(self.sigmoid(output), label.unsqueeze(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            print(loss/len(rois))

    def track(self):
        frame_gen = create_dummy_video(self.num_frames)
        init_frame = next(frame_gen)
        gt = init_frame[1]
        self.init_train(init_frame[0], gt)
        self.model.eval()
        for frame, _ in frame_gen:
            rois, coords, _ = self.sampler.sample_generator(frame, gt, show=False)
            max_roi = None
            max_score = 0
            max_coord = []
            for roi_idx, (coord, roi) in enumerate(zip(coords, rois)):
                roi = roi/255
                roi = torch.tensor(roi, dtype=torch.float32)
                score = self.model(roi.unsqueeze(0))
                if score > max_score or roi_idx == 0:
                    max_score = score
                    max_roi = roi
                    max_coord = coord
            gt = [max_coord[0], max_coord[1], gt[2], gt[3]]
            cv2.rectangle(
                    frame, (int(gt[1] - gt[2]/2), int(gt[0] - gt[3]/2)),
                    (int(gt[1] + gt[2]/2), int(gt[0] + gt[3]/2)),
                    (0, 255, 0), 1)
            cv2.imshow("frame", frame)
            cv2.waitKey(10)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = Tracker()
    tracker.track()