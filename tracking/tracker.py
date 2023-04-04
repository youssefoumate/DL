import sys
from model import BboxClassifier, ResNet
sys.path.append("..")
from utils.dummy_video import create_dummy_video
from dataloader import Sampling
import cv2
import torch
from losses import bce_loss
from tqdm import tqdm

class Tracker():
    def __init__(self) -> None:
        self.backbone = ResNet()
        self.classifier = BboxClassifier()
        self.num_frames = 1000
        self.sampler = Sampling()
        self.sigmoid = torch.nn.Sigmoid()
        pass
    def init_train(self, init_frame=None, init_gt=None, epochs=3):
        optimizer = torch.optim.Adam(list(self.backbone.parameters()) + list(self.classifier.parameters()), lr=0.001)
        for _ in tqdm(range(epochs)):
            rois, _, labels = self.sampler.sample_generator(init_frame, init_gt, show=False)
            sum_loss = 0
            for roi, label in zip(rois, labels):
                roi = roi/255
                roi = torch.tensor(roi, dtype=torch.float32)
                label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
                output = self.backbone(roi.unsqueeze(0))
                output = self.classifier(output)
                loss = bce_loss(self.sigmoid(output), label)
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
        self.backbone.eval() 
        self.classifier.eval()
        for frame, _ in frame_gen:
            rois, coords, _ = self.sampler.sample_generator(frame, gt, show=False)
            max_score = 0
            max_coord = []
            for roi_idx, (coord, roi) in enumerate(zip(coords, rois)):
                roi = roi/255
                roi = torch.tensor(roi, dtype=torch.float32)
                output = self.backbone(roi.unsqueeze(0))
                score = self.classifier(output)
                if score > max_score or roi_idx == 0:
                    max_score = score
                    max_coord = coord
            gt = [max_coord[0], max_coord[1], gt[2], gt[3]]
            cv2.rectangle(
                    frame, (int(gt[1] - gt[2]/2), int(gt[0] - gt[3]/2)),
                    (int(gt[1] + gt[2]/2), int(gt[0] + gt[3]/2)),
                    (0, 255, 0), 3)
            cv2.imshow("frame", frame)
            cv2.waitKey(10)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = Tracker()
    tracker.track()