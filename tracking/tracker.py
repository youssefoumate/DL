import sys
from model import BboxClassifier, BboxRegressor, ResNet
sys.path.append("..")
from utils.dummy_video import create_dummy_video
from dataloader import Sampling
import cv2
import torch
from losses import bce_loss, mse_loss
from tqdm import tqdm
import numpy as np

class Tracker():
    def __init__(self) -> None:
        self.backbone = ResNet()
        self.classifier = BboxClassifier()
        self.regressor = BboxRegressor()
        self.num_frames = 1000
        self.sampler = Sampling()
        self.sigmoid = torch.nn.Sigmoid()
        pass
    
    def init_train(self, init_frame=None, init_gt=None, epochs=20):
        reg_optimizer = torch.optim.Adam(list(self.backbone.parameters()) + 
                                    list(self.regressor.parameters()), 
                                    lr=0.01)
        optimizer = torch.optim.Adam(list(self.backbone.parameters()) + 
                                    list(self.classifier.parameters()), 
                                    lr=0.01)
        init_frame = torch.tensor(init_frame.transpose(2, 0, 1), dtype=torch.float32)
        print("train bbox regressor...")
        for _ in tqdm(range(epochs)):
            out_feat = self.backbone(init_frame.unsqueeze(0))
            pred_coords = self.regressor(out_feat)
            init_gt = torch.tensor(init_gt).float()
            loss = mse_loss(pred_coords, init_gt)
            reg_optimizer.zero_grad()
            loss.backward()
            reg_optimizer.step()
            print("reg_loss: ", loss.item())
        print("train bbox classifier...")
        for _ in tqdm(range(epochs)):
            sum_loss = 0
            output = self.backbone(init_frame.unsqueeze(0))
            out_sample = output.clone()
            #exit()
            rois, _, labels = self.sampler.sample_generator(out_sample.detach().numpy(), init_gt, show=False)
            for roi, label in zip(rois, labels):
                roi = roi/255
                roi = torch.tensor(roi, dtype=torch.float32)
                label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
                output = self.classifier(roi)
                loss = bce_loss(self.sigmoid(output), label)
                sum_loss += loss
            optimizer.zero_grad()
            sum_loss.backward()
            optimizer.step()
            print("classifier: ", loss/len(rois))

    def track(self):
        frame_gen = create_dummy_video(self.num_frames)
        init_frame = next(frame_gen)
        gt = init_frame[1]
        self.init_train(init_frame[0], gt)
        self.backbone.eval() 
        self.classifier.eval()
        for frame, _ in frame_gen:
            frame_in = torch.tensor(frame.transpose(2, 0, 1), dtype=torch.float32)
            output = self.backbone(frame_in.unsqueeze(0))
            rois, coords, _ = self.sampler.sample_generator(output.detach().numpy(), gt, show=False)
            cv2.imshow("feat_map", cv2.resize(np.average(output.squeeze(0).detach().numpy(), 0), (256, 256)))
            cv2.waitKey(1)
            max_score = 0
            max_coord = []
            for roi_idx, (coord, roi) in enumerate(zip(coords, rois)):
                roi = roi/255
                roi = torch.tensor(roi, dtype=torch.float32)
                print(roi.shape)
                score = self.classifier(roi)
                if score > max_score or roi_idx == 0:
                    max_score = score
                    max_coord = coord
            gt = [max_coord[0], max_coord[1], gt[2], gt[3]]
            cv2.rectangle(
                    frame, (int(gt[1] - gt[2]/2), int(gt[0] - gt[3]/2)),
                    (int(gt[1] + gt[2]/2), int(gt[0] + gt[3]/2)),
                    (0, 255, 0), 3)
            cv2.imshow("frame", frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = Tracker()
    tracker.track()