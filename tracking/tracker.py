import sys
from model import ResNet
sys.path.append("..")
from utils.dummy_video import create_dummy_video
from dataloader import Sampling
import cv2

class Tracker():
    def __init__(self) -> None:
        self.backbone = ResNet()
        self.num_frames = 1000
        self.sampler = Sampling()
        pass
    def init_train(self, init_frame=None, init_gt=None, epochs=10):
        rois = self.sampler.sample_generator(init_frame, init_gt, show=True)
    def track(self):
        frame_gen = create_dummy_video(self.num_frames)
        init_frame = next(frame_gen)
        self.init_train(init_frame[0], init_frame[1])
        """
        for frame in frame_gen:
            cv2.imshow("frame", frame)
            cv2.waitKey(10)
        """
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = Tracker()
    tracker.track()