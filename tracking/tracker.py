import sys
from model import ResNet
sys.path.append("..")
from utils.dummy_video import create_dummy_video
import cv2
class Tracker():
    def __init__(self) -> None:
        self.backbone = ResNet()
        self.num_frames = 1000
        pass
    def init_train(self, init_frame=None, epochs=10):
        #TODO: finetune the model on the first frame for 10 epochs
        pass
    def track(self):
        frame_gen = create_dummy_video(self.num_frames)
        init_frame = next(frame_gen)
        for frame in frame_gen:
            cv2.imshow("frame", frame)
            cv2.waitKey(10)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = Tracker()
    tracker.track()