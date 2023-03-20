from model import ResNet
class Tracker():
    def __init__(self) -> None:
        self.backbone = ResNet()
        pass
    def init_train(self, first_frame=None, epochs=10):
        #TODO: finetune the model on the first frame for 10 epochs
        pass
    def track(self):
        #TODO: inference the model on the rest of the frames
        pass

if __name__ == "__main__":
    tracker = Tracker()
    tracker.init_train()
    tracker.track()