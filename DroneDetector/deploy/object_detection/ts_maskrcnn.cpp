#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torchvision/vision.h>
using namespace std;

c10::IValue get_tracing_inputs(cv::Mat& img, c10::Device device) {
  const int height = img.rows;
  const int width = img.cols;
  const int channels = 3;

  auto input =
      torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
  // HWC to CHW
  input = input.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
  return input;
}

struct MaskRCNNOutputs {
  at::Tensor pred_boxes, pred_classes, pred_masks, scores;
  int num_instances() const {
    return pred_boxes.sizes()[0];
  }
};

MaskRCNNOutputs get_outputs(std::string export_method, c10::IValue outputs) {
  // Given outputs of the model, extract tensors from it to turn into a
  // common MaskRCNNOutputs format.
  auto out_tuple = outputs.toTuple()->elements();
    // They are ordered alphabetically by their field name in Instances
    return MaskRCNNOutputs{
        out_tuple[0].toTensor(),
        out_tuple[1].toTensor(),
        out_tuple[2].toTensor(),
        out_tuple[3].toTensor()
    };
}

int main(int argc, const char* argv[]) {
  if (argc != 4) {
    cerr << R"xx(
Usage:
   ./torchscript_mask_rcnn model.ts input.jpg EXPORT_METHOD

   EXPORT_METHOD can be "tracing".
)xx";
    return 1;
  }
  std::string image_file = argv[2];
  std::string export_method = argv[3];
  assert(export_method == "tracing");

  torch::jit::FusionStrategy strat = {{torch::jit::FusionBehavior::DYNAMIC, 1}};
  torch::jit::setFusionStrategy(strat);
  torch::autograd::AutoGradMode guard(false);
  auto module = torch::jit::load(argv[1]);

  assert(module.buffers().size() > 0);
  // Assume that the entire model is on the same device.
  // We just put input to this device.
  auto device = (*begin(module.buffers())).device();

  cv::Mat input_img = cv::imread(image_file, cv::IMREAD_COLOR);
  auto inputs = get_tracing_inputs(input_img, device);

  // Run the network
  auto output = module.forward({inputs});

  // run 3 more times to benchmark
  int N_benchmark = 1, N_warmup = 1;
  auto start_time = chrono::high_resolution_clock::now();
  for (int i = 0; i < N_benchmark + N_warmup; ++i) {
    if (i == N_warmup)
      start_time = chrono::high_resolution_clock::now();
    output = module.forward({inputs});
  }
  auto end_time = chrono::high_resolution_clock::now();
  auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
                .count();
  cout << "Latency (should vary with different inputs): "
       << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;

  // Parse Mask R-CNN outputs
  auto rcnn_outputs = get_outputs(export_method, output);
  /*
  cout << "Number of detected objects: " << rcnn_outputs.num_instances()
       << endl;

  cout << "pred_boxes: " << rcnn_outputs.pred_boxes.toString() << " "
       << rcnn_outputs.pred_boxes.sizes() << endl;
  cout << "scores: " << rcnn_outputs.scores.toString() << " "
       << rcnn_outputs.scores.sizes() << endl;
  cout << "pred_classes: " << rcnn_outputs.pred_classes.toString() << " "
       << rcnn_outputs.pred_classes.sizes() << endl;
  cout << "pred_masks: " << rcnn_outputs.pred_masks.toString() << " "
       << rcnn_outputs.pred_masks.sizes() << endl;*/
  for (size_t i = 0; i < rcnn_outputs.num_instances(); i++)
  {
    //cout << rcnn_outputs.pred_boxes[i] << endl;
    //cout << rcnn_outputs.pred_boxes[i][0] << endl;
    at::Tensor box = rcnn_outputs.pred_boxes[i];
    cv::Point pt1(box[0].item<int>(), box[1].item<int>());
    cv::Point pt2(box[2].item<int>(), box[3].item<int>());
    cv::rectangle(input_img, pt1, pt2, cv::Scalar(0,255,0), 3);
  }
  cv::imshow("img.jpg", input_img);
  cv::waitKey(0);
  cout << rcnn_outputs.pred_boxes << endl;
  return 0;
}