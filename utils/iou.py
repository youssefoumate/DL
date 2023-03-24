import numpy as np

def calc_iou (pred_box, gt_box):
    # pred_box and gt_box are arrays of shape (4,) with (center_x, center_y, w, h)
    pred_xmin = pred_box[0] - gt_box[2] / 2
    pred_xmax = pred_box[0] + gt_box[2] / 2
    pred_ymin = pred_box[1] - gt_box[3] / 2
    pred_ymax = pred_box[1] + gt_box[3] / 2

    gt_xmin = gt_box[0] - gt_box[2] / 2
    gt_xmax = gt_box[0] + gt_box[2] / 2
    gt_ymin = gt_box[1] - gt_box[3] / 2
    gt_ymax = gt_box[1] + gt_box[3] / 2

    inter_xmin = max(pred_xmin, gt_xmin)
    inter_xmax = min(pred_xmax, gt_xmax)
    inter_ymin = max(pred_ymin, gt_ymin)
    inter_ymax = min(pred_ymax, gt_ymax)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)

    inter_area = inter_w * inter_h
    pred_area = gt_box[2] * gt_box[3]
    gt_area = gt_box[2] * gt_box[3]

    iou_score = inter_area / (pred_area + gt_area - inter_area)

    return iou_score

if __name__ == "__main__":
    # define a single box with format (center_x, center_y, w, h)
    box = np.array([121, 210, 40, 70])

    # define a ground-truth box with format (center_x, center_y, w, h)
    gt_box = np.array([120, 210, 40, 70])

    # calculate the IOU score
    iou_score = calc_iou(box, gt_box)
    print(iou_score)