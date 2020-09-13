import json
import pickle as pickle
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def precision(tp, fp):
    # if tp==0 and fp==0:
    #     return 0.0
    # else:
    return tp / (tp + fp+1e-8)

def recall(tp, fn):
    # if tp==0 and fn==0:
    #     return 0.0
    # else:
    return tp / (tp + fn+1e-8)

def f1_score(p, r):
    # if p==0.0 and r==0.0:
    #     return 0.0
    # else:
    return (2 * p * r) / (p + r+1e-8)

# Returns true if two rectangles bbox1 and bbox2 overlap
def doOverlap(bbox1, bbox2):
    l1_x, l1_y, r1_x, r1_y = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    l2_x, l2_y, r2_x, r2_y = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    # If one rectangle is on left side of other
    if (l1_x >= r2_x or l2_x >= r1_x):
        return False

    # If one rectangle is above other
    if (l1_y <= r2_y or l2_y <= r1_y):
        return False

    return True

def intersects(bbox1, bbox2):
    l1_x, l1_y, r1_x, r1_y = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    l2_x, l2_y, r2_x, r2_y = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    return not (r1_x < l2_x or l1_x > r2_x or r1_y < l2_y or l1_y > r2_y)

#code borrowed from :https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0, 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou, interArea/boxAArea #boxA is gt


def get_num_gt_boxes(path):
    # gt_dir_ = "/media/data/clevr-dataset-gen/question_generation/CLEVR_train_question2bboxes.json"
    gt_ = load_json(path)
    num_boxes = 0
    for q, boxes in tqdm(gt_.items()):
        num_boxes += len(boxes.items())
    print(num_boxes)
    return num_boxes