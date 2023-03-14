import numpy as np
import csv

path = './faces'
list_file = './list.csv'
alg_file = './detections.csv'
gt_file = './gt.csv'

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_bounding_coordinates(rectA, rectB):
    x1 = max(rectA[0], rectB[0])
    y1 = max(rectA[1], rectB[1])
    x2 = min(rectA[2], rectB[2])
    y2 = min(rectA[3], rectB[3])

    return (x1, y1, x2, y2)


def get_precision_recall_f1(iou_list, num_files, threshold):
    tp = 0
    fp = 0
    fn = 0

    for iou in iou_list:
        if iou['iou_score'] >= threshold:
            tp += 1
        else:
            fp += 1
            fn += 1
    
    # Recall = TP / (TP+FN)
    recall = tp / num_files
    # Precision = TP / (TP+FP)
    precision = tp / (tp + fp)
    
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        # 2PR / (P+R)
        f1 = (2 * precision * recall) / (precision + recall)
    return (recall, precision, f1)


iou_list = []
gt_list = []
pred_bound = []
with open(alg_file, newline='') as det_csv:
    alg_reader = csv.DictReader(det_csv, fieldnames=['Filepath', 'x1', 'y1', 'x2', 'y2'])
    with open(gt_file, newline='') as gt_csv:
        gt_reader = csv.DictReader(gt_csv, fieldnames=['Filepath', 'x1', 'y1', 'x2', 'y2'])    
        for alg_row in alg_reader:
            photo_name = alg_row['Filepath'].replace('./faces/', '')
            for gt_row in gt_reader:
                gt_list.append(gt_row)
                if photo_name == gt_row['Filepath'].replace('C:/Lab08/faces/', ''):
                    rect1 = (int(alg_row['x1']), int(alg_row['y1']), int(alg_row['x2']), int(alg_row['y2']))
                    rect2 = (int(gt_row['x1']), int(gt_row['y1']), int(gt_row['x2']), int(gt_row['y2']))
                    pred_bound.append(rect1)
                    iou_list.append({'Filepath': photo_name, 'iou_score' : bb_intersection_over_union(rect1, rect2)})
                    break

print('GT List: ', gt_list)
print('Prediction Bounding: ', pred_bound)
print('IoU List: ', iou_list)

with open(list_file, mode='w', newline='') as list_csv:
    fieldnames = ['Filepath', 'iou_score']
    writer = csv.DictWriter(list_csv, fieldnames=fieldnames)
    writer.writeheader()
    for iou in iou_list:
        writer.writerow(iou)

iou_total=0.0
for iou in iou_list:
    iou_total += iou['iou_score']

iou_avg = iou_total / len(iou_list)

print('IoU Average: ', iou_avg)

for threshold in np.arange(0, 1, 0.1):
    recall, precision, f1 = get_precision_recall_f1(iou_list, 8, threshold)
    print(f'=== Threshold {threshold:.1f} ===')
    print(f'Recall: {recall:.5f}')
    print(f'Precision: {precision:.5f}')
    print(f'F1 {f1:.5f}')
    