import pandas as pd
import numpy as np

def get_iou(first_rect, second_rect):
    # determine the (x, y)-coordinates of the given rectangles
    x_start = max(first_rect[0], second_rect[0])
    y_start = max(first_rect[1], second_rect[1])
    x_end = min(first_rect[2], second_rect[2])
    y_end = min(first_rect[3], second_rect[3])
    # computing the area of intersection
    area_of_intersection = max(0, x_end - x_start + 1) * max(0, y_end - y_start + 1)
    # computing the area of both the prediction and ground-truth rectangles
    area_of_first = (first_rect[2] - first_rect[0] + 1) * (first_rect[3] - first_rect[1] + 1)
    area_of_second = (second_rect[2] - second_rect[0] + 1) * (second_rect[3] - second_rect[1] + 1)
    # returning the intersection over union value 
    # that is the intersection area dividing by:
    # the sum of prediction + ground-truth areas - the interesection area
    return area_of_intersection / float(area_of_first + area_of_second - area_of_intersection)

def test(model_results):
    try:
        gts = pd.read_csv('ground-truths.txt', names=['x1', 'y1', 'x2', 'y2'])
    except FileNotFoundError:
        print("Can't read/find ground-truths.txt")
        return  # Exit the function if file not found
    except pd.errors.EmptyDataError:
        print("ground-truths.txt is empty")
        return

    print(gts)
    ground_truths = []
    for gt in gts.values:
        ground_truths.append(list(gt))

    predictions = np.zeros((len(ground_truths)), dtype=int)
    valid_results = np.zeros((len(model_results)), dtype=int)

    for i, result in enumerate(model_results):
        for j, truth in enumerate(ground_truths):
            if(get_iou(result, truth) > 0.7):
                predictions[j] = 1
                valid_results[i] = 1

    # 1s of valid_results is true pos
    tp = sum(valid_results == 1)
    # 0s of valid_results is false pos
    fp = sum(valid_results == 0)
    # 0s of predictions is false neg
    fn = sum(predictions == 0)
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f'precision: {precision:.2f}', f'recall: {recall:.2f}', f'f1-score {f1:.2f}', sep=' | ')
    print('-'*60)
    print(f'false positive: {fp}', f'false negative: {fn}', f'true positive: {tp}', sep=' | ')