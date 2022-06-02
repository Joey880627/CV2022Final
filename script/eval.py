import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from utils import AverageMeter
# Object for prediction
from predict import predict,generate_output_file

def true_negative_curve(confs: np.ndarray, labels: np.ndarray, nr_thresholds: int = 1000):
    """Compute true negative rates
    Args:
        confs: the algorithm outputs
        labels: the ground truth labels
        nr_thresholds: number of splits for sliding thresholds

    Returns:

    """
    thresholds = np.linspace(0, 1, nr_thresholds)
    tn_rates = []
    for th in thresholds:
        # thresholding
        predict_negatives = (confs < th).astype(int)
        # true negative
        tn = np.sum((predict_negatives * (1 - labels) > 0).astype(int))
        tn_rates.append(tn / np.sum(1 - labels))
    return np.array(tn_rates)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    """Calculate the IoU score between two segmentation masks
    Args:
        mask1: 1st segmentation mask
        mask2: 2nd segmentation mask
    """
    if len(mask1.shape) == 3:
        mask1 = mask1.sum(axis=-1)
    if len(mask2.shape) == 3:
        mask2 = mask2.sum(axis=-1)
    area1 = cv2.countNonZero((mask1 > 0).astype(int))
    area2 = cv2.countNonZero((mask2 > 0).astype(int))
    if area1 == 0 or area2 == 0:
        return 0
    area_union = cv2.countNonZero(((mask1 + mask2) > 0).astype(int))
    area_inter = area1 + area2 - area_union
    return area_inter / area_union

def benchmark(dataset_path: str, subjects: list):
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    iou_meter = AverageMeter()
    iou_meter_sequence = AverageMeter()
    label_validity = []
    output_conf = []
    sequence_idx = 0
    for subject in subjects:
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            iou_meter_sequence.reset()
            label_name = os.path.join(image_folder, '0.png')
            if not os.path.exists(label_name):
                print(f'Labels are not available for {image_folder}')
                continue
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                label_name = os.path.join(image_folder, f'{idx}.png')
                image = np.asarray(Image.open(image_name))
                label = np.asarray(Image.open(label_name).convert('RGB'))
                # TODO: Modify the code below to run your method or load your results from disk
                # output, conf = my_awesome_algorithm(image)
                # output = label
                # conf = 1.0

                # Predict function
                output, conf = predict(image)

                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                    iou = mask_iou(output, label)
                    iou_meter.update(conf * iou)
                    iou_meter_sequence.update(conf * iou)
                else:  # empty ground truth label
                    label_validity.append(0.0)
                output_conf.append(conf)
            # print(f'[{sequence_idx:03d}] Weighted IoU: {iou_meter_sequence.avg()}')
    tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
    wiou = iou_meter.avg()
    atnr = np.mean(tn_rates)
    score = 0.7 * wiou + 0.3 * atnr
    print(f'\n\nOverall weighted IoU: {wiou:.4f}')
    print(f'Average true negative rate: {atnr:.4f}')
    print(f'Benchmark score: {score:.4f}')

    return score


def benchmark_all(dataset_path: str, subjects: list):
    """Compute the weighted IoU and average true negative rate
    Print all subjects' result
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    wiou_dict = {}
    atnr_dict = {}
    score_dict = {}
    iou_meter_all = AverageMeter()
    label_validity_all = []
    output_conf_all = []
    sequence_idx = 0
    for subject in subjects:
        iou_meter = AverageMeter()
        label_validity = []
        output_conf = []
        for action_number in range(26):
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            sequence_idx += 1
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            label_name = os.path.join(image_folder, '0.png')
            if not os.path.exists(label_name):
                print(f'Labels are not available for {image_folder}')
                # continue
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                image_name = os.path.join(image_folder, f'{idx}.jpg')
                image = np.asarray(Image.open(image_name))
                # TODO: Modify the code below to run your method or load your results from disk
                # Predict function
                output, conf = predict(image)

                # generate output file to submit
                if subject == 'S5':
                    generate_output_file(output,conf,action_number+1,idx)

                if not os.path.exists(label_name):
                    continue

                label_name = os.path.join(image_folder, f'{idx}.png')
                label = np.asarray(Image.open(label_name).convert('RGB'))
                if np.sum(label.flatten()) > 0:
                    label_validity.append(1.0)
                    label_validity_all.append(1.0)
                    iou = mask_iou(output, label)
                    iou_meter.update(conf * iou)
                    iou_meter_all.update(conf * iou)
                else:  # empty ground truth label
                    label_validity.append(0.0)
                    label_validity_all.append(0.0)
                output_conf.append(conf)
                output_conf_all.append(conf)
        
        if len(label_validity) == 0:
            continue
        
        tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
        wiou = iou_meter.avg()
        atnr = np.mean(tn_rates)
        score = 0.7 * wiou + 0.3 * atnr
        wiou_dict[subject] = wiou
        atnr_dict[subject] = atnr
        score_dict[subject] = score

    if len(label_validity_all) == 0:
        return
    
    for subject in list(wiou_dict.keys()):
        print('================================')
        print(f'{subject} overall weighted IoU: {wiou_dict[subject]:.4f}')
        print(f'{subject} average true negative rate: {atnr_dict[subject]:.4f}')
        print(f'{subject} benchmark score: {score_dict[subject]:.4f}')

    tn_rates_all = true_negative_curve(np.array(output_conf_all), np.array(label_validity_all))
    wiou_all = iou_meter_all.avg()
    atnr_all = np.mean(tn_rates_all)
    score_all = 0.7 * wiou_all + 0.3 * atnr_all
    print('================================')
    print(f'Average overall weighted IoU: {wiou_all:.4f}')
    print(f'Average average true negative rate: {atnr_all:.4f}')
    print(f'Average benchmark score: {score_all:.4f}')
    print('================================')
    return score

if __name__ == '__main__':
    dataset_path = '../dataset/public'
    subjects = ['S1', 'S5']
    benchmark_all(dataset_path, subjects)
