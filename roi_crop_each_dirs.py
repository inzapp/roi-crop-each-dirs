import os
from glob import glob

import cv2
import numpy as np

g_raw = None


def iou(a, b):
    a_x_min, a_y_min, a_x_max, a_y_max = a
    b_x_min, b_y_min, b_x_max, b_y_max = b
    intersection_width = min(a_x_max, b_x_max) - max(a_x_min, b_x_min)
    intersection_height = min(a_y_max, b_y_max) - max(a_y_min, b_y_min)
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    a_area = abs((a_x_max - a_x_min) * (a_y_max - a_y_min))
    b_area = abs((b_x_max - b_x_min) * (b_y_max - b_y_min))
    union_area = a_area + b_area - intersection_area
    return intersection_area / (float(union_area) + 1e-5)


def to_x1_y1_x2_y2(box):
    cx, cy, w, h = box
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return [x1, y1, x2, y2]


def to_cx_cy_w_h(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    return [cx, cy, w, h]


def draw_rect(img, box, color):
    x1, y1, x2, y2 = box
    img_height, img_width = img.shape[0], img.shape[1]
    x1 = int(x1 * img_width)
    x2 = int(x2 * img_width)
    y1 = int(y1 * img_height)
    y2 = int(y2 * img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)

    # cv2.imshow('raw', g_raw)
    # key = cv2.waitKey(0)
    # if key == 27:
    #     exit(0)


def convert_to_origin_box(roi_box, roi):
    cx, cy, w, h = roi_box
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    cx = cx * roi_w + roi_x1
    cy = cy * roi_h + roi_y1
    w = w * roi_w
    h = h * roi_h
    return [cx, cy, w, h]


def is_iou_over(origin_box, origin_converted_roi_box, iou_threshold):
    origin_box = to_x1_y1_x2_y2(origin_box)
    origin_converted_roi_box = to_x1_y1_x2_y2(origin_converted_roi_box)
    return iou(origin_box, origin_converted_roi_box) > iou_threshold


def convert_to_roi_box(origin_box, roi):
    cx, cy, w, h = origin_box
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    cx = (cx - roi_x1) / roi_w
    cy = (cy - roi_y1) / roi_h
    w = w / roi_w
    h = h / roi_h
    return [cx, cy, w, h]


def crop_roi_img(img, roi):
    raw_height, raw_width = img.shape[0], img.shape[1]
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    roi_x1_s32 = int(roi_x1 * raw_width)
    roi_x2_s32 = int(roi_x2 * raw_width)
    roi_y1_s32 = int(roi_y1 * raw_height)
    roi_y2_s32 = int(roi_y2 * raw_height)

    img = img[roi_y1_s32:roi_y2_s32, roi_x1_s32:roi_x2_s32]
    if img.shape[0] == 0 or img.shape[1] == 0:
        return None
    return img


def roi_crop_with_label_convert(path, roi, index):
    global g_raw
    label_path = f'{path[:-4]}.txt'
    if not os.path.exists(label_path):
        print(f'label not exist : {label_path}')
        return

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    g_raw = img.copy()

    roi_img = crop_roi_img(img, roi)
    if roi_img is None:
        return

    with open(label_path, 'rt') as f:
        lines = f.readlines()

    roi_label_content = ''
    for line in lines:
        class_index, cx, cy, w, h = list(map(float, line.replace('\n', '').split()))
        class_index = int(class_index)
        origin_box = [cx, cy, w, h]

        roi_box = convert_to_roi_box(origin_box, roi)
        roi_box = to_x1_y1_x2_y2(roi_box)
        roi_box = np.clip(np.array(roi_box), 0.0, 1.0)
        roi_box = to_cx_cy_w_h(roi_box)
        cx, cy, w, h = roi_box
        if w < 0.001 or h < 0.001:
            continue

        origin_converted_roi_box = convert_to_origin_box(roi_box, roi)
        if not is_iou_over(origin_box, origin_converted_roi_box, 0.5):
            continue

        roi_label_content += f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'

    roi_img_path = f'{path[:-4]}_roi_cropped_{index}.jpg'
    roi_label_path = f'{label_path[:-4]}_roi_cropped_{index}.txt'
    cv2.imwrite(roi_img_path, roi_img)
    with open(roi_label_path, 'wt') as f:
        f.writelines(roi_label_content)

    print(f'save success ===> {roi_img_path}')


# split roi big : 2/3, small : 1/3
def get_rois(dir_path):
    roi_file_path = f'{dir_path}/roi.txt'
    if not (os.path.exists(roi_file_path) and os.path.isfile(roi_file_path)):
        return None

    with open(roi_file_path, 'rt') as f:
        lines = f.readlines()

    x1, y1, x2, y2 = list(map(float, lines[0].replace('\n', '').split()))

    height = y2 - y1
    big_y1 = y1 + height * 0.2891
    small_y2 = y1 + height * 0.2891

    rois = []
    rois.append([x1, big_y1, x2, y2])
    rois.append([x1, y1, x2, small_y2])
    return rois


def main():
    for cur_dir_path in glob('*'):
        if not os.path.isdir(cur_dir_path):
            continue

        rois = get_rois(cur_dir_path)
        for img_path in glob(rf'{cur_dir_path}/*.jpg'):
            for i, cur_roi in enumerate(rois):
                roi_crop_with_label_convert(img_path, cur_roi, i)


if __name__ == '__main__':
    main()
