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


def multiply_width_height(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(x1 * width)
    x2 = int(x2 * width)
    y1 = int(y1 * height)
    y2 = int(y2 * height)
    return [x1, y1, x2, y2]


def draw_rect(img, box, color):
    img_height, img_width = img.shape[0], img.shape[1]
    x1, y1, x2, y2 = multiply_width_height(box, img_width, img_height)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)


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


def remove_object_in_image(img, box):
    height, width = img.shape[0], img.shape[1]
    box = to_x1_y1_x2_y2(box)
    box = multiply_width_height(box, width, height)
    x1, y1, x2, y2 = box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
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
            roi_img = remove_object_in_image(roi_img, roi_box)
            continue

        roi_label_content += f'{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n'

    roi_img_path = f'{path[:-4]}_roi_cropped_{index}.jpg'
    roi_label_path = f'{label_path[:-4]}_roi_cropped_{index}.txt'
    cv2.imwrite(roi_img_path, roi_img)
    with open(roi_label_path, 'wt') as f:
        f.writelines(roi_label_content)

    print(f'save success ===> {roi_img_path}')


def test(dir_path, big_roi, small_roi):
    big_x1, big_y1, big_x2, big_y2 = big_roi
    small_x1, small_y1, small_x2, small_y2 = small_roi
    for path in glob(f'{dir_path}/*.jpg'):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        height, width = img.shape[0], img.shape[1]

        b_x1 = int(big_x1 * width)
        b_y1 = int(big_y1 * height)
        b_x2 = int(big_x2 * width)
        b_y2 = int(big_y2 * height)

        s_x1 = int(small_x1 * width)
        s_y1 = int(small_y1 * height)
        s_x2 = int(small_x2 * width)
        s_y2 = int(small_y2 * height)

        cv2.rectangle(img, (b_x1, b_y1), (b_x2, b_y2), (0, 0, 255), thickness=2)
        cv2.rectangle(img, (s_x1, s_y1), (s_x2, s_y2), (0, 255, 0), thickness=2)

        cv2.imshow('img', img)
        key = cv2.waitKey(0)
        if key == 27:
            exit(0)


# [[big], [small]]
def get_rois(dir_path):
    roi_file_path = f'{dir_path}/roi.txt'
    if not (os.path.exists(roi_file_path) and os.path.isfile(roi_file_path)):
        return None

    with open(roi_file_path, 'rt') as f:
        lines = f.readlines()

    big_x1, big_y1, big_x2, big_y2 = list(map(float, lines[0].replace('\n', '').split()))
    small_x1, small_y1, small_x2, small_y2 = list(map(float, lines[1].replace('\n', '').split()))

    # big_x1 = big_x1
    # big_y1 = big_y1 + (small_y2 - small_y1)
    # big_x2 = big_x2
    # big_y2 = big_y2

    # test(dir_path, [big_x1, big_y1, big_x2, big_y2], [small_x1, small_y1, small_x2, small_y2])

    rois = []
    rois.append([big_x1, big_y1, big_x2, big_y2])
    rois.append([small_x1, small_y1, small_x2, small_y2])
    return rois


def main():
    for cur_dir_path in glob('*'):
        if os.path.isdir(cur_dir_path):
            rois = get_rois(cur_dir_path)
            for img_path in glob(rf'{cur_dir_path}/*.jpg'):
                for i, cur_roi in enumerate(rois):
                    roi_crop_with_label_convert(img_path, cur_roi, i)


if __name__ == '__main__':
    main()
