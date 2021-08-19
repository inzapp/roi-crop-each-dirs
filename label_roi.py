import os
from glob import glob

import cv2

g_win_name = 'ROI setter v1.0 by Inzapp'
g_ratio_width = 576
g_ratio_height = 384
g_ratio = g_ratio_height / float(g_ratio_width)

g_thickness = 2
g_box_color = (0, 0, 255)


def draw_boxes(img):
    for box in g_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), g_box_color, g_thickness)


# def get_force_ratio_y2(x1, y1, x2, y2):
#     # global g_ratio, g_box_color, g_thickness
#     # width = x2 - x1
#     # height = int(width * g_ratio)  # force ratio
#     # y2 = y1 + height
#     return y2


# force ration between 1 : 1 ~ 16 : 9
def get_force_ratio_y2(x1, y1, x2, y2):
    # global g_ratio, g_box_color, g_thickness
    # width = x2 - x1
    # height = int(width * g_ratio)  # force ratio
    # y2 = y1 + height
    return y2


def mouse_callback(event, cur_x, cur_y, flag, _):
    global g_win_name, g_raw, g_x1, g_y1, g_x2, g_y2, g_boxes, g_box_color, g_thickness
    raw_copy = g_raw.copy()

    # start click
    if event == 1 and flag == 1:
        g_x1 = cur_x
        g_y1 = cur_y

    # while dragging
    elif event == 0 and flag == 1:
        g_x2 = cur_x
        g_y2 = cur_y

        if len(g_boxes) % 2 == 1:
            draw_boxes(raw_copy)
            g_x1 = cur_x
            g_y1 = g_boxes[-1][1]  # last box start y
            g_x2 = g_x1 + 576
            g_y2 = g_y1 + 192
            cv2.rectangle(raw_copy, (g_x1, g_y1), (g_x2, g_y2), g_box_color, g_thickness)
            cv2.imshow(g_win_name, raw_copy)
        else:
            x1, y1 = g_x1, g_y1
            # cur_y = get_force_ratio_y2(x1, y1, cur_x, cur_y)  # force ratio
            draw_boxes(raw_copy)
            cv2.rectangle(raw_copy, (g_x1, g_y1), (g_x2, g_y2), g_box_color, g_thickness)
            cv2.imshow(g_win_name, raw_copy)

    # end click
    elif event == 4 and flag == 0:
        # cur_y = get_force_ratio_y2(g_start_x, g_start_y, cur_x, cur_y)  # force ratio
        width = g_x2 - g_x1
        height = g_y2 - g_y1
        if width == 0 or height == 0:
            return
        g_boxes.append([g_x1, g_y1, g_x2, g_y2])  # x1, y1, x2, y2
        print(g_boxes)
        save()

    # right click
    elif event == 5 and flag == 0:
        if len(g_boxes) > 0:
            g_boxes.pop()
            draw_boxes(raw_copy)
            cv2.imshow(g_win_name, raw_copy)
        save()
        print(g_boxes)


def load_saved_boxes_if_exist(label_path):
    global g_width, g_height
    boxes = list()
    if os.path.exists(label_path) and os.path.isfile(label_path):
        with open(label_path, 'rt') as f:
            lines = f.readlines()
        for line in lines:
            x1, y1, x2, y2 = list(map(float, line.replace('\n', '').split()))
            x1 = int(x1 * g_width)
            x2 = int(x2 * g_width)
            y1 = int(y1 * g_height)
            y2 = int(y2 * g_height)
            boxes.append([x1, y1, x2, y2])
        return boxes
    else:
        return []


def save():
    global g_label_path, g_boxes, g_width, g_height
    label_str = ''
    for box in g_boxes:
        x1, y1, x2, y2 = box
        x1 /= g_width
        x2 /= g_width
        y1 /= g_height
        y2 /= g_height
        label_str += f'{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}\n'
    with open(g_label_path, 'wt') as f:
        f.writelines(label_str)


def get_img_paths():
    paths = []
    dir_paths = glob(f'*')
    for dir_path in dir_paths:
        if os.path.isdir(dir_path):
            jpg_paths = glob(f'{dir_path}/*.jpg')
            if len(jpg_paths) > 0:
                paths.append(jpg_paths[0])
    return paths


def get_label_path(file_path):
    sp = file_path.replace('\\', '/').split('/')
    label_path = ''
    for i in range(len(sp) - 1):
        label_path += sp[i]
        label_path += '/'
    label_path += 'roi.txt'
    return label_path


def resize_with_ratio(img):
    size = 1024
    height, width = img.shape[0], img.shape[1]
    if height < size and width < size:
        return img

    bigger = max(height, width)
    weight = size / float(bigger)
    img = cv2.resize(img, (0, 0), fx=weight, fy=weight)
    return img


img_paths = get_img_paths()
if len(img_paths) == 0:
    print('No image files in path.')
    exit(0)

index = 0
while True:
    g_file_path = img_paths[index]
    print(g_file_path)
    g_label_path = get_label_path(g_file_path)
    g_raw = cv2.imread(g_file_path, cv2.IMREAD_COLOR)
    # g_raw = resize_with_ratio(g_raw)
    g_raw_copy = g_raw.copy()
    g_height, g_width = g_raw.shape[0], g_raw.shape[1]
    g_boxes = load_saved_boxes_if_exist(g_label_path)
    if len(g_boxes) > 0:
        draw_boxes(g_raw_copy)
    g_x1, g_y1, g_x2, g_y2 = 0, 0, 0, 0
    cv2.namedWindow(g_win_name)
    cv2.imshow(g_win_name, g_raw_copy)
    cv2.setMouseCallback(g_win_name, mouse_callback)

    while True:
        res = cv2.waitKey(0)

        # go to next if input key was 'd'
        if res == ord('d'):
            if index == len(img_paths) - 1:
                print('Current image is last image')
            else:
                index += 1
                break

        # go to previous image if input key was 'a'
        elif res == ord('a'):
            if index == 0:
                print('Current image is first image')
            else:
                index -= 1
                break

        # exit if input key was ESC
        elif res == 27:
            exit(0)
