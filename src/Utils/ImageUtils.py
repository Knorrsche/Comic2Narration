import uuid
from PIL import Image

import cv2


# TODO: maybe opencv uses BGR instead of RGB
import cv2


def draw_bounding_box(image, bbox, color, thickness=2, number=None, type=None):
    x = int(bbox['x'])
    y = int(bbox['y'])
    w = int(bbox['width'])
    h = int(bbox['height'])

    start_point = (x, y)
    end_point = (x + w, y + h)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    font_scale = 2
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_text_with_background(image, text, x, y):
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_w, text_h = text_size

        centered_x = x - text_w // 2
        background_rect_x1 = centered_x - 2
        background_rect_y1 = y - text_h - 2
        background_rect_x2 = centered_x + text_w + 2
        background_rect_y2 = y + 2

        image = cv2.rectangle(image, (background_rect_x1, background_rect_y1), (background_rect_x2, background_rect_y2),
                              (255, 255, 255), -1)
        image = cv2.putText(image, text, (centered_x, y), font, font_scale, color, font_thickness, cv2.LINE_AA)

        return image

    if number is not None:
        text_x_n = x + w // 2
        text_y_n = y + h - 10
        image = draw_text_with_background(image, str(number), text_x_n, text_y_n)

    if type is not None:
        text_x_t = x + w // 2
        text_y_t = y + h - 30
        image = draw_text_with_background(image, type, text_x_t, text_y_t)

    return image
def calculate_overlap_percentage(box_a, box_b):
        x_min_a = box_a['x']
        y_min_a = box_a['y']
        x_max_a = box_a['x']+box_a['width']
        y_max_a = box_a['y']+box_a['height']

        x_min_b = box_b['x']
        y_min_b = box_b['y']
        x_max_b = box_b['x']+box_b['width']
        y_max_b = box_b['y']+box_b['height']

        x_min_inter = max(x_min_a, x_min_b)
        y_min_inter = max(y_min_a, y_min_b)
        x_max_inter = min(x_max_a, x_max_b)
        y_max_inter = min(y_max_a, y_max_b)

        if x_min_inter < x_max_inter and y_min_inter < y_max_inter:
            intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

            area_a = (x_max_a - x_min_a) * (y_max_a - y_min_a)

            overlap_percentage = (intersection_area / area_a)
            return overlap_percentage
        else:
            return 0.0

def calculate_iou(bbox1, bbox2):
    x1_start = bbox1['x']
    y1_start = bbox1['y']
    x1_end = x1_start + bbox1['width']
    y1_end = y1_start + bbox1['height']

    x2_start = bbox2['x']
    y2_start = bbox2['y']
    x2_end = x2_start + bbox2['width']
    y2_end = y2_start + bbox2['height']

    x_inter_start = max(x1_start, x2_start)
    y_inter_start = max(y1_start, y2_start)
    x_inter_end = min(x1_end, x2_end)
    y_inter_end = min(y1_end, y2_end)

    inter_width = max(0, x_inter_end - x_inter_start)
    inter_height = max(0, y_inter_end - y_inter_start)
    intersection_area = inter_width * inter_height

    bbox1_area = bbox1['width'] * bbox1['height']
    bbox2_area = bbox2['width'] * bbox2['height']

    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def image_from_bbox(image, bbox):

    x = int(bbox['x'])
    y = int(bbox['y'])
    w = int(bbox['width'])
    h = int(bbox['height'])

    extracted_image = image[y:y+h, x:x+w, :]

    return extracted_image

def is_bbox_overlapping(bbox_1, bbox_2):
    x1_left = bbox_1['x'] - bbox_1['width']
    x1_right = bbox_1['x'] + bbox_1['width']
    y1_top = bbox_1['y'] - bbox_1['height']
    y1_bottom = bbox_1['y'] + bbox_1['height']

    x2_left = bbox_2['x'] - bbox_2['width']
    x2_right = bbox_2['x'] + bbox_2['width']
    y2_top = bbox_2['y'] - bbox_2['height']
    y2_bottom = bbox_2['y'] + bbox_2['height']

    overlap = (x2_left < x1_right and x2_right > x1_left and
               y2_top < y1_bottom and y2_bottom > y1_top)

    contains = (x1_left <= x2_left and x1_right >= x2_right and
                y1_top <= y2_top and y1_bottom >= y2_bottom)

    return overlap or contains
