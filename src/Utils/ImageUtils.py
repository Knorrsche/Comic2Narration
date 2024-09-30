import cv2


# TODO: maybe opencv uses BGR instead of RGB
import cv2


def draw_bounding_box(image, bbox, color, thickness=2, number=None):
    if number is None:
        number = 0

    x = int(bbox['x'] - bbox['width'] / 2)
    y = int(bbox['y'] - bbox['height'] / 2)
    w = int(bbox['width'])
    h = int(bbox['height'])
    start_point = (x, y)
    end_point = (x + w, y + h)

    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    if number is not None:
        font_scale = 5
        font_thickness = 3
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(number)
        while True:
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_w, text_h = text_size

            if text_w > w - 10 or text_h > h - 10:
                font_scale -= 0.1
                if font_scale < 0.1:
                    break
            else:
                break

        text_x = end_point[0] - int(text_w) - 5
        text_y = end_point[1] - 5

        background_rect_x1 = text_x - 2
        background_rect_y1 = text_y - text_h - 2
        background_rect_x2 = text_x + text_w + 2
        background_rect_y2 = text_y + 2

        image = cv2.rectangle(image, (background_rect_x1, background_rect_y1), (background_rect_x2, background_rect_y2), (255, 255, 255), -1)

        image = cv2.putText(image, text, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

    return image


def calculate_iou(bbox1, bbox2):
    x1_start = bbox1['x'] - bbox1['width'] / 2
    y1_start = bbox1['y'] - bbox1['height'] / 2
    x1_end = x1_start + bbox1['width']
    y1_end = y1_start + bbox1['height']

    x2_start = bbox2['x'] - bbox2['width'] / 2
    y2_start = bbox2['y'] - bbox2['height'] / 2
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
    x = int(bbox['x'] - bbox['width'] / 2)
    y = int(bbox['y'] - bbox['height'] / 2)
    w = int(bbox['width'])
    h = int(bbox['height'])

    return image[y:y + h, x:x + w, :]


def is_bbox_overlapping(bbox_1, bbox_2):
    x1_left = bbox_1['x'] - bbox_1['width'] / 2
    x1_right = bbox_1['x'] + bbox_1['width'] / 2
    y1_top = bbox_1['y'] - bbox_1['height'] / 2
    y1_bottom = bbox_1['y'] + bbox_1['height'] / 2

    x2_left = bbox_2['x'] - bbox_2['width'] / 2
    x2_right = bbox_2['x'] + bbox_2['width'] / 2
    y2_top = bbox_2['y'] - bbox_2['height'] / 2
    y2_bottom = bbox_2['y'] + bbox_2['height'] / 2

    if (x2_left <= x1_right and x2_right >= x1_left) and (y2_top <= y1_bottom and y2_bottom >= y1_top):
        return True
    else:
        return False
