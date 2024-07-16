import cv2


# TODO: maybe opencv uses BGR instead of RGB
def draw_bounding_box(image, bbox, color, thickness=2):
    x = int(bbox['x'] - bbox['width'] / 2)
    y = int(bbox['y'] - bbox['height'] / 2)
    w = int(bbox['width'])
    h = int(bbox['height'])
    start_point = (x, y)
    end_point = (x + w, y + h)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


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
