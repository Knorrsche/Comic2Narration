import cv2

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

        return image[y:y+h, x:x+w, :]

def is_bbox_overlapping(bbox_1, bbox_2):
    x1 = bbox_1['x']
    y1 = bbox_1['y']
    w1 = bbox_1['width']
    h1 = bbox_1['height']

    x2 = bbox_2['x']
    y2 = bbox_2['y']
    w2 = bbox_2['width']
    h2 = bbox_2['height']

    x1_left = x1 - w1 
    x1_right = x1 + w1 
    y1_top = y1 - h1 
    y1_bottom = y1 + h1 
    x2_left = x2 - w2 
    x2_right = x2 + w2 
    y2_top = y2 - h2 
    y2_bottom = y2 + h2 

    if (x1_right >= x2_left and x1_left <= x2_right and
        y1_bottom >= y2_top and y1_top <= y2_bottom):
        return True
    else:
        if (x1_right == x2_left or x1_left == x2_right) and (y1_bottom >= y2_top and y1_top <= y2_bottom):
            return True
        if (y1_bottom == y2_top or y1_top == y2_bottom) and (x1_right >= x2_left and x1_left <= x2_right):
            return True
        
        return False