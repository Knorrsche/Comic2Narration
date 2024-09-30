import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple
import concurrent.futures
from wdv3_timm import Tagger


def sliding_window(image: np.ndarray, step_size: int, window_size: Tuple[int, int]) -> List[Tuple[int, int]]:
    windows = []
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            windows.append((x, y))
    return windows


def is_character(tags: dict, threshold: float = 0.91) -> bool:
    valid_tags = {tag: score for tag, score in tags.items() if score >= threshold}

    if len(valid_tags) > 0 and all(
            tag in ["no_humans", "english_text", "monochrome", "greyscale"] for tag in valid_tags.keys()):
        return False

    return len(valid_tags) > 0


def process_window(window_data: Tuple[int, int, str, Tuple[int, int]], model) -> Tuple[int, int, List[str]]:
    x, y, image_path, window_size = window_data
    full_image = cv2.imread(image_path)

    if full_image is None:
        print(f"Error: Could not read the full image at {image_path}.")
        return None

    window = full_image[y:y + window_size[1], x:x + window_size[0]]

    window_image_path = Path(image_path).with_name(f"window_{x}_{y}.png")
    cv2.imwrite(str(window_image_path), window)

    tags = model.process_image(window_image_path)

    window_image_path.unlink(missing_ok=True)

    if is_character(tags):
        print('Character found')
        return (x, y, tags)
    return None


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea) if boxAArea + boxBArea - interArea > 0 else 0
    return iou


def non_max_suppression(results: List[Tuple[int, int, List[str]]], window_size: Tuple[int, int],
                        iou_threshold: float = 0.1) -> List[Tuple[int, int, List[str]]]:
    if not results:
        return []

    boxes = [(x, y, window_size[0], window_size[1]) for (x, y, tags) in results]  # Dynamic size based on the window
    picked = []

    scores = []
    for (_, _, tags) in results:
        sorted_tags = sorted(tags.items(), key=lambda item: item[1], reverse=True)
        top_scores = [score for _, score in sorted_tags[:min(len(sorted_tags), 5)]]
        scores.append(sum(top_scores))

    indices = np.argsort(scores)[::-1]

    while len(indices) > 0:
        current_index = indices[0]
        picked.append(results[current_index])
        indices = indices[1:]

        current_box = boxes[current_index]
        ious = np.array([calculate_iou(current_box, boxes[i]) for i in indices])

        indices = indices[ious < iou_threshold]

    return picked


def draw_bounding_boxes(image: np.ndarray, results: List[Tuple[int, int, List[str]]],
                        window_size: Tuple[int, int]) -> np.ndarray:
    for (x, y, tags) in results:
        cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0),
                      2)
    return image


def main(image_path: str, model, step_size: int = 10, window_size: Tuple[int, int] = (64, 64)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}. Please check the path.")
        return

    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    step_size = max(1, image.shape[1] // 10)
    window_size = (image.shape[1] // 8, image.shape[0] // 4)
    windows = sliding_window(image, step_size, window_size)

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_window = {executor.submit(process_window, (x, y, image_path, window_size), model): (x, y) for (x, y)
                            in windows}
        for future in concurrent.futures.as_completed(future_to_window):
            result = future.result()
            if result:
                results.append(result)

    filtered_results = non_max_suppression(results, window_size, iou_threshold=0.05)

    image_with_boxes = draw_bounding_boxes(original_image, filtered_results, window_size)

    output_path = Path(image_path).with_name(f"detected_{Path(image_path).name}")
    cv2.imwrite(str(output_path), image_with_boxes)

    for (x, y, tags) in filtered_results:
        print(f"Character found at position ({x}, {y}) with tags: {tags}")


if __name__ == "__main__":
    tagger = Tagger(model_name='vit', gen_threshold=0.35, char_threshold=0.75)

    image_path = r"C:\Users\derra\Desktop\images\tester.png"  # Update this path accordingly

    main(image_path, tagger)
