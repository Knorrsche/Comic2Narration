import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
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


def non_max_suppression(results: List[Tuple[int, int, Dict[str, float]]], window_size: Tuple[int, int],
                        iou_threshold: float = 0.1, solo_scaling_factor: float = 50.0) -> List[Tuple[int, int, Dict[str, float]]]:
    if not results:
        return []

    # Create bounding boxes from results
    boxes = [(x, y, window_size[0], window_size[1]) for (x, y, tags) in results]
    picked = []

    scores = []
    for (_, _, tags) in results:
        # Scale the 'solo' tag score
        solo_score = 0
        if(tags.get('solo', 0) ):
                solo_score = tags.get('solo', 0) * solo_scaling_factor  # Scale 'solo' confidence
        other_scores = [score for tag, score in tags.items() if tag != 'solo']  # Get scores of other tags

        # Calculate the total score
        total_score = solo_score + sum(other_scores)

        # Append the total score
        scores.append(total_score)

    # Sort indices by scores in descending order (higher score -> higher priority)
    indices = np.argsort(scores)[::-1]

    while len(indices) > 0:
        current_index = indices[0]
        picked.append(results[current_index])
        indices = indices[1:]

        current_box = boxes[current_index]
        iou = np.array([calculate_iou(current_box, boxes[i]) for i in indices])

        # Only keep indices whose IoU is below the threshold
        indices = indices[iou < iou_threshold]

    return picked



def draw_bounding_boxes(image: np.ndarray, results: List[Tuple[int, int, List[str]]],
                        window_size: Tuple[int, int]) -> np.ndarray:
    for (x, y, tags) in results:
        cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0),
                      2)
    return image


#IDEA: increase size. If maximum, look at the new max and the old max area and analyze it. If it contains no character info related to old, remove, or of it includes other characters
def expand_and_verify_box(box: Tuple[int, int, List[str]], image_path: str, model, initial_window_size: Tuple[int, int],
                          step_size: int = 30) -> Tuple[int, int, int, int, List[str]]:
    """
    Steadily scale up the bounding box in one direction at a time and verify if the tags remain consistent.
    If the new top 5 tags are within the top 10 tags of the original set, continue expanding.
    Otherwise, use the previous box as the final one.
    """
    x, y, tags = box
    width, height = initial_window_size
    full_image = cv2.imread(image_path)

    if full_image is None:
        print(f"Error: Could not read the full image at {image_path}. Returning original box.")
        return (x, y, width, height, tags)  # Return the original box

    previous_tags = tags

    def verify_tags(new_tags, previous_tags) -> bool:
        # Filter tags that have a confidence level of 0.6 or above
        new_high_conf_tags = {tag: score for tag, score in new_tags.items() if score >= 0.6}
        prev_high_conf_tags = {tag: score for tag, score in previous_tags.items() if score >= 0.6}

        # Ensure that the tag "solo" is not removed
        if "solo" in prev_high_conf_tags and "solo" not in new_high_conf_tags:
            print("Tag 'solo' was removed during expansion in this direction. Skipping further expansion in this direction.")
            return False

        # Count the number of new tags that are not in the previous set of high-confidence tags
        new_tag_count = len([tag for tag in new_high_conf_tags if tag not in prev_high_conf_tags])

        # If more than 2 new tags are found, stop expanding
        return new_tag_count <= 3

    # Define the possible directions for expansion
    directions = ['right', 'down', 'left', 'up']
    invalid_directions = set()  # Track directions that can no longer be expanded

    while True:
        expanded = False

        # Create a list of valid directions based on the current bounding box position
        valid_directions = []

        for direction in directions:
            if direction in invalid_directions:
                continue  # Skip directions that are no longer valid

            # Determine the new width and height based on the current direction
            current_x, current_y, current_width, current_height = x, y, width, height

            if direction == 'right':
                current_width += step_size
            elif direction == 'down':
                current_height += step_size
            elif direction == 'left':
                current_x -= step_size
                current_width += step_size
            elif direction == 'up':
                current_y -= step_size
                current_height += step_size

            # Check if the proposed bounding box stays within the image boundaries
            if (current_x >= 0 and
                    current_y >= 0 and
                    current_x + current_width <= full_image.shape[1] and
                    current_y + current_height <= full_image.shape[0]):
                valid_directions.append(direction)

        if not valid_directions:  # If no valid directions, break the loop
            print("No valid directions left for expansion. Exiting expansion loop.")
            break

        for direction in valid_directions:
            # Make a copy of current box parameters
            current_x, current_y, current_width, current_height = x, y, width, height

            # Expand in the valid direction
            if direction == 'right':
                current_width += step_size
            elif direction == 'down':
                current_height += step_size
            elif direction == 'left':
                current_x -= step_size
                current_width += step_size
            elif direction == 'up':
                current_y -= step_size
                current_height += step_size

            # Extract the expanded window
            expanded_window = full_image[current_y:current_y + current_height, current_x:current_x + current_width]

            # Write expanded window temporarily to disk
            window_image_path = Path(image_path).with_name(f"expanded_window_{current_x}_{current_y}.png")
            cv2.imwrite(str(window_image_path), expanded_window)

            # Recalculate tags
            expanded_tags = model.process_image(window_image_path)

            # Remove temporary file
            window_image_path.unlink(missing_ok=True)

            # Check if the new tags are consistent with the previous tags
            if verify_tags(expanded_tags, previous_tags):
                # Update the bounding box and tags if consistent
                x, y, width, height = current_x, current_y, current_width, current_height
                previous_tags = expanded_tags
                expanded = True
                print(f"Expanded box to ({x}, {y}, {width}, {height}) with new tags: {expanded_tags}")
            else:
                # Mark this direction as invalid for further expansion
                print(f"More than 2 new high-confidence tags or missing 'solo' tag found for box at ({current_x}, {current_y}) when expanding {direction}. Skipping this direction.")
                invalid_directions.add(direction)

        # If no expansion was successful in any direction, stop the process
        if not expanded:
            break

    # Return the final box parameters and tags
    return (x, y, width, height, previous_tags)




def draw_expanded_bounding_boxes(image: np.ndarray, results: List[Tuple[int, int, int, int, List[str]]]) -> np.ndarray:
    for result in results:
        if result is not None:
            x, y, width, height, tags = result
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return image


def main(image_path: str, model, step_size: int = 10, window_size: Tuple[int, int] = (64, 64)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}. Please check the path.")
        return

    original_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    step_size = max(1, image.shape[1] // 10)
    window_size = (image.shape[1] // 6, image.shape[0] // 3)
    windows = sliding_window(image, step_size, window_size)

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_window = {executor.submit(process_window, (x, y, image_path, window_size), model): (x, y) for (x, y)
                            in windows}
        for future in concurrent.futures.as_completed(future_to_window):
            result = future.result()
            if result:
                results.append(result)

    filtered_results = non_max_suppression(results, window_size, iou_threshold=0.1)

    # Now expand and verify each filtered result
    expanded_results = []
    for box in filtered_results:
        expanded_box = expand_and_verify_box(box, image_path, model, window_size)
        expanded_results.append(expanded_box)

    # Filter out any None values before drawing
    expanded_results = [res for res in expanded_results if res is not None]

    image_with_boxes = draw_expanded_bounding_boxes(original_image, expanded_results)

    output_path = Path(image_path).with_name(f"detected_{Path(image_path).name}")
    cv2.imwrite(str(output_path), image_with_boxes)

    for (x, y, width, height, tags) in expanded_results:
        print(f"Character found at position ({x}, {y}) with tags: {tags}")


if __name__ == "__main__":
    tagger = Tagger(model_name='vit', gen_threshold=0.35, char_threshold=0.75)

    image_path = r"C:\Users\derra\Desktop\images\tester2.png"

    main(image_path, tagger)
