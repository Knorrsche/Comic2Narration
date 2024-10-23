from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from PIL import Image
from simple_parsing import field, parse_known_args
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    px = max(image.size)
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels_hf(
        repo_id: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == 9)[0]),
        general=list(np.where(df["category"] == 0)[0]),
        character=list(np.where(df["category"] == 4)[0]),
    )

    return tag_data


def get_tags(
        probs: Tensor,
        labels: LabelData,
        gen_threshold: float,
        char_threshold: float,
):
    probs = list(zip(labels.names, probs.numpy()))

    rating_labels = dict([probs[i] for i in labels.rating])

    gen_labels = [probs[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    char_labels = [probs[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ").replace("(", "\\(").replace(")", "\\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


@dataclass
class ScriptOptions:
    image_file: Path = field(positional=True)
    model: str = field(default="vit")
    gen_threshold: float = field(default=0.35)
    char_threshold: float = field(default=0.75)


class Tagger:
    def __init__(self, model_name: str, gen_threshold: float, char_threshold: float):
        self.model_name = model_name
        self.gen_threshold = gen_threshold
        self.char_threshold = char_threshold
        self.repo_id = MODEL_REPO_MAP.get(self.model_name)

        print(f"Loading model '{self.model_name}' from '{self.repo_id}'...")
        self.model: nn.Module = timm.create_model("hf-hub:" + self.repo_id).eval()
        state_dict = timm.models.load_state_dict_from_hf(self.repo_id)
        self.model.load_state_dict(state_dict)

        print("Loading tag list...")
        self.labels: LabelData = load_labels_hf(repo_id=self.repo_id)

        print("Creating data transform...")
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def process_image(self, image_path: Path):
        print(f"\nProcessing image: {image_path}")
        #if not image_path.is_file():
        #    print(f"Image file not found: {image_path}")
        #    return

        #print("Loading image and preprocessing...")
        img_input: Image.Image = Image.open(image_path)
        img_input = pil_ensure_rgb(img_input)
        img_input = pil_pad_square(img_input)
        inputs: Tensor = self.transform(img_input).unsqueeze(0)
        inputs = inputs[:, [2, 1, 0]]

        #print("Running inference...")
        with torch.inference_mode():
            if torch_device.type != "cpu":
                self.model = self.model.to(torch_device)
                inputs = inputs.to(torch_device)

            outputs = self.model.forward(inputs)
            outputs = F.sigmoid(outputs)

            if torch_device.type != "cpu":
                inputs = inputs.to("cpu")
                outputs = outputs.to("cpu")
                self.model = self.model.to("cpu")

        #print("Processing results...")
        caption, taglist, ratings, character, general = get_tags(
            probs=outputs.squeeze(0),
            labels=self.labels,
            gen_threshold=self.gen_threshold,
            char_threshold=self.char_threshold,
        )

        #self.display_results(caption, taglist, ratings, character, general)
        return general

    def display_results(self, caption, taglist, ratings, character, general):
        print("--------")
        print(f"Caption: {caption}")
        print("--------")
        print(f"Tags: {taglist}")

        print("--------")
        print("Ratings:")
        for k, v in ratings.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"Character tags (threshold={self.char_threshold}):")
        for k, v in character.items():
            print(f"  {k}: {v:.3f}")

        print("--------")
        print(f"General tags (threshold={self.gen_threshold}):")
        for k, v in general.items():
            print(f"  {k}: {v:.3f}")

        print("Done!")


def main(image_dir: str, model_name: str = "vit", gen_threshold: float = 0.35, char_threshold: float = 0.75):
    tagger = Tagger(model_name, gen_threshold, char_threshold)

    image_paths = list(Path(image_dir).glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No image files found in directory: {image_dir}")

    for image_path in image_paths:
        tagger.process_image(image_path)


if __name__ == "__main__":
    opts, _ = parse_known_args(ScriptOptions)
    if opts.model not in MODEL_REPO_MAP:
        print(f"Available models: {list(MODEL_REPO_MAP.keys())}")
        raise ValueError(f"Unknown model name '{opts.model}'")

    main(str(Path(r'C:\Users\derra\Desktop\tagger_cluster\easy')), opts.model, opts.gen_threshold, opts.char_threshold)
