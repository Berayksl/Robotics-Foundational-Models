import torch
import clip
from PIL import Image
import os
import time
from torchmetrics.multimodal.clip_score import CLIPScore
import cv2
import numpy as np


def clip_similarity_score(model, preprocess, prompts, image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    pil_image = Image.fromarray(image).convert("RGB")
    #image = load_resized(image_path)
    # cv2.imshow("Image", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    # cv2.waitKey(500)
    image_input = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)

    return similarity.max().item()
