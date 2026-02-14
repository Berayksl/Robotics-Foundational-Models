import torch
import clip
from PIL import Image
import os
import time
from torchmetrics.multimodal.clip_score import CLIPScore
import cv2
import numpy as np


start_time = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()
end_time = time.time()
print(f"Loaded CLIP model in {end_time - start_time:.2f} seconds.")

tv_prompts = ["a hallway leading to a kitchen"
"a corridor leading to a kitchen"
"a passageway to a kitchen"
"a doorway leading to a kitchen"
"an entrance to a kitchen",
"a kitchen"
    
#     "a frying pan in a kitchen",
# "a pan on a stove",
# "a metal cooking pan",
# "a pan on a countertop",

#     "a television",
#     "a TV",
#     "a flat screen television",
#     "a television screen",
#     "a flat screen TV", 
#     "a black television screen",
#     "a turned off television"
]


start_time = time.time()

text_tokens = clip.tokenize(tv_prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def load_resized(image_path, size=(768, 448)):
    image = Image.open(image_path).convert("RGB")
    return image.resize(size, Image.BICUBIC)

def clip_tv_score(image_path):
    #image = Image.open(image_path).convert("RGB")
    image = Image.open(image_path)
    #image = load_resized(image_path)
    # cv2.imshow("Image", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
    # cv2.waitKey(500)
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)

    return similarity.max().item()

# def clip_tv_score_verbose(image_path):
#     image = Image.open(image_path).convert("RGB")

#     image_input = preprocess(image).unsqueeze(0).to(device)

#     with torch.no_grad():
#         image_features = model.encode_image(image_input)
#         image_features /= image_features.norm(dim=-1, keepdim=True)
#         similarity = (image_features @ text_features.T).squeeze(0)

#     best_idx = similarity.argmax().item()
#     return similarity[best_idx].item(), tv_prompts[best_idx]



image_dir = "/home/bera/Pictures/SPOC screenshots/find a pan"

for fname in sorted(os.listdir(image_dir)):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(image_dir, fname)
        score = clip_tv_score(path)
        print(f"{fname}: CLIP similarity score = {score:.3f}")

end_time = time.time()
print(f"Processed {len(os.listdir(image_dir))} images in {end_time - start_time:.2f} seconds.")


# score, prompt = clip_tv_score_verbose(path)
# print(f"{fname}: {score:.3f} (matched: '{prompt}')")