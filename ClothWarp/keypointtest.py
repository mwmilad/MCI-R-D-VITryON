import json
import numpy as np
from PIL import Image, ImageDraw

# ---- Load JSON ----
with open(r"data\zolando-hd-resized\train\openpose_json\00000_00_keypoints.json") as f:
    data = json.load(f)

keypoints = data["people"][0]["pose_keypoints_2d"]
keypoints = np.array(keypoints).reshape(-1, 3)
print(keypoints.shape)

# ---- Image size (set this to your real image size) ----
width = 768
height = 1024

# ---- Create black mask ----
mask = Image.new("L", (width, height), 0)  # "L" = grayscale
draw = ImageDraw.Draw(mask)

# ---- Draw white circles ----
radius = 10
for x, y, conf in keypoints:
    if conf > 0.3:  # confidence threshold
        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        draw.ellipse([left_up, right_down], fill=255)

# ---- Show mask ----
mask.show()