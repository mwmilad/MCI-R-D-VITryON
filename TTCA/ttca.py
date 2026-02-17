import cv2
import numpy as np 


REFERENCE_PATH =  r"TTCA\sample\cloth\00318_00.jpg"
TEXTURE_PATH = r"TTCA\sample\texture_sample\texture2.jpg"
MASK_CLOTH = r"TTCA\sample\inshop_cloth_mask\00318_00.jpg"
OUT_PATH = r"TTCA"

# Read images
clothing = cv2.imread(REFERENCE_PATH)
texture = cv2.imread(TEXTURE_PATH)
mask = cv2.imread(MASK_CLOTH, cv2.IMREAD_GRAYSCALE)

h, w = clothing.shape[:2]
ph, pw = texture.shape[:2]
total_area = clothing.shape[1] * clothing.shape[0]
area = cv2.countNonZero(mask)
print(f"total_area / Area: {area / total_area}")
print(f"texture shape: width={pw}, height={ph}")



# Load your mask (grayscale, cloth region = white/non-zero)
mask = cv2.imread(MASK_CLOTH, cv2.IMREAD_GRAYSCALE)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Get the largest contour (assumes one main cloth object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle coordinates
    mx, my, mw, mh = cv2.boundingRect(largest_contour)
    
    
    # Draw rectangle on the image
    cv2.rectangle(clothing, (mx, my), (mx + mw, my + mh), (0, 255, 0), 2)
    
    # Print coordinates
    print(f"Rectangle coordinates: x={mx}, y={my}, width={mw}, height={mh}")
    print(f"Top-left: ({mx}, {my}), Bottom-right: ({mx + mw}, {my + mh})")
    
    # Display result
    cv2.imshow('Rectangle on Mask', clothing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found in the mask.")
cv2.imshow('main texture', texture)
texture = cv2.resize(texture, (int(mh / 5), int(mw/5)), cv2.INTER_CUBIC)
cv2.imshow("texture", texture)
cv2.waitKey(0)

# tiled_pattern = np.zeros_like(clothing)
# for y in range(0, h, ph):
#     for x in range(0, w, pw):
#         tiled_pattern[y:y+pw, x:x+pw] = texture[:min(ph, h-y), :min(pw, w-x)]

# result = clothing.copy()
# result[mask==255] = tiled_pattern[mask==255]

# print(type(result))
# cv2.imshow('output', result)
# cv2.waitKey(0)

# cv2.imshow('image', reference_image)
# cv2.imshow('texture', texture_image)
# cv2.imshow('mask', mask_image)
# cv2.waitKey()

