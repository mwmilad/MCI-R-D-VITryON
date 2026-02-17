import cv2
import numpy as np
import math

def read_images(texture_path, cloth_path, mask_path):
    tex = cv2.imread(texture_path, cv2.IMREAD_COLOR)
    cloth = cv2.imread(cloth_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if tex is None or cloth is None or mask is None:
        raise FileNotFoundError("One of the input files wasn't found or couldn't be read.")
    # Ensure mask is binary 0/255
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return tex, cloth, mask

def make_tiled_texture(texture, target_w, target_h, tile_scale=1.0, rotation_deg=0):
    # Resize texture by tile_scale, but do NOT stretch to target size.
    th, tw = texture.shape[:2]
    new_w = max(1, int(tw * tile_scale))
    new_h = max(1, int(th * tile_scale))
    tex_resized = cv2.resize(texture, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if rotation_deg != 0:
        # rotate around center
        M = cv2.getRotationMatrix2D((new_w/2, new_h/2), rotation_deg, 1.0)
        tex_resized = cv2.warpAffine(tex_resized, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Tile it enough to cover target_w x target_h
    reps_w = math.ceil(target_w / new_w)
    reps_h = math.ceil(target_h / new_h)
    tiled = np.tile(tex_resized, (reps_h, reps_w, 1))
    tiled = tiled[:target_h, :target_w].copy()
    return tiled

def apply_texture_keep_luminance(cloth, tiled_patch, mask_patch, x, y):
    """
    Replace color/chroma of cloth in mask with texture, but keep L channel from cloth so wrinkles remain.
    mask_patch is the mask cropped to the patch size (0/255).
    """
    # Convert to LAB
    lab_cloth = cv2.cvtColor(cloth, cv2.COLOR_BGR2LAB)
    lab_tex = cv2.cvtColor(tiled_patch, cv2.COLOR_BGR2LAB)

    y1, y2 = y, y + lab_tex.shape[0]
    x1, x2 = x, x + lab_tex.shape[1]

    # Operate only where mask_patch == 255
    mask_bool = (mask_patch == 255)

    # For pixels in mask: keep L from cloth, take a,b from texture
    # Copy a,b channels from texture into cloth in the mask area
    # Note shapes: lab_cloth[y1:y2, x1:x2, 1:3] and lab_tex[:,:,1:3]
    dest_ab = lab_cloth[y1:y2, x1:x2, 1:3].astype(np.int16)
    src_ab = lab_tex[:, :, 1:3].astype(np.int16)

    # Only replace where mask==255
    for c in range(2):
        ch = dest_ab[:, :, c]
        ch[mask_bool] = src_ab[:, :, c][mask_bool]
        dest_ab[:, :, c] = ch

    lab_cloth[y1:y2, x1:x2, 1:3] = dest_ab.astype(np.uint8)
    result = cv2.cvtColor(lab_cloth, cv2.COLOR_LAB2BGR)
    return result

def apply_texture_seamless_clone(cloth, tiled_patch, mask_patch, x, y, clone_mode=cv2.NORMAL_CLONE):
    # create source image same size as cloth, place tiled_patch at (x,y)
    src = np.zeros_like(cloth)
    h, w = tiled_patch.shape[:2]
    src[y:y+h, x:x+w] = tiled_patch

    # mask must be same size as src, and 3-channel? seamlessClone expects single-channel mask
    big_mask = np.zeros((cloth.shape[0], cloth.shape[1]), dtype=np.uint8)
    big_mask[y:y+h, x:x+w] = mask_patch

    center = (x + w // 2, y + h // 2)
    # seamlessClone will try to blend gradients (Poisson)
    cloned = cv2.seamlessClone(src, cloth, big_mask, center, clone_mode)
    return cloned

def texture_on_cloth(texture_path, cloth_path, mask_path,
                     tile_scale=1.0, rotation_deg=0,
                     blend_method='lab', use_seamless_mode=cv2.NORMAL_CLONE):
    """
    blend_method: 'lab' to keep luminance, 'seamless' to use seamlessClone, 'alpha' to simple alpha blend
    """
    tex, cloth, mask = read_images(texture_path, cloth_path, mask_path)
    # bounding rect of mask
    x, y, w, h = cv2.boundingRect(mask)
    # make tiled patch sized to the bounding box (or a bit larger if you like)
    tiled = make_tiled_texture(tex, w, h, tile_scale=tile_scale, rotation_deg=rotation_deg)
    mask_patch = mask[y:y+h, x:x+w]

    if blend_method == 'lab':
        result = apply_texture_keep_luminance(cloth, tiled, mask_patch, x, y)
    elif blend_method == 'seamless':
        result = apply_texture_seamless_clone(cloth, tiled, mask_patch, x, y, clone_mode=use_seamless_mode)
    elif blend_method == 'alpha':
        # simple alpha blend: places texture over cloth, preserving original outside mask
        result = cloth.copy()
        patch = tiled.copy()
        # ensure mask is 3-channel for direct indexing
        mask3 = cv2.merge([mask_patch, mask_patch, mask_patch])
        inv = cv2.bitwise_not(mask3)
        result[y:y+h, x:x+w] = (result[y:y+h, x:x+w] & (inv)) | (patch & mask3)
    else:
        raise ValueError("Unknown blend_method")

    return result

if __name__ == "__main__":
    # === USER PARAMETERS ===
    texture_path = r"C:\PythonProject\MCI-R-D-VITryON\TTCA\sample\texture_sample\texture3.jpg"
    cloth_path   = r"TTCA\sample\cloth\00049_00.jpg"
    mask_path    = r"TTCA\sample\inshop_cloth_mask\00049_00.jpg"
    tile_scale   = 0.5   # try 0.5, 1.0, 2.0 to change tile size
    rotation_deg = 0
    blend_method = 'lab' # 'lab' (recommended), 'seamless', or 'alpha'
    # =======================

    out = texture_on_cloth(texture_path, cloth_path, mask_path,
                           tile_scale=tile_scale,
                           rotation_deg=rotation_deg,
                           blend_method=blend_method)
    cv2.imwrite("cloth_with_texture.png", out)
    print("Saved as cloth_with_texture.png")