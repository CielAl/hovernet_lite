import json
import logging
from typing import Callable, Set
import cv2
from skimage.util import img_as_ubyte
import PIL.Image
import cv2
import numpy as np
from scipy.ndimage import measurements, binary_fill_holes
from skimage.draw import polygon
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from typing import List
from hovernet_lite.data_type import InstClass, NucGeoData
import matplotlib.cm as cm
import os
import imageio
from .misc import save_json
import torch


gray_cmap: Callable[[np.ndarray], np.ndarray] = cm.get_cmap("gray")
IMG_TYPE_INST = 'inst'
IMG_TYPE_PROB = 'prob'
IMG_TYPE_BINARY = 'binary'
SET_VALID_IMG_TYPE: Set = {IMG_TYPE_INST, IMG_TYPE_PROB, IMG_TYPE_BINARY}


def processed_nuclei_pred(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred


def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def pixel_value(image_type: str, inst_class: InstClass, inst_id) -> np.uint8:
    assert image_type in SET_VALID_IMG_TYPE
    assert inst_class == IMG_TYPE_INST or 0. <= inst_class['probability'] <= 1., \
        f"If inst_class is IMG_TYPE_PROB, then the" \
        f" probability field value must be within 0. and 1." \
        f"Got {inst_class['probability']}"

    # only read from inst_class['probability'] if necessary, otherwise just return a 0 dummy value
    # into the value_dict.
    prob_val = 0 if inst_class == IMG_TYPE_INST else inst_class['probability'] * 255
    value_dict = {
        IMG_TYPE_INST: inst_id,
        IMG_TYPE_PROB: prob_val,
        IMG_TYPE_BINARY: 255,
    }
    return np.uint8(np.round(value_dict[image_type]))


def get_img_from_json_coords(tile_size, nuc_geo_list: List[NucGeoData], image_type: str, im=None):
    # data = json.loads(value)
    if im is None:
        im = np.zeros((tile_size, tile_size), 'uint8')
    for idx, nuc_geo in enumerate(nuc_geo_list, start=1):
        coords = nuc_geo['geometry']['coordinates']
        for c in coords:
            poly = np.array(c)
            rr, cc = polygon(poly[:, 0], poly[:, 1], im.shape)
            # im[rr,cc] = i
            pix_val = pixel_value(image_type, nuc_geo['properties']['classification'], idx)
            im[cc, rr] = pix_val
    return im


def to_gray_mask(inst_map: np.ndarray) -> np.ndarray:
    return gray_cmap(inst_map)[:, :, 0]


def save_image_on_flag(img: np.ndarray, save_flag: bool, export_folder, prefix: str, suffix=".png"):
    if not save_flag:
        return
    export_file_name = os.path.join(export_folder, f"{prefix}{suffix}")
    directory = os.path.dirname(export_file_name)
    os.makedirs(directory, exist_ok=True)
    # noinspection PyTypeChecker
    imageio.imwrite(export_file_name, img)


def save_json_on_flag(data, save_flag: bool, export_folder, prefix: str, suffix=".json"):
    if not save_flag:
        return
    export_file_name = os.path.join(export_folder, f"{prefix}{suffix}")
    directory = os.path.dirname(export_file_name)
    os.makedirs(directory, exist_ok=True)
    # with open(export_file_name, 'w') as root:
    #     json.dump(data, root, indent=4)
    save_json(export_file_name, data=data, indent=4)


def save_overlaid_tile_on_flag(tile: torch.Tensor,
                               mask: np.ndarray,
                               flag: bool,
                               transforms: Callable[[torch.Tensor], PIL.Image.Image],
                               export_folder: str,
                               prefix: str):
    """
    Export a tile for inspection if flag is set.
    Args:
        tile: individual tile
        mask: individual mask
        flag:
        transforms:
        export_folder:
        prefix:

    Returns:

    """
    if not flag:
        return
    mask_ubyte = img_as_ubyte(mask)
    tile_pil = transforms(tile).convert("RGB")
    # noinspection PyTypeChecker
    tile_array = np.array(tile_pil, copy=False)
    prob_map = cv2.applyColorMap(mask_ubyte, cv2.COLORMAP_JET)

    # Overlay the color map on the image using addWeighted()
    overlay = cv2.addWeighted(tile_array, 0.7, prob_map, 0.3, 0)
    save_image_on_flag(overlay, flag, export_folder, prefix, suffix='_overlaid.jpg')
