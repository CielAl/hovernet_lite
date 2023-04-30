from typing import TypedDict, Union, List

import numpy as np
import torch
from PIL import Image


class DatasetOut(TypedDict):
    img: Union[np.ndarray, Image.Image, torch.Tensor]
    prefix: str


class InstInfo(TypedDict):
    bbox: Union[np.ndarray, None]
    centroid: Union[np.ndarray, None]
    contour: Union[np.ndarray, None]
    type_prob: Union[float, None]
    type: Union[int, None]


class InstGeo(TypedDict):
    type: str
    coordinates: List[List[List[float]]]


class InstClass(TypedDict):
    name: str
    probability: float


class InstProperties(TypedDict):
    isLocked: str
    measurements: List
    classification: InstClass


class NucGeoData(TypedDict):
    type: str
    id: str
    tile_size: int
    geometry: InstGeo
    properties: InstProperties
