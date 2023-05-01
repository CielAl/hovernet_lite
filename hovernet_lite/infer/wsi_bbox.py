"""
Read tile regions from WSI using json that records the list of tile bbox (left, top, right, bottom)
--> List[List[Tuple[int, int, int, int]] nested by ROIs (nest_level = 1)
We assume that [WSI_FILENAME_WITH_EXTENSION][BBOX_SUFFIX] = BBOX Filename
"""
from hovernet_lite.infer._helper import main_process
from hovernet_lite._import_openslide import openslide
from hovernet_lite.util.misc import load_json, List
from hovernet_lite.logger import get_logger
from hovernet_lite.args import BaseArgs
from hovernet_lite.infer_manager.dataset_proto import SimpleSeqDataset, pre_processor
import sys
import glob
import os
from typing import Tuple, Dict, Union
import itertools
from functools import partial
####


class BBoxArgs(BaseArgs):
    """
    --data_pattern is the WSI location
    """
    def additional_args(self):
        self.parser.add_argument("--bbox_pattern", help="input wildcard for bbox coord files", type=str, required=True)
        self.parser.add_argument("--bbox_suffix", help="suffix of bbox. ", type=str,
                                 default="_bbox.json")
        self.parser.add_argument("--tile_size", help="Optional. Define the size of bounding box before resize"
                                                     " in case the"
                                                     " bbox is cutoff at the boundary of images which causes"
                                                     " inconsistent tile size in batches. "
                                                     "If not set then use the bbox's size itself",
                                 type=int, required=False)
        self.parser.add_argument("--nest_level", help="Nest level of the bbox list."
                                                      "0 if the json contains List[Tuple[left, top, right, bottom]],"
                                                      "1 if List[List[Tuple[left, top, right, bottom]], etc. ",
                                 type=int, required=True)


def window_convert(window: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """TO OpenSlide style (left, top) + (width, height)
    Args:
        window:
    Returns:
    """
    left, top, right, bottom = window

    left = int(left)
    top = int(top)
    right = int(right)
    bottom = int(bottom)

    location = (left, top)
    width = right - left
    height = bottom - top
    size = (width, height)
    return location, size


def bbox_flatten(bbox_list: List, nest_level):
    # bbox_list: List = load_json(file)
    assert isinstance(bbox_list, List)
    repeat = nest_level
    while repeat > 0:
        bbox_list = sum(bbox_list, [])
        repeat -= 1
    return bbox_list


def bbox_from_json(file, nest_level):
    bbox_list: List = load_json(file)
    return bbox_flatten(bbox_list, nest_level)


def bbox_img_loader(uri: Tuple[str, Tuple[int, int, int, int]], tile_size: Union[int, None]):
    """
    For dataset
    Args:
        uri: A pair of (wsi file location, bounding box from json in top, left, right, bottom format)
        tile_size: gauge the size in the window. Use partial(bbox_img_loader, min_tile_size=xxx). No effect if set
            to None
    Returns:

    """
    wsi_name, bbox = uri
    osh: openslide.OpenSlide = openslide.OpenSlide(wsi_name)
    location, size = window_convert(bbox)
    size = tuple(x if tile_size is None else tile_size for x in size)
    return osh.read_region(location, 0, size).convert("RGB")


def _sanitize_files(wsi_files, bbox_files, bbox_suffix):
    bbox_set = set([os.path.basename(x) for x in bbox_files])
    for wsi in wsi_files:
        basename = os.path.basename(wsi)
        corresponding_bbox_name = f"{basename}{bbox_suffix}"
        if corresponding_bbox_name not in bbox_set:
            logger.warning(f"{corresponding_bbox_name} not found in bbox lists")
            wsi_files.remove(wsi)
    return wsi_files


def _dict_by_bname(file_list) -> Dict[str, str]:
    return {os.path.basename(file): file for file in file_list}


def wsi_tile_coords(wsi_files,
                    bbox_files,
                    bbox_suffix,
                    nest_level) -> Tuple[List[Tuple[str, Tuple[int, int, int, int]]], List[str], List[str]]:
    """
    Note: grouping based on wsi patterns, not bbox's.
    Args:
        wsi_files:
        bbox_files:
        bbox_suffix:
        nest_level: nest level of the list of bbox in json
    Returns:

    """
    # remove wsi  from the list that does not have a corresponding bbox
    wsi_files = _sanitize_files(wsi_files, bbox_files, bbox_suffix)
    logger.info(f"{len(wsi_files)} images matched.")
    assert len(wsi_files) > 0, f"{len(wsi_files)} images remained after matching to the bbox list"

    # basename --> fullpath
    wsi_dict = _dict_by_bname(wsi_files)
    bbox_dict = _dict_by_bname(bbox_files)

    # bbox aligned to wsi - for each wsi remained in the list, map it to the
    wsi_to_bbox_loc: Dict[str, str] = {wsi_full: bbox_dict[f"{wsi_base}{bbox_suffix}"]
                                       for wsi_base, wsi_full in wsi_dict.items()}
    # 1 wsi --> multiple bbox (list of bbox)
    wsi_to_bbox_list: Dict[str, List[Tuple[int, int, int, int]]] = {wsi_full: bbox_from_json(bbox_full, nest_level)
                                                                    for wsi_full, bbox_full in wsi_to_bbox_loc.items()}

    # pair: wsi_fullname, bbox_tuple for function bbox_img_loader
    uri_collection: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for wsi_full, bbox_list in wsi_to_bbox_list.items():
        uri_list: List[Tuple[str, Tuple[int, int, int, int]]] = list(itertools.product([wsi_full], bbox_list))
        uri_collection += uri_list

    # get filepart_bbox_tuple as output name prefix
    path_prefix_collection: List[str] = []
    name_prefix_collection: List[str] = []
    for wsi_full, bbox_coord in uri_collection:
        filepart, _ = os.path.splitext(os.path.basename(wsi_full))
        # based on wsi folders not
        wsi_dir = os.path.dirname(wsi_full)
        path_prefix = os.path.join(wsi_dir, filepart)
        path_prefix_collection.append(path_prefix)
        name_prefix_collection.append(f"{filepart}_{bbox_coord}")
    return uri_collection, path_prefix_collection, name_prefix_collection


def wsi_bbox_dataset(opt_in, logger_in) -> SimpleSeqDataset:
    logger_in.info("Generate WSI dataset with BBoxes")
    wsi_files = glob.glob(opt_in.data_pattern)
    preproc = pre_processor(opt_in.pad_size, opt.resize_in)
    # note that one WSI corresponds to one json but one json contains multiple bboxes (tiles)
    bbox_files = glob.glob(opt_in.bbox_pattern)
    uri_collection, path_collection, name_prefix_collection = wsi_tile_coords(wsi_files, bbox_files,
                                                                              opt_in.bbox_suffix,
                                                                              opt_in.nest_level)

    # uri_collection = [(wsi_loc, (left, top, right, bottom)), ...]
    # name_prefix_collection = [wsi_folders/wsi_filepart_(left, top, right, bottom), ...]
    prefix_list = SimpleSeqDataset.generate_path_prefix(path_collection,
                                                        name_prefix_collection,
                                                        opt_in.data_pattern, opt_in.group_out,
                                                        opt_in.group_by_file)
    loader = partial(bbox_img_loader, tile_size=opt.tile_size)
    return SimpleSeqDataset(uri_collection, loader=loader, transforms=preproc, prefix_list=prefix_list)


if __name__ == '__main__':
    argv = sys.argv[1:]
    tile_args = BBoxArgs(argv)
    opt = tile_args.get_opts()
    logger_name = __name__
    logger = get_logger(logger_name)
    dataset = wsi_bbox_dataset(opt, logger)
    main_process(opt, logger, dataset)

