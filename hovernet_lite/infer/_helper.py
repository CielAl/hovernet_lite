import logging
import os
from typing import List, Dict, TypedDict
import geojson
import traceback
from argparse import Namespace
import torch
from hovernet_lite.util.misc import save_json, get_timestamp
from hovernet_lite.model.builder import load_model
from hovernet_lite.infer_manager.dataset_proto import SimpleSeqDataset
from hovernet_lite.infer_manager.implementation import Inference
from hovernet_lite.data_type import NucGeoData
from hovernet_lite.util.postprocessing import get_img_from_json_coords, save_json_on_flag, \
    save_image_on_flag, save_overlaid_tile_on_flag
from hovernet_lite.infer_manager.dataset_proto import post_processor
import numpy as np
import sys


class ErrorInfo(TypedDict):
    error_msg: str
    stack_trace: str
    name: str
    batch_name_list: List[str]


def num_nucleus_type(type_info: Dict, default: int):
    assert default is not None or type_info is not None, f"Either of num_nuc and tpye_info must be specified: " \
                                                         f"num_nuc: {default} - type_info {type_info}"
    if type_info is not None:
        return len(type_info) - 1
    return int(default)


def batch_process(geo_data_batch: List[List[NucGeoData]],
                  batch_img: torch.Tensor,
                  prefix_batch: List[str], logger: logging.Logger, opt: Namespace):
    batch_img = batch_img.detach().cpu()
    for geo_data_list, tile_single, prefix_single in zip(geo_data_batch, batch_img, prefix_batch):
        geo_data_list: List[NucGeoData]
        logger.info(f"Working on {prefix_single}")
        if len(geo_data_list) <= 0:
            continue
        tile_size = geo_data_list[0]['tile_size']
        size_incr = 2 * opt.pad_size

        im = get_img_from_json_coords(tile_size + size_incr, geo_data_list, opt.mask_type)
        save_json_on_flag(geo_data_list, save_flag=bool(opt.save_json),
                          export_folder=opt.export_folder, prefix=prefix_single)

        # nuc_mask = im[opt.pad_size:tile_size - opt.pad_size,
        #               opt.pad_size:tile_size - opt.pad_size]
        # nuc_pil = Image.fromarray(nuc_mask)
        nuc_mask = im
        crop_size = tile_size - size_incr
        logger.debug(f"center crop: {crop_size}")
        post_proc = post_processor(crop_size, opt.resize_out)
        nuc_pil = post_proc(nuc_mask)
        nuc_mask_out = np.array(nuc_pil, copy=False)
        save_image_on_flag(nuc_mask_out, save_flag=bool(opt.save_mask),
                           export_folder=opt.export_folder, prefix=prefix_single)
        post_proc(tile_single)
        save_overlaid_tile_on_flag(tile_single, mask=nuc_mask_out, flag=opt.extract_tile, transforms=post_proc,
                                   export_folder=opt.export_folder, prefix=prefix_single)


def main_process(opt: Namespace, logger: logging.Logger, dataset: SimpleSeqDataset):
    assert opt.save_json or opt.save_mask, f"Nothing to Export - Either save_mask or save_json must be set to True"

    # start
    os.makedirs(opt.export_folder, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.visible_device

    # net inputs and data

    with open(opt.type_info_path, 'r') as f:
        type_info = geojson.load(f)
    # read num
    num_of_nuc_types = num_nucleus_type(type_info, default=opt.default_num_type)  # int()
    logger.info('Type info has been loaded.')

    model = load_model(opt.weight_path, opt.net_mode, num_of_nuc_types=num_of_nuc_types)
    logger.info('Model has been loaded.')
    infer = Inference.build(model, type_info=type_info)
    # result: Tuple[List[List[NucGeoData]], List[str]] = infer.infer_dataset(dataset,
    #                                                                        num_of_nuc_types=num_of_nuc_types,
    #                                                                        batch_size=opt.batch_size,
    #                                                                        num_workers=opt.num_workers
    #                                                                        )
    # geo_collection, prefix_list = result
    timestamp_str = get_timestamp()
    opt_save_name = os.path.join(opt.export_folder, f"opt_{timestamp_str}.json")
    if os.path.exists(opt_save_name):
        logger.warning(f"Warning: OPT already exists: {opt_save_name}")
    save_json(opt_save_name, vars(opt), indent=4)
    error_data_list = []
    for geo_data_batch, batch_img, prefix_batch in infer.infer_dataset(dataset,
                                                                       num_of_nuc_types=num_of_nuc_types,
                                                                       batch_size=opt.batch_size,
                                                                       num_workers=opt.num_workers
                                                                       ):
        # noinspection PyBroadException
        try:
            batch_process(geo_data_batch, batch_img, prefix_batch, logger, opt)
        except KeyboardInterrupt:
            sys.exit(130)
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            info_dict = ErrorInfo(error_msg=error_msg, stack_trace=stack_trace,
                                  name=prefix_batch[0], batch_name_list=prefix_batch)
            logger.critical(f"{error_msg}")
            logger.critical(f"{stack_trace}")
            logger.critical(f"First of batch: {prefix_batch[0]}")
            logger.critical(f"All Batch: {prefix_batch}")

            error_data_list.append(info_dict)
    error_save_name = os.path.join(opt.export_folder, f"error_{timestamp_str}.json")
    save_json(error_save_name, error_data_list, indent=4)
    logger.info("Nucleus segmentation done!")
