import glob
import os
from typing import Tuple, List, Dict
import geojson

from hovernet_lite.model.builder import load_model
from hovernet_lite.infer_manager.dataset_proto import batch_pre_processor, SimpleSeqDataset, pil_loader
from hovernet_lite.infer_manager.implementation import Inference
from hovernet_lite.data_type import NucGeoData
from hovernet_lite.util.postprocessing import get_img_from_json_coords, IMG_TYPE_PROB, save_json_on_flag, \
    save_image_on_flag


def num_nucleus_type(type_info: Dict, default: int):
    assert default is not None or type_info is not None, f"Either of num_nuc and tpye_info must be specified: " \
                                                         f"num_nuc: {default} - type_info {type_info}"
    if type_info is not None:
        return len(type_info) - 1
    return int(default)


def main_process(opt, logger, dataset: SimpleSeqDataset):
    assert opt.save_json or opt.save_mask, f"Nothing to Export - Either save_mask or save_json must be set to True"

    # start
    os.makedirs(opt.export_folder, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.visible_device

    # net inputs and data
    size_incr = 2 * opt.pad_size
    with open(opt.type_info_path, 'r') as f:
        type_info = geojson.load(f)
    # read num
    num_of_nuc_types = num_nucleus_type(type_info, default=opt.default_num_type)  # int()
    logger.info('Type info has been loaded.')

    model = load_model(opt.weight_path, opt.net_mode, num_of_nuc_types=num_of_nuc_types)
    logger.info('Model has been loaded.')
    infer = Inference.build(model, type_info=type_info)
    result: Tuple[List[List[NucGeoData]], List[str]] = infer.infer_dataset(dataset,
                                                                           num_of_nuc_types=num_of_nuc_types,
                                                                           batch_size=opt.batch_size,
                                                                           num_workers=opt.num_workers
                                                                           )
    geo_collection, prefix_list = result
    for geo_data_list, prefix_single in zip(geo_collection, prefix_list):
        if len(geo_data_list) <= 0:
            continue
        tile_size = geo_data_list[0]['tile_size']
        im = get_img_from_json_coords(tile_size + size_incr, geo_data_list, IMG_TYPE_PROB)
        save_json_on_flag(geo_data_list, save_flag=opt.save_json,
                          export_folder=opt.export_folder, prefix=prefix_single)

        nuc_mask = im[opt.pad_size:tile_size - opt.pad_size,
                      opt.pad_size:tile_size - opt.pad_size]
        save_image_on_flag(nuc_mask, save_flag=opt.save_mask,
                           export_folder=opt.export_folder, prefix=prefix_single)

    logger.info("Nucleus segmentation done!")
