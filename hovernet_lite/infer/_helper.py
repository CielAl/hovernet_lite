import glob
import os
from typing import Tuple, List
import geojson

from hovernet_lite.model.builder import load_model
from hovernet_lite.util.dataset import batch_pre_processor, SimpleSeqDataset, pil_loader
from hovernet_lite.util.infer_helper import Inference, NucGeoData
from hovernet_lite.util.postprocessing import get_img_from_json_coords, IMG_TYPE_PROB, save_json_on_flag, \
    save_image_on_flag


def tile_datasset(opt, logger) -> SimpleSeqDataset:
    files = glob.glob(opt.data_pattern)
    logger.info(f'{len(files)} files have been read.')
    preproc = batch_pre_processor(opt.pad_size)
    file_parts = [os.path.splitext(os.path.basename(x))[0] for x in files]
    prefix_list = [os.path.basename(os.path.dirname(x)) for x in files]
    name_ids = [os.path.join(x, y) for x, y in zip(prefix_list, file_parts)]
    logger.debug(f"sample name_ids: {name_ids[0]}")
    dataset = SimpleSeqDataset(files, loader=pil_loader, transforms=preproc, name_id_list=name_ids)
    return dataset


def main_process(opt, logger, dataset: SimpleSeqDataset):
    assert opt.save_json or opt.save_mask, f"Nothing to Export - Either save_mask or save_json must be set to True"

    # start
    num_of_nuc_types = int(opt.num_types)
    os.makedirs(opt.export_folder, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.visible_device

    # net inputs and data
    size_incr = 2 * opt.pad_size

    with open(opt.type_info_path, 'r') as f:
        type_info = geojson.load(f)
    logger.info('Type info has been loaded.')

    model = load_model(opt.weight_path, opt.net_mode, num_of_nuc_types=num_of_nuc_types)
    logger.info('Model has been loaded.')

    infer = Inference.build(model, type_info=type_info)
    result: Tuple[List[List[NucGeoData]], List[str]] = infer.infer_dataset(dataset,
                                                                           num_of_nuc_types=num_of_nuc_types,
                                                                           batch_size=opt.batch_size,
                                                                           num_workers=opt.num_workers
                                                                           )
    geo_collection, name_ids = result

    for geo_data_list, name_id_single in zip(geo_collection, name_ids):
        if len(geo_data_list) <= 0:
            continue
        tile_size = geo_data_list[0]['tile_size']
        im = get_img_from_json_coords(tile_size + size_incr, geo_data_list, IMG_TYPE_PROB)
        save_json_on_flag(geo_data_list, save_flag=opt.save_json,
                          export_folder=opt.export_folder, name_id=name_id_single)

        nuc_mask = im[opt.pad_size:tile_size - opt.pad_size,
                   opt.pad_size:tile_size - opt.pad_size]
        save_image_on_flag(nuc_mask, save_flag=opt.save_mask,
                           export_folder=opt.export_folder, name_id=name_id_single)

    logger.info("Nucleus segmentation done!")
