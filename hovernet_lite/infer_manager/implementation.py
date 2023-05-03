from collections import OrderedDict
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Dict, Union, Tuple, Generator
from torch.utils.data import DataLoader

from hovernet_lite.constants import DEFAULT_NUC_OFFSET
from hovernet_lite.data_type import InstInfo, InstGeo, InstClass, InstProperties, NucGeoData
from hovernet_lite.util.postprocessing import processed_nuclei_pred, get_bounding_box
from hovernet_lite.infer_manager.dataset_proto import SimpleSeqDataset
from hovernet_lite.logger import GlobalLoggers
# from hovernet_lite.error_handler import remediate_call
logger = GlobalLoggers.instance().get_logger(__name__)


class Inference:
    MAX_DATA_COUNT = np.inf
    model: nn.Module
    type_info: Dict[str, List[Union[str, int]]]
    max_count: Union[float, int]

    def __init__(self, model, type_info, max_count: Union[int, float] = MAX_DATA_COUNT):
        self.model = model
        self.type_info = type_info
        self.max_count = max_count
        for params in self.model.parameters():
            params.requires_grad_(False)

    @classmethod
    def build(cls, model, type_info, max_count: Union[int, float] = MAX_DATA_COUNT):
        return cls(model=model, type_info=type_info, max_count=max_count)

    @staticmethod
    def class_info_to_inst_dict(inst_info_dict: Dict[int, InstInfo],
                                pred_inst: np.ndarray,
                                pred_type: np.ndarray,
                                num_of_nuc_types: Union[int, None]):
        """
        Write the label and output softmax score of each instance (nuclei) into the inst_info_dict
        Args:
            inst_info_dict: Dictionary that maps instance id to InstInfo
            pred_inst: The concatenated prediction outputs from HoverNet: [Tp, Np, Hv]. tp is the map of
                instances (class label on each pixel) with size HxWx1. Np is the softmax response of whether a pixel
                is nuclei or background, with size HxWx1. Hv is the two-channel map of horizontal/vertical
                distance to center, with size HxWx2. A complete pred_inst is with shape of H x W x 4
            pred_type:
            num_of_nuc_types:

        Returns:

        """

        # only if number of nuc_types is defined
        if num_of_nuc_types is None:
            return

        assert pred_inst is not None
        assert pred_type is not None
        assert inst_info_dict is not None and isinstance(inst_info_dict, Dict)

        # Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            rmin, cmin, rmax, cmax = (inst_info_dict[inst_id]["bbox"]).flatten()
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]
            inst_map_crop = (
                    inst_map_crop == inst_id
            )  # TODO: duplicated operation, may be expensive
            inst_type = inst_type_crop[inst_map_crop]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)
            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["type_prob"] = float(type_prob)

    @staticmethod
    def spatial_info_to_inst_dict(inst_info_dict: Dict[int, InstInfo],
                                  pred_inst: np.ndarray,
                                  num_of_nuc_types: int,
                                  return_centroids: bool):
        """
        Write the spatial information of nuclei to the instance info dict of
            class/instance id -->  dict of (bbox/centroid/contour)
        Args:
            pred_inst: The concatenated prediction outputs from HoverNet: [Np, Hv]. Np is the softmax response of
                whether a pixel is nuclei or background, with size HxWx1.
                Hv is the two-channel map of horizontal/vertical distance to center, with size HxWx2.
                A complete pred_inst is with shape of H x W x 3
            inst_info_dict: The dict to output the results
            num_of_nuc_types: Number of nuclei type. Not effective in this procedure but serve as a flag.
                Either of num_of_nuc_types or return_centroids must be set to not None or True respectively.
            return_centroids: Whether to return the centroid

        Returns:

        """
        # do nothing if number of nuc instance is not specified and not in return_centroids mode.
        if not return_centroids and num_of_nuc_types is None:
            return

        assert inst_info_dict is not None and isinstance(inst_info_dict, Dict)
        inst_id_list = np.unique(pred_inst)[1:]  # exlcude background

        # for each nuclei class
        for inst_id in inst_id_list:
            inst_map = pred_inst == inst_id
            # TODO: change format of bbox output
            # find the bbox (row column)
            rmin, rmax, cmin, cmax = get_bounding_box(inst_map)
            inst_bbox = np.array([[rmin, cmin], [rmax, cmax]])
            # map in the bbox
            inst_map = inst_map[inst_bbox[0][0]: inst_bbox[1][0],
                                inst_bbox[0][1]: inst_bbox[1][1]]
            inst_map = inst_map.astype(np.uint8)
            inst_moment = cv2.moments(inst_map)
            # previously inst_contour = cv2.findContours, but that also involves the hierarchy which is unused
            # now explicitly save the contours only
            inst_contour, _ = cv2.findContours(
                inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            # * opencv protocol format may break
            # previously inst_contour[0][0] for the contour line of the first object
            # since now inst_contour does not have hierarchy --> change to inst_contour[0]
            # we assume only one nuclei object is in the bbox.
            inst_contour = np.squeeze(inst_contour[0].astype("int32"))
            # < 3 points dont make a contour, so skip, likely artifact too
            # as the contours obtained via approximation => too small or sth
            if inst_contour.shape[0] < 3:
                continue
            if len(inst_contour.shape) != 2:
                # todo issue warning
                continue  # ! check for trickery shape
            inst_centroid = [
                (inst_moment["m10"] / inst_moment["m00"]),
                (inst_moment["m01"] / inst_moment["m00"]),
            ]
            inst_centroid = np.array(inst_centroid)

            inst_contour[:, 0] += inst_bbox[0][1]  # X
            inst_contour[:, 1] += inst_bbox[0][0]  # Y
            inst_centroid[0] += inst_bbox[0][1]  # X
            inst_centroid[1] += inst_bbox[0][0]  # Y

            info_dict_single = InstInfo(bbox=inst_bbox, centroid=inst_centroid, contour=inst_contour,
                                        type_prob=None, type=None)
            inst_info_dict[inst_id] = info_dict_single

    @staticmethod
    def process_info_dict(pred_map: np.ndarray,
                          num_of_nuc_types: int = None,
                          return_centroids: bool = True) -> Dict[int, InstInfo]:
        """Postprocessing script for image tiles.
        Args:
            pred_map: The concatenated prediction outputs from HoverNet: [Tp, Np, Hv]. tp is the map of
                instances (class label on each pixel) with size HxWx1. Np is the softmax response of whether a pixel
                is nuclei or background, with size HxWx1. Hv is the two-channel map of horizontal/vertical
                distance to center, with size HxWx2. A complete pred_inst is with shape of H x W x 4

            num_of_nuc_types: number of types considered at output of nc branch
            return_centroids: whether to return centroids of nuclei

        Returns:
            pred_inst:     pixel-wise nuclear instance segmentation prediction
            pred_type_out: pixel-wise nuclear type prediction

        """
        if num_of_nuc_types is not None:
            pred_type = pred_map[..., :1]
            pred_inst = pred_map[..., 1:]
            pred_type = pred_type.astype(np.int32)
        else:
            pred_inst = pred_map
            pred_type = None

        pred_inst = np.squeeze(pred_inst)
        pred_inst = processed_nuclei_pred(pred_inst)

        inst_info_dict = dict()
        Inference.spatial_info_to_inst_dict(inst_info_dict, pred_inst, num_of_nuc_types, return_centroids)
        Inference.class_info_to_inst_dict(inst_info_dict, pred_inst, pred_type, num_of_nuc_types)

        # ! WARNING: ID MAY NOT BE CONTIGUOUS
        # inst_id in the dict maps to the same value in the `pred_inst`
        return inst_info_dict

    @staticmethod
    def batch_inst_info_dict(model: nn.Module,
                             batch_imgs: torch.Tensor,
                             num_of_nuc_types) -> List[Dict[int, InstInfo]]:
        # patch_imgs_gpu = torch.from_numpy(patch_imgs).type(torch.FloatTensor).to('cuda')
        # patch_imgs_gpu = patch_imgs_gpu.permute(0, 3, 1, 2).contiguous() # to NCHW

        model.eval()  # infer mode

        # --------------------------------------------------------------
        with torch.no_grad():  # dont compute gradient
            logger.debug(f"Batch Size: {batch_imgs.shape[0]}")
            batch_imgs = batch_imgs.type(torch.FloatTensor).to('cuda')
            pred_dict = model(batch_imgs)

            pred_dict = OrderedDict(
                [(k, v.permute(0, 2, 3, 1).contiguous().detach().cpu()) for k, v in pred_dict.items()]
            )
            pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)[..., 1:].detach().cpu()
            if "tp" in pred_dict:
                type_map = F.softmax(pred_dict["tp"], dim=-1)
                type_map = torch.argmax(type_map, dim=-1, keepdim=True)
                type_map = type_map.type(torch.FloatTensor).detach().cpu()
                pred_dict["tp"] = type_map
            pred_output = torch.cat(list(pred_dict.values()), -1)

        # * Its up to user to define the protocol to process the raw output per step!
        pred_output = pred_output.detach().cpu().numpy()
        batch_inst_info = []
        for i in range(pred_output.shape[0]):
            inst_info = Inference.process_info_dict(pred_output[i, ...],
                                                    num_of_nuc_types=num_of_nuc_types,
                                                    return_centroids=True)
            batch_inst_info.append(inst_info)
        return batch_inst_info

    @staticmethod
    def process_batch_geo_data(model: nn.Module,
                               batch_img: torch.Tensor,
                               num_of_nuc_types: int,
                               batch_base_coords: Union[np.ndarray, List[Union[Tuple[int, int], np.ndarray]]],
                               type_info: Dict[str, List[Union[str, int]]],
                               max_count: Union[int, float] = MAX_DATA_COUNT,
                               offset: int = DEFAULT_NUC_OFFSET) -> List[List[NucGeoData]]:
        """
        Process the geo data of a batch and write it to geo_data_list
        Args:
            model: The hovernet model
            batch_img: batch of images NxCxHxW
            num_of_nuc_types: number of nuclei types (from type_info)
            batch_base_coords: the base coord of each element in the batch to reconstruct WSI masks
            type_info: type info dict read from json files
            max_count: max count of instances. Early stop if the counter exceeds the max_count. default is inf
            offset: offset of contour coords. Default 46

        Returns:
            counter --> how many values were processed
        """

        # offset: int = 92 // 2
        # batch[0, :, :, :] = np.array(tile)
        # batchInfo[0] = (0*downsampleRatio+coord[0],0*downsampleRatio+coord[1])
        # batch_base_coords[0] = (0, 0)

        list_of_info_dicts: List[Dict[int, InstInfo]] = Inference.batch_inst_info_dict(model,
                                                                                       batch_img,
                                                                                       num_of_nuc_types,)
        counter = 0

        tile_size = batch_img.shape[2]

        geo_data_batch: List[List[NucGeoData]] = []
        for base_coord_single, pred_single in zip(batch_base_coords, list_of_info_dicts):
            geo_data_list: List[NucGeoData] = []
            # pred_single: Dict[int, InstInfo]
            for inst_key, inst_info in pred_single.items():
                if counter >= max_count:
                    # quit if exceed predefined limit
                    break
                if len(inst_info['contour']) < 10:
                    continue
                # n*2 coords for contour lines returned by cv2.findContours
                contour_coords: np.ndarray = np.array(inst_info['contour'])
                # add the offset and the base coord --> so this can be reused for multiple tiles if stitch is
                # necessary in future steps (e.g., WSI masks)
                contour_coords += [base_coord_single[0] + offset, base_coord_single[1] + offset]

                # noinspection PyTypeChecker
                contour_list: List[List[int]] = contour_coords.tolist()
                contour_list.append(contour_list[0])

                pred_geo = InstGeo(type="Polygon", coordinates=[contour_list])
                nuc_class_name: str = type_info[str(inst_info['type'])][0]
                pred_class = InstClass(name=nuc_class_name, probability=inst_info['type_prob'])
                pred_prop = InstProperties(isLocked="false", measurements=[], classification=pred_class)
                dict_data = NucGeoData(type="Feature", id="PathCellObject", geometry=pred_geo,
                                       properties=pred_prop, tile_size=tile_size)
                geo_data_list.append(dict_data)
                counter += 1
            geo_data_batch.append(geo_data_list)
        return geo_data_batch

    @staticmethod
    def _default_batch_coords(num_batch) -> List[Tuple[int, int]]:
        return [(0, 0)] * num_batch

    def infer_img(self,
                  batch_img: torch.Tensor,
                  num_of_nuc_types: int,
                  batch_base_coords: Union[np.ndarray, List[Union[Tuple[int, int], np.ndarray]], None] = None,
                  offset: int = DEFAULT_NUC_OFFSET) -> List[List[NucGeoData]]:
        """
        Args:
            batch_img: batch to process
            num_of_nuc_types:  Number of nuclei types.
            batch_base_coords: Base coordinates of each element in the batch for re-stitch in future steps
            offset: Offset to correct the contour lines.

        Returns:
             List[List[NucGeoData]]: List[NucGeoData] --> all nuclei of one input --> batchify
        """
        if batch_base_coords is None:
            batch_base_coords = Inference._default_batch_coords(batch_img.shape[0])
        geo_data_batch = Inference.process_batch_geo_data(self.model,
                                                          batch_img,
                                                          num_of_nuc_types,
                                                          batch_base_coords,
                                                          self.type_info,
                                                          self.max_count,
                                                          offset)
        return geo_data_batch

    def infer_batch(self, batch: Dict[str, Union[torch.Tensor, List[str]]],
                    num_of_nuc_types: int,
                    batch_base_coords: Union[np.ndarray, List[Union[Tuple[int, int], np.ndarray]], None] = None,
                    offset: int = DEFAULT_NUC_OFFSET):
        batch_img: torch.Tensor = batch[SimpleSeqDataset.KEY_IMG]
        prefix_list: List[str] = batch[SimpleSeqDataset.KEY_NAME_PREFIX]
        logger.debug(f"Process Batch: {batch_img.shape}")
        geo_data_batch = self.infer_img(batch_img, num_of_nuc_types, batch_base_coords, offset)
        # geo_collection.append(geo_data_batch)
        # suffix_collection.append(prefix_list)
        # still batchiftied (List[List[GeoDat]] --> flatten to list[Geodata] i.e.
        return geo_data_batch, batch_img.detach().cpu(), prefix_list

    def infer_dataset(self,
                      dataset: SimpleSeqDataset,
                      num_of_nuc_types: int,
                      batch_size: int = 1,
                      num_workers: int = 0,
                      batch_base_coords: Union[np.ndarray, List[Union[Tuple[int, int], np.ndarray]], None] = None,
                      offset: int = DEFAULT_NUC_OFFSET)\
            -> Generator[Tuple[List[List[NucGeoData]], torch.Tensor, List[str]], None, None]:
        """
        Generator wrapper to process the dataset. Somehow it reduces the readability and make it harder to
            handle errors while running on HPCs.
        Args:
            dataset: SimpleSeqDataset which returns the image and the name identifier (for export purpose)
            batch_size: batch_size to process within a DataLoader
            num_workers: num_workers for the DataLoader
            num_of_nuc_types:  Number of nuclei types.
            batch_base_coords: Base coordinates of each element in the batch for re-stitch in future steps
            offset: Offset to correct the contour lines.
        Returns:

        """
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                 pin_memory=False)
        # geo_collection = []
        # suffix_collection = []
        # for batch_idx in range(len(data_loader)):
        for batch in data_loader:
            # batch_img: torch.Tensor = batch[SimpleSeqDataset.KEY_IMG]
            # prefix_list: List[str] = batch[SimpleSeqDataset.KEY_NAME_PREFIX]
            # logger.debug(f"Process Batch: {batch_img.shape}")
            # geo_data_batch = self.infer_img(batch_img, num_of_nuc_types, batch_base_coords, offset)
            # # geo_collection.append(geo_data_batch)
            # # suffix_collection.append(prefix_list)
            # # still batchiftied (List[List[GeoDat]] --> flatten to list[Geodata] i.e.
            # batch_out = self.infer_batch(batch,
            #                              num_of_nuc_types,
            #                              batch_base_coords,
            #                              offset)
            # batch_out = remediate_call(self.infer_batch, __name__,  batch[SimpleSeqDataset.KEY_NAME_PREFIX],
            #                            batch, num_of_nuc_types, batch_base_coords, offset)
            # batch = next(iter(data_loader))

            batch_out = self.infer_batch(batch,
                                         num_of_nuc_types,
                                         batch_base_coords,
                                         offset)
            geo_data_batch, batch_img_cpu, prefix_list = batch_out
            yield geo_data_batch, batch_img_cpu, prefix_list

        # return sum(geo_collection, []), sum(suffix_collection, [])
