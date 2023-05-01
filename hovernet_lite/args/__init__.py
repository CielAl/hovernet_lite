import argparse
from abc import ABC, abstractmethod


class BaseArgs(ABC):
    PROG = "hovernet_lite"
    parser: argparse.ArgumentParser

    @staticmethod
    def base_arg_parser():
        parser = argparse.ArgumentParser(prog=BaseArgs.PROG, description='Run a lite version of HoverNet for Inference')
        parser.add_argument("--data_pattern", help="input wildcard", type=str, required=True)
        parser.add_argument("--weight_path", help="Path of the pretrained model to load", type=str,
                            default='./pretrained_models/hovernet_fast_monusac_type_tf2pytorch.tar')
        parser.add_argument("--net_mode", help="Infer Mode - chose from original and fast", type=str,
                            default='fast', choices=['original', 'fast'])
        parser.add_argument("--default_num_type", help="Default Number of Nucleus Types if Type Info is Missing."
                                                       "Num of nucleus types is len(type_info_json) - 1",
                            type=int)
        parser.add_argument("--pad_size", help="Padding to input images. The default value of the official branch "
                                               "is 48", type=int,
                            default=48)

        parser.add_argument("--resize_in", help="Resize tile after being read, e.g., if there is GPU VRAM limit",
                            type=int)
        parser.add_argument("--resize_out", help="Resize mask if necessary, e.g., if the resize_in is set due to"
                                                 "VRAM limitation and the mask of original size is desired.",
                            type=int)

        parser.add_argument("--batch_size", help="Batch size for the dataloader during inference procedure", type=int,
                            default=6)

        parser.add_argument("--num_workers", help="Num of workers in data loader", type=int,
                            default=6)

        parser.add_argument("--visible_device", help="visible device id of GPUs. comma separated string", type=str,
                            default="0, 1")

        parser.add_argument("--type_info_path", help="Path of the type_info_json", type=str,
                            default='./type_info/type_info_monusac.json')
        parser.add_argument("--export_folder", help="Export Folder", type=str,
                            default='./samples/output/')

        parser.add_argument("--save_json", help="Flag to Export geojson of nuclei for each mask", type=int,
                            default=0)

        parser.add_argument("--save_mask", help="Flag to Export Mask", type=int,
                            default=1)

        parser.add_argument("--extract_tile", help="Flag to export mask-overlaid tiles (e.g., for inspection)",
                            action='store_true', default=False)

        parser.add_argument("--mask_type", help="Flag to Export Mask - inst: pixel value as inst/class id."
                                                "binary - binary mask of wherever nuclei presents."
                                                "prob - the softmax score map of nuclei vs. background", type=str,
                            default='prob', choices=['prob', 'binary', 'inst'])

        parser.add_argument("--group_out", help="Whether to group the outputs by the asterisk-matched subdirectories"
                                                "For instance, if data_pattern is /A/*/*.png wherein the first asterisk"
                                                "corresponds to any WSI name, with --group_out enabled the final output"
                                                "loc will be [export_folder]/[any_matched_wsi_name]/[tile_name]",
                            action='store_true',
                            default=False)

        parser.add_argument("--group_by_file", help="Additional to --group_out, whether group on individual input file"
                                                    "e.g., for input pattern: *.png and a matach: fileA.png,"
                                                    " a folder fileA will be created if this flag is set."
                                                    "No effect if group_out is False",
                            default=False,
                            action='store_true')
        # opt, _ = parser.parse_known_args(argv)
        return parser

    def __init__(self, argv):
        self.parser = BaseArgs.base_arg_parser()
        self.argv = argv

    @abstractmethod
    def additional_args(self):
        return NotImplemented

    def get_opts(self):
        self.additional_args()
        opt, _ = self.parser.parse_known_args(self.argv)
        return opt
