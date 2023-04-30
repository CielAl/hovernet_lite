import os.path
from typing import Callable, Any, List
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Pad, PILToTensor, CenterCrop, ToPILImage
from hovernet_lite.data_type import DatasetOut
from hovernet_lite.util.misc import find_wildcards, path_components


def batch_pre_processor(pad_size):
    # no scaling for now with PILToTensor
    return Compose([
        Pad(pad_size),
        PILToTensor(),
    ])


def batch_post_processor(img_size):
    return Compose([
        CenterCrop(img_size),
        ToPILImage()
    ])


def pil_loader(fname: str) -> Image.Image:
    return Image.open(fname)


class SimpleSeqDataset(Dataset):
    """
    The Dataset Interface to be use for all potential inference procedures. Returns the image representation of the
        data point and an output_name_suffix, where in export_folder (in opts) + output_name_suffix = full output path.
        The output_name_suffix may contain any new subdirectories under the export_folder.
    """
    KEY_IMG: str = 'img'
    KEY_NAME_PREFIX: str = 'prefix'

    prefix_list: List[str]
    uri_list: List[Any]
    loader: Callable[[Any], Image.Image]
    transforms: Callable

    def __init__(self, uri_list: List[Any],
                 prefix_list: List[str],
                 loader: Callable[[Any], Image.Image] = pil_loader,
                 transforms: Callable = None):
        """
        Args:
            uri_list: a list of URIs that lead to image resources. It might be a list of filenames, or a list of
                partials to load resources, or anything else.
            prefix_list: name identifiers (e.g., suffix) for each individual tile output for export purpose
            loader: loading func to fetch image from uri_list
            transforms: preprocessing process
        """
        self.uri_list = uri_list
        self.prefix_list = prefix_list
        assert len(self.uri_list) == len(self.prefix_list)
        self.transforms = transforms
        self.loader = loader

    def __len__(self):
        return len(self.uri_list)

    def __getitem__(self, index):
        data = self.loader(self.uri_list[index])
        if self.transforms is not None:
            data = self.transforms(data)
        out = DatasetOut(img=data, prefix=self.prefix_list[index])
        return out

    @staticmethod
    def generate_path_prefix(input_path_prefix_list: List[str], input_pattern: str,
                             group_flag: bool,
                             group_by_file: bool) -> List[str]:
        """
        Generate the output prefix. export_folder (in opts) + prefix + output_name_suffix = full output path.
            Can optionally group the output files based on wildcards in input_pattern.
            If the input pattern is /A/*/*.png, and the input_list is [/A/1/a.png, /A/2/b.png], the resulting
            prefix would be [1, 2]
        Args:
            input_path_prefix_list: list of paths corresponding to inputs that are used to derive the out filenames.
                Doesn't need to be actual input names. Necessary suffix (e.g., output id/coords) can be concatenated
                into the input_path correspondingly.
            input_pattern: the input pattern string (e.g., path with wildcard). The wildcard location might represent
                name of higher hierarchy (e.g., the WSI name) that can be used to group the masks.
            group_flag: whether to group the output file using input_pattern.
            group_by_file: Additional to --group_out, whether group on individual input file"
                "e.g., for input fileA.png, a folder fileA will be created as well
        Returns:

        """

        raw_pattern_component, matched_idx = find_wildcards(input_pattern)
        # get rid of the right most level (e.g., if the asterisk is to match the filename)
        if not group_by_file:
            right_most_idx = len(raw_pattern_component) - 1
            matched_idx = [x for x in matched_idx if x < right_most_idx]
        # grouping disabled or nothing to group
        filepart_list = [os.path.splitext(os.path.basename(x))[0] for x in input_path_prefix_list]
        if not group_flag or len(matched_idx) == 0:
            # copy
            return filepart_list
        component_list = [path_components(x) for x in input_path_prefix_list]
        # find all subdirectories corresponding to asterisks
        components_to_group: List[List[str]] = [[component[idx] for idx in matched_idx] for component in component_list]
        # for each input: concatenate the list of asterisks-matched subdirectory with basename
        grouped_prefix_list = [comp_list + [fpart] for comp_list, fpart in zip(components_to_group, filepart_list)]
        # for each input: use os.path.join to concatenate the above list into a path
        output = [os.path.join(*x) for x in grouped_prefix_list]
        return output
