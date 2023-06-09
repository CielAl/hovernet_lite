import os.path
from typing import Callable, Any, List, Union
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Pad, PILToTensor, CenterCrop, ToPILImage, Resize
from hovernet_lite.data_type import DatasetOut
from hovernet_lite.util.misc import find_wildcards, path_components
from hovernet_lite.error_handler import remediate_call, ExceptionSignal
from torch.utils.data.dataloader import default_collate, DataLoader


def identity(value):
    return value


def pre_processor(pad_size, resize_in: int = None):
    # no scaling for now with PILToTensor
    resize_func = identity if resize_in is None else Resize(size=resize_in)
    return Compose([
        resize_func,
        Pad(pad_size),
        PILToTensor(),
    ])


def post_processor(crop_size, resize_out):
    resize_func = identity if resize_out is None else Resize(size=resize_out)
    return Compose([
        ToPILImage(),
        CenterCrop(crop_size),
        resize_func,
    ])


def pil_loader(fname: str) -> Image.Image:
    return Image.open(fname)


class SimpleSeqDataset(Dataset):
    """
    The Dataset Interface to be use for all potential inference procedures. Returns the image representation of the
        data point and an output_name_suffix, where in export_folder (in opts) + output_name_suffix = full output path.
        The output_name_suffix may contain any new subdirectories under the export_folder.
        Return None if the current data is unreadable, wherein the none value can be handled afterward.
        so the pipeline won't be broken.
    """
    KEY_IMG: str = 'img'
    KEY_NAME_PREFIX: str = 'prefix'

    prefix_list: List[str]
    uri_list: List[Any]
    loader: Callable[[Any], Image.Image]
    __transforms: Callable

    @property
    def transforms(self):
        return self.__transforms

    @transforms.setter
    def transforms(self, x: Callable):
        assert isinstance(x, Callable)
        self.__transforms = x

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
        self.__transforms = transforms
        self.loader = loader

    def __len__(self):
        return len(self.uri_list)

    def __getitem__(self, index):
        # data = self.loader(self.uri_list[index])
        input_arg = self.uri_list[index]
        name = self.prefix_list[index]
        # data = self.loader(input_arg)
        data: Union[Image.Image, ExceptionSignal] = remediate_call(self.loader, __name__, name, False, input_arg)
        if isinstance(data, ExceptionSignal):
            return None
        if self.transforms is not None:
            data = self.transforms(data)
        out = DatasetOut(img=data, prefix=name)
        return out

    @staticmethod
    def _not_none(batch):
        return batch is not None

    @staticmethod
    def collate_drop_none(batch):
        batch = list(filter(SimpleSeqDataset._not_none, batch))
        if len(batch) > 0:
            return default_collate(batch)
        return None

    def get_data_loader(self, batch_size, num_workers, pin_memory=True):
        return DataLoader(self, batch_size=batch_size, num_workers=num_workers,
                          collate_fn=SimpleSeqDataset.collate_drop_none,
                          shuffle=False, pin_memory=pin_memory)

    @staticmethod
    def generate_path_prefix(input_path_prefix_list: List[str],
                             input_basename_list: List[str],
                             input_pattern: str,
                             group_flag: bool,
                             group_by_file: bool) -> List[str]:
        """
        Generate the output prefix. export_folder (in opts) + prefix + output_name_suffix = full output path.
            Can optionally group the output files based on wildcards in input_pattern.
            If the input pattern is /A/*/*.png, and the input_list is [/A/1/a.png, /A/2/b.png], the resulting
            prefix would be [1, 2]. Note that the function will not remove any file type extension. Preprocess the list
            if necessary.
        Args:
            input_path_prefix_list: list of paths corresponding to inputs that are used to derive the out filenames.
                Doesn't need to be actual input names. Necessary suffix (e.g., output id/coords) can be concatenated
                into the input_path correspondingly.
            input_basename_list: list of prefix of the filename itself. Does not contain any paths. Will always be
                base names.
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
        # basename of filenames to output
        filepart_list = [os.path.basename(x) for x in input_basename_list]
        # grouping disabled or nothing to group
        if not group_flag or len(matched_idx) == 0:
            # return the basenames directly if no grouping
            return filepart_list
        # all folders along the path
        component_list = [path_components(x) for x in input_path_prefix_list]
        # find all subdirectories corresponding to asterisks
        components_to_group: List[List[str]] = [[component[idx] for idx in matched_idx] for component in component_list]
        # for each input: concatenate the list of asterisks-matched subdirectory with basename
        grouped_prefix_list = [comp_list + [fpart] for comp_list, fpart in zip(components_to_group, filepart_list)]
        # for each input: use os.path.join to concatenate the above list into a path
        output = [os.path.join(*x) for x in grouped_prefix_list]
        return output
