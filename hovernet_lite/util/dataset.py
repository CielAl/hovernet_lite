from typing import Callable, Any, List

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Pad, PILToTensor, CenterCrop, ToPILImage


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
    KEY_IMG: str = 'img'
    KEY_NAME_ID: str = 'name_id'

    name_id_list: List[str]
    uri_list: List[Any]
    loader: Callable[[Any], Image.Image]
    transforms: Callable

    def __init__(self, uri_list: List[Any],
                 name_id_list: List[str],
                 loader: Callable[[Any], Image.Image] = pil_loader,
                 transforms: Callable = None):
        """
        Args:
            uri_list: a list of URIs that lead to image resources
            name_id_list: name identifiers (e.g., suffix) for each individual tile output for export purpose
            loader: loading func to fetch image from uri_list
            transforms: preprocessing process
        """
        self.uri_list = uri_list
        self.name_id_list = name_id_list
        assert len(self.uri_list) == len(self.name_id_list)
        self.transforms = transforms
        self.loader = loader

    def __len__(self):
        return len(self.uri_list)

    def __getitem__(self, index):
        out = dict()
        data = self.loader(self.uri_list[index])
        if self.transforms is not None:
            data = self.transforms(data)
        out[SimpleSeqDataset.KEY_IMG] = data
        out[SimpleSeqDataset.KEY_NAME_ID] = self.name_id_list[index]
        return out
