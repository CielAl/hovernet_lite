# hovernet_lite
A compact and lightweighted version of HoverNet, derived and adapted from the [official PyTorch implementation](https://github.com/vqdang/hover_net) 
The sample pretrained monusac and pannuke models from the official repository are stored in ./pretrained_models/ along with the corresponding type info JSON files.

The libary unwrap the inference procedure and expose the convention of dataset in `hovernet_lite.infer_manager.data_proto.SimpleSeqDataset` which allows the customization of most of input formats other than tiles, (e.g., WSI + cooresponding bounding box of tiles to segment or HDF5 dataset so long as the corresponding reading protocal is defined as the loading function `SimpleSeqDataset.loader` of the dataset). Details are listed in dosctring inside `hovernet_lite.infer_manager.data_proto` and  `hovernet_lite.infer`.

### TODO: WSI-level mask generation

# Usage

## Tile Mode
Segment the nuclei of individual tiles.
```
$ python -m hovernet_lite.infer.tile -h
usage: hovernet_lite [-h] --data_pattern DATA_PATTERN [--weight_path WEIGHT_PATH] [--net_mode {original,fast}] [--default_num_type DEFAULT_NUM_TYPE] [--pad_size PAD_SIZE] [--resize_in RESIZE_IN] [--resize_out RESIZE_OUT] [--batch_size BATCH_SIZE]
                     [--num_workers NUM_WORKERS] [--visible_device VISIBLE_DEVICE] [--type_info_path TYPE_INFO_PATH] [--export_folder EXPORT_FOLDER] [--save_json SAVE_JSON] [--save_mask SAVE_MASK] [--extract_tile] [--mask_type {prob,binary,inst}]  
                     [--group_out] [--group_by_file]

Run a lite version of HoverNet for Inference

optional arguments:
  -h, --help            show this help message and exit
  --data_pattern DATA_PATTERN
                        input wildcard
  --weight_path WEIGHT_PATH
                        Path of the pretrained model to load
  --net_mode {original,fast}
                        Infer Mode - chose from original and fast
  --default_num_type DEFAULT_NUM_TYPE
                        Default Number of Nucleus Types if Type Info is Missing.Num of nucleus types is len(type_info_json) - 1
  --pad_size PAD_SIZE   Padding to input images. The default value of the official branch is 48
  --resize_in RESIZE_IN
                        Resize tile after being read, e.g., if there is GPU VRAM limit
  --resize_out RESIZE_OUT
                        Resize mask if necessary, e.g., if the resize_in is set due toVRAM limitation and the mask of original size is desired.
  --batch_size BATCH_SIZE
                        Batch size for the dataloader during inference procedure
  --num_workers NUM_WORKERS
                        Num of workers in data loader
  --visible_device VISIBLE_DEVICE
                        visible device id of GPUs. comma separated string
  --type_info_path TYPE_INFO_PATH
                        Path of the type_info_json
  --export_folder EXPORT_FOLDER
                        Export Folder
  --save_json SAVE_JSON
                        Flag to Export geojson of nuclei for each mask
  --save_mask SAVE_MASK
                        Flag to Export Mask
  --extract_tile        Flag to export mask-overlaid tiles (e.g., for inspection)
  --mask_type {prob,binary,inst}
                        Flag to Export Mask - inst: pixel value as inst/class id.binary - binary mask of wherever nuclei presents.prob - the softmax score map of nuclei vs. background
  --group_out           Whether to group the outputs by the asterisk-matched subdirectoriesFor instance, if data_pattern is /A/*/*.png wherein the first asteriskcorresponds to any WSI name, with --group_out enabled the final outputloc will be      
                        [export_folder]/[any_matched_wsi_name]/[tile_name]
  --group_by_file       Additional to --group_out, whether group on individual input filee.g., for input pattern: *.png and a matach: fileA.png, a folder fileA will be created if this flag is set.No effect if group_out is False


```


## WSI + bbox JSON
Process the tiles from WSI files specified by bounding box coordinates (left/top/right/bottom) JSON files. The convention can be modified correspondingly in `hovernet_lite.infer.wsi_bbox`.

```
$ python -m hovernet_lite.infer.wsi_bbox -h
usage: hovernet_lite [-h] --data_pattern DATA_PATTERN [--weight_path WEIGHT_PATH] [--net_mode {original,fast}] [--default_num_type DEFAULT_NUM_TYPE] [--pad_size PAD_SIZE] [--resize_in RESIZE_IN] [--resize_out RESIZE_OUT] [--batch_size BATCH_SIZE]
                     [--num_workers NUM_WORKERS] [--visible_device VISIBLE_DEVICE] [--type_info_path TYPE_INFO_PATH] [--export_folder EXPORT_FOLDER] [--save_json SAVE_JSON] [--save_mask SAVE_MASK] [--extract_tile] [--mask_type {prob,binary,inst}]  
                     [--group_out] [--group_by_file] --bbox_pattern BBOX_PATTERN [--bbox_suffix BBOX_SUFFIX] [--tile_size TILE_SIZE] --nest_level NEST_LEVEL

Run a lite version of HoverNet for Inference

optional arguments:
  -h, --help            show this help message and exit
  --data_pattern DATA_PATTERN
                        input wildcard
  --weight_path WEIGHT_PATH
                        Path of the pretrained model to load
  --net_mode {original,fast}
                        Infer Mode - chose from original and fast
  --default_num_type DEFAULT_NUM_TYPE
                        Default Number of Nucleus Types if Type Info is Missing.Num of nucleus types is len(type_info_json) - 1
  --pad_size PAD_SIZE   Padding to input images. The default value of the official branch is 48
  --resize_in RESIZE_IN
                        Resize tile after being read, e.g., if there is GPU VRAM limit
  --resize_out RESIZE_OUT
                        Resize mask if necessary, e.g., if the resize_in is set due toVRAM limitation and the mask of original size is desired.
  --batch_size BATCH_SIZE
                        Batch size for the dataloader during inference procedure
  --num_workers NUM_WORKERS
                        Num of workers in data loader
  --visible_device VISIBLE_DEVICE
                        visible device id of GPUs. comma separated string
  --type_info_path TYPE_INFO_PATH
                        Path of the type_info_json
  --export_folder EXPORT_FOLDER
                        Export Folder
  --save_json SAVE_JSON
                        Flag to Export geojson of nuclei for each mask
  --save_mask SAVE_MASK
                        Flag to Export Mask
  --extract_tile        Flag to export mask-overlaid tiles (e.g., for inspection)
  --mask_type {prob,binary,inst}
                        Flag to Export Mask - inst: pixel value as inst/class id.binary - binary mask of wherever nuclei presents.prob - the softmax score map of nuclei vs. background
  --group_out           Whether to group the outputs by the asterisk-matched subdirectoriesFor instance, if data_pattern is /A/*/*.png wherein the first asteriskcorresponds to any WSI name, with --group_out enabled the final outputloc will be      
                        [export_folder]/[any_matched_wsi_name]/[tile_name]
  --group_by_file       Additional to --group_out, whether group on individual input filee.g., for input pattern: *.png and a matach: fileA.png, a folder fileA will be created if this flag is set.No effect if group_out is False
  --bbox_pattern BBOX_PATTERN
                        input wildcard for bbox coord files
  --bbox_suffix BBOX_SUFFIX
                        suffix of bbox.
  --tile_size TILE_SIZE
                        Optional. Define the size of bounding box before resize in case the bbox is cutoff at the boundary of images which causes inconsistent tile size in batches. If not set then use the bbox's size itself
  --nest_level NEST_LEVEL
                        Nest level of the bbox list.0 if the json contains List[Tuple[left, top, right, bottom]],1 if List[List[Tuple[left, top, right, bottom]], etc.
```
