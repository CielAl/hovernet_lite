import argparse


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pattern", help="input wildcard", type=str)
    parser.add_argument("--weight_path", help="Path of the model to load", type=str,
                        default='./pretrained_models/hovernet_fast_monusac_type_tf2pytorch.tar')
    parser.add_argument("--net_mode", help="Infer Mode", type=str,
                        default='fast')
    parser.add_argument("--num_types", help="Number of Nucleus Types", type=int,
                        default=6)
    parser.add_argument("--pad_size", help="Padding to input images", type=int,
                        default=48)

    parser.add_argument("--batch_size", help="Batch Size", type=int,
                        default=6)

    parser.add_argument("--num_workers", help="Num of workers in data loader", type=int,
                        default=6)

    parser.add_argument("--visible_device", help="visible device id of GPUs. comma separated string", type=str,
                        default="0, 1")

    parser.add_argument("--type_info_path", help="Path of the type_info_json", type=str,
                        default='./type_info/type_info_monusac.json')
    parser.add_argument("--export_folder", help="Export Folder", type=str,
                        default='./sample_out/')

    parser.add_argument("--save_json", help="Flag to Export Json", type=bool,
                        default=False)

    parser.add_argument("--save_mask", help="Flag to Export Mask", type=bool,
                        default=True)

    opt, _ = parser.parse_known_args([])
    return opt

