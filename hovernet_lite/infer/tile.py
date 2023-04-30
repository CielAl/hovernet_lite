from hovernet_lite.infer._helper import main_process
from hovernet_lite.util.misc import get_logger
from hovernet_lite.args import BaseArgs
from hovernet_lite.infer_manager.dataset_proto import SimpleSeqDataset, pil_loader, batch_pre_processor
import sys
import glob

####


class TileArgs(BaseArgs):

    def additional_args(self):
        ...


def tile_dataset(opt_in, logger_in) -> SimpleSeqDataset:
    files = glob.glob(opt_in.data_pattern)
    logger_in.info(f'{len(files)} files have been read.')
    preproc = batch_pre_processor(opt_in.pad_size)
    # file_parts = [os.path.splitext(os.path.basename(x))[0] for x in files]
    # prefix_list = [os.path.basename(os.path.dirname(x)) for x in files]
    # prefix = [os.path.join(x, y) for x, y in zip(prefix_list, file_parts)]
    prefix_list = SimpleSeqDataset.generate_path_prefix(files, opt_in.data_pattern, opt_in.group_out,
                                                        opt_in.group_by_file)
    logger_in.debug(f"sample prefix: {prefix_list[0]}")
    return SimpleSeqDataset(files, loader=pil_loader, transforms=preproc, prefix_list=prefix_list)


if __name__ == '__main__':
    argv = sys.argv[1:]
    tile_args = TileArgs(argv)
    opt = tile_args.get_opts()
    logger_name = __name__
    logger = get_logger(logger_name)
    dataset = tile_dataset(opt, logger)
    main_process(opt, logger, dataset)

