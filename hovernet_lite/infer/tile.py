from hovernet_lite.infer._helper import main_process, tile_datasset
from hovernet_lite.util.misc import get_logger
from ..args import get_opt

####


if __name__ == '__main__':
    opt = get_opt()
    logger_name = __name__
    logger = get_logger(logger_name)
    dataset = tile_datasset(opt, logger)
    main_process(opt, logger, dataset)

