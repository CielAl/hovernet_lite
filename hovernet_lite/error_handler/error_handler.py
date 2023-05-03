from typing import Callable, Union, List
import sys
import traceback
from hovernet_lite.logger import GlobalLoggers
from hovernet_lite.data_type import ErrorInfo


class ExceptionSignal:

    def __init__(self, info_dict: ErrorInfo):
        self.info_dict = info_dict


def remediate_call(func: Callable, logger_name, occurrence_str: Union[str, List[str]],
                   verbose: bool,  *args, **kwargs):
    """
    Execute the func using the args and kwargs. Log the error and return ErrorSignal if an exception is raised.
    Args:
        func:
        logger_name:
        occurrence_str:
        verbose: whether print the error
        *args:
        **kwargs:

    Returns:

    """
    try:
        return func(*args, **kwargs)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        if verbose:
            GlobalLoggers.instance().get_logger(logger_name).critical(f"{error_msg}")
            GlobalLoggers.instance().get_logger(logger_name).critical(f"{stack_trace}")
            GlobalLoggers.instance().get_logger(logger_name).critical(f"Error at: {occurrence_str}")
        info_dict = ErrorInfo(error_msg=error_msg, stack_trace=stack_trace,
                              name=occurrence_str)
        GlobalLoggers.instance().error_list.append(info_dict)
        return ExceptionSignal(info_dict)
