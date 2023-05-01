import os.path
from typing import List, Tuple
import re
import json
import datetime

REG_BRACKET = r'\[.*?\]'
REG_ASTERISK = r'\*+'
REG_SINGLE = r'\?+'

REG_WILD = '|'.join([REG_BRACKET, REG_ASTERISK, REG_SINGLE])
DEFAULT_TIME_FORMAT = "%Y%m%d%H%M%S"


def get_timestamp(time_format=DEFAULT_TIME_FORMAT):
    return datetime.datetime.now().strftime(time_format)


def path_components(path: str, max_depth=65536) -> List[str]:
    """
    Find all components along a path string that can be parsed by glob.glob
    Args:
        path: Path string. Doesn't need to be an existing one.
        max_depth: maximum of depth to check from top (end) to bottom (root)
    Returns:
    """
    def __append_non_empty(str_list: List, s: str):
        if s != "":
            str_list.append(s)

    path_work = path
    components = []
    counter = 0
    while True:
        if counter >= max_depth:
            break
        path_prev = path_work
        path_work, directory = os.path.split(path_work)
        __append_non_empty(components, directory)
        # exit
        if path_work == path_prev:
            break
    __append_non_empty(components, path_work)
    counter += 1
    return components[::-1]


def find_wildcards(path: str) -> Tuple[List[str], List[int]]:
    components = path_components(path)
    matched_idx = [idx for idx, p in enumerate(components) if len(re.findall(REG_WILD, p)) > 0]
    return components, matched_idx


def load_json(uri: str):
    with open(uri, 'r') as root:
        return json.load(root)


def save_json(uri: str, data, indent=4):
    with open(uri, 'w') as root:
        return json.dump(data, root, indent=indent)
