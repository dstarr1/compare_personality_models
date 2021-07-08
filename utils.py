"""
Shared utility functions.
"""
import json


def read_local_config(config_fname):
    ''' Read configs from local, non-git commited json file.
    '''
    with open(config_fname) as fp:
        local_config = json.load(fp)
    return local_config
