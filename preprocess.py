__author__ = 'tureoyz'

import os
import sys
import pandas as pd
data_path = os.path.join(os.path.dirname(__file__), 'data')

def load_feat_type_info(fname = os.path.join(data_path, 'features_type.csv')):
    f = open(fname, 'r')
    i = 0
    type_info = dict()
    for line in f:
        if i != 0:
            strs = line.rstrip(' \r\n').split(',')
            type_info[strs[0]] = strs[1]
        i += 1
    return type_info

