from preprocess.rawdata import get_data
from preprocess.rawdata import display_history
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('../..')))+'/secret')
import config

get_data(config)
#display_history()

'''
import pandas as pd

reg = pd.read_csv('../../secret/data/raw/reg.csv')
reg6 = pd.read_csv('../../secret/data/raw/reg2006.csv')
reg7 = pd.read_csv('../../secret/data/raw/reg2007.csv')
reg8 = pd.read_csv('../../secret/data/raw/reg2008.csv')

result = pd.concat([reg,reg6,reg7,reg8])
result.to_csv('demo_data.csv')
'''
