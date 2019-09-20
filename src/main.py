from preprocess.rawdata import get_data
from preprocess.rawdata import display_history
import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.abspath('../..')))+'/secret')
import config

#get_data(config)
display_history()
