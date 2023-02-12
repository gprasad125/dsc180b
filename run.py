import sys
import warnings
import json
import os

from src.dataset.make_dataset import *
from src.models.sentiment import *

def main(targets):

    if 'test' in targets:
        fp = os.path.join('data/test', 'data.csv')

    try:
        df = load_df_sentiment(fp)
        api_key = targets[1]
        out = calc_sentiment(df, api_key)
        return out

    except Exception as e:
        print(e)

if __name__ == '__main__':

    targets = sys.argv[1:]
    warnings.filterwarnings('ignore')
    main(targets)