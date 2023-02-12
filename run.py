import sys
import warnings
import json
import os

from src.dataset.make_dataset import *
from src.models.sentiment import *
from src.models.classifier import *

def main(targets):

    if 'test' in targets:
        fp = os.path.join('data/test', 'data.csv')

    try:

        # load dataframes 
        df_sentiment = load_df_sentiment(fp)
        df_classifier = load_df_relevance(fp, 'data/test/test_out.csv')

        # set up API keys
        api_key = targets[1]

        # run models
        sentiment = calc_sentiment(df_sentiment, api_key)
        classification = find_relevance(df_classifier, api_key)

        # evaluate
        accuracy = evaluate(classification)

        return sentiment, accuracy

    except Exception as e:
        print(e)

if __name__ == '__main__':

    targets = sys.argv[1:]
    warnings.filterwarnings('ignore')
    main(targets)