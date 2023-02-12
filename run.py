import sys
import warnings
import json
import os

from src.dataset.make_dataset import *
from src.models.sentiment import *
from src.models.classifier import *

def main(targets):

    # test data 
    if 'test' in targets:
        fp = os.path.join('data/test', 'data.csv')

    # all data 
    else:
        fp = os.path.join('data/raw', 'SentimentLabeled_10112022.csv')

    try:

        # load dataframes 
        df_sentiment = load_df_sentiment(fp)
        df_classifier = load_df_relevance(fp)

        # set up API keys
        api_key = targets[1]

        # run models & evaluate
        sentiment = calc_sentiment(df_sentiment, api_key)
        relevance = find_relevance(df_classifier, api_key)

        # return metrics of evaluation
        return sentiment, relevance

    except Exception as ex:
        exception_msg = "Exception Type: {0}. Arguments: \n{1!r}"
        message = exception_msg.format(type(ex).__name__, ex.args)
        print(message)

if __name__ == '__main__':

    targets = sys.argv[1:]
    warnings.filterwarnings('ignore')
    main(targets)