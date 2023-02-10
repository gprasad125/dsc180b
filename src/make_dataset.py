import pandas as pd

####### CLEANING FUNCTIONS ########

def find_relevance(bucket):
    """
    Finds relevance of Tweet from man-made encoding.
    Bucket 1 contains relevant Tweets, and all others are irrelevant.
    """

    if bucket == "1":
        return True

    return False



###################################


def load_df(inpath, outpath):
    """
    Loads raw data from an inpath. Dependent on command line system arguments.
    Processes data with cleaning functions, and saves it to a CSV file for ML & EDA.
    """

    # load csv & specify columns
    df = pd.read_csv(inpath)
    df = df[["text", "SentimentScore", "country", "Bucket"]]

    # clean columns
    df["Bucket"] = df["Bucket"].apply(find_relevance)

    # save cleaned df to new csv file
    df.to_csv("outpath", index = False)