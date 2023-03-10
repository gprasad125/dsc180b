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

def clean_data(df_raw):
    """
    This function takes in a Pandas DataFrame, drops extra columns from the original dataframe,
    and keep only rows that contain integer sentiment score."""

    df = df_raw.drop_duplicates()
    df = df[(df['country'] == 'China') & (df['SentimentScore'].isnull() == False)]
    df = df[['id', 'text', 'SentimentScore']]

    df_cleaned = df.drop_duplicates(subset='id', keep=False)
    df_cleaned = df_cleaned[df_cleaned.SentimentScore.apply(float.is_integer)]

    return df_cleaned

def score_to_label(score):
    """
    This function converts a 5-point scale sentiment score to its corresponding sentiment category.
    """
    if score <= 2:
        return 'negative'
    elif score == 3:
        return 'neutral'
    elif score > 3:
        return 'positive'

###################################


def load_df_relevance(path, outpath):
    """
    Loads raw data from an inpath. Dependent on command line system arguments.
    Processes data with cleaning functions, and returns it to be used. 
    """

    # load csv & specify columns
    df = pd.read_csv(path)
    df = df[["text", "country", "Bucket"]]

    # clean columns
    df["Relevant"] = df["Bucket"].apply(find_relevance)

    # save cleaned df to new csv file
    df.to_csv(outpath, index=False)
    return df

def load_df_sentiment(path, outpath):
    """
    This function takes in the path of csv file, loads csv into dataframe,
    and returns the cleaned dataframe
    """

    df_raw = pd.read_csv(path)
    df_cleaned = clean_data(df_raw)
    df_cleaned['sentiment'] = df_cleaned.SentimentScore.apply(score_to_label)

    # save cleaned df to new csv file
    df_cleaned.to_csv(outpath, index=False)
    return df_cleaned


