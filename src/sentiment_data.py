import pandas as pd

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

def load_csv(path):
    """
    This function takes in the path of csv file, loads csv into dataframe,
    and returns the cleaned dataframe
    """

    df_raw = pd.read_csv(path)
    df_cleaned = clean_data(df_raw)
    df_cleaned['sentiment'] = df_cleaned.SentimentScore.apply(score_to_label)

    return df_cleaned