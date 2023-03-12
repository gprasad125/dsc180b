from doctest import DocFileCase
import pandas as pd
import matplotlib.pyplot as plt

### Part 1: Cleaning Functions
def get_party(party):
    
    if not pd.isnull(party):
        if 'Republican' in party:
            party = 'Republican'
        elif 'Democrat' in party:
            party = 'Democrat'
        
    return party

def get_state(state):
    
    if not pd.isnull(state):
        if '{' in state:
            start = state.index('{') + 1
            end = state.index('}')
            state = state[start:end]

    return state

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

### Part 2: Load Data 
def load_full_data(path):

    df = pd.read_csv(path)
    df['term_partisanship'] = df['term_partisanship'].apply(get_party)
    df['term_state'] = df['term_state'].apply(get_state)

    return df

def load_df_relevance(df):
    """
    Loads raw data from an dataframe. Dependent on command line system arguments.
    Processes data with cleaning functions, and returns it to be used. 
    """

    # load csv & specify columns
    df = df[["text", "country", "Bucket"]]

    # clean columns
    df["Relevant"] = df["Bucket"].apply(find_relevance)

    # return
    return df

def load_df_sentiment(df):
    """
    This function takes in the path of csv file, loads csv into dataframe,
    and returns the cleaned dataframe
    """

    df_cleaned = clean_data(df)
    df_cleaned['sentiment'] = df_cleaned.SentimentScore.apply(score_to_label)

    # save cleaned df to new csv file
    return df_cleaned

# Part 3: Generate Visuals
def visualize(info, df):

    plt.figure()

    if info == 'party':

        fig, ax = plt.subplots()
        df['term_partisanship'].value_counts().plot(kind='bar')
        plt.title('Distribution of Parties in Data')
        plt.xticks(rotation=0)
        plt.xlabel('Party')
        plt.ylabel('Number of Tweets')

        return fig, ax
    
    elif info == 'state':

        fig, ax = plt.subplots(figsize=(15, 5))
        df.groupby('term_state')['country'].size().plot(kind = 'bar')
        plt.title('Number of Tweets by State')
        plt.xlabel('State')
        plt.ylabel('Number of Tweets')

        return fig, ax

    elif info == 'relevance':
        fig, ax = plt.subplots()
        df.groupby('Relevant').size().plot(kind='bar')
        plt.title('Distribution of Relevance')
        plt.xticks(rotation=0)
        plt.xlabel('Relevance')
        plt.ylabel('Count')

        return fig, ax

    elif info == 'sentiment':
        fig, ax = plt.subplots()
        counts = df['SentimentScore'].value_counts().to_frame().reset_index()
        counts = counts.sort_values(by='index')
        counts['sentiment'] = counts['index'].apply(score_to_label)
        colors = {'negative': 'r', 'neutral': 'g', 'positive': 'b'}
        plt.bar(counts['index'], counts.SentimentScore, color=[colors[i] for i in counts['sentiment']])
        plt.title("Distribution of Sentiments")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Count")
        handles = [plt.Rectangle((0,0),1,1, color=colors[l]) for l in ['negative', 'neutral', 'positive']]
        plt.legend(handles, ['negative', 'neutral', 'positive'], title="Sentiment")

        return fig, ax

    elif info == 'overtime':

        fig, ax = plt.subplots(figsize=(8,5))

        df['date'] = pd.to_datetime(df['date'])
        df = df.drop_duplicates()
        df['term_partisanship'] = df['term_partisanship'].str.strip('{}')
        df = df[(df['country'] == 'China') & (df['SentimentScore'].isnull() == False)]

        # keep rows that are not duplicate and with an integer sentiment score
        df_cleaned = df.drop_duplicates(subset='id', keep=False)
        df_cleaned = df_cleaned[df_cleaned.SentimentScore.apply(float.is_integer)]

        df_bucket1 = df[df['Bucket'] == '1'].set_index('date')
        for p in df['term_partisanship'].unique():
            dff = df_bucket1[df_bucket1['term_partisanship'] == p]
            dff = dff[dff['country'] == 'China']
            df_party = dff.groupby(dff.index.year)['SentimentScore'].mean().to_frame()
            plt.plot(df_party.index, df_party.SentimentScore, label=p)
        plt.legend(loc='upper right')
        plt.xlabel("Year")
        plt.ylabel("Average Sentiment Score")
        plt.ylim(1, 5)
        plt.title('Yearly Average Sentiment Score Toward China Per Political Party')

        return fig, ax


