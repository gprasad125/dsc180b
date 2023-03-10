import pandas as pd
import matplotlib.pyplot as plt

# import sys
# sys.path.append('../dataset/')

# from make_dataset import *

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

### Part 2: Load Data 
def load_full_data(path):

    df = pd.read_csv(path)
    df['term_partisanship'] = df['term_partisanship'].apply(get_party)
    df['term_state'] = df['term_state'].apply(get_state)

    return df

def load_df_relevance(df):
    """
    Loads raw data from an inpath. Dependent on command line system arguments.
    Processes data with cleaning functions, and returns it to be used. 
    """

    # load csv & specify columns
    df = df[["text", "country", "Bucket"]]

    # clean columns
    df["Relevant"] = df["Bucket"].apply(find_relevance)

    # return
    return df

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


