import pandas as pd
import matplotlib.pyplot as plt

from varname import nameof

def get_upload_path(variable):

    path = 'data/results/' + nameof(variable) + '.png'
    return path

def generate_visuals(df_sentiment, df_relevance):

    # create visuals
    sentiment_hist = df_sentiment['sentiment'].value_counts().plot(kind='bar')
    plt.title('Distribution of Sentiment')
    plt.savefig('data/results/' + nameof(sentiment_hist) + '.png')

    plt.figure()

    relevance_hist = df_relevance['Relevant'].value_counts().plot(kind='bar')
    plt.title('Distribution of Relevance')
    plt.savefig('data/results/' + nameof(relevance_hist) + '.png')

    return