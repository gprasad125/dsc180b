import pandas as pd
import openai
import time

def classify_sentiment(df, prompt):
    
    i = 0
    sentiments = []
    
    while i < len(df):
        
        row = df.iloc[i]
        tweet = row.text
        tweet_id = row.id
    
        prompt_text = prompt + tweet
        
        try:
    
            response = openai.Completion.create(
              model="text-davinci-003",
              prompt=prompt_text,
              max_tokens=256,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
            sentiment = response['choices'][0]['text'].strip()
            sentiments.append((tweet_id,sentiment))
            i += 1
            
        except Exception as e:
            time.sleep(60)
    
    return sentiments

def extract_sentiment(text):
    if text[0] != 'A':
        start_idx = text.find('Answer')
        text = text[start_idx:]
    text = text.lower().splitlines()[0]
    if 'negative' in text:
        return 'negative'
    elif 'positive' in text:
        return 'positive'
    elif 'neutral' in text:
        return 'neutral'

def classify_sentiment(df, api_key):
    question_prompt = """Determine the given tweet's sentiment toward China. Return either positive, neutral, or negative.

    Example 1: "The humanitarian, security and health threats personified in the coronavirus are being exacerbated by \
    authoritarian socialist policies and the dishonestly of foreign aggressors and abusers, like China. \
    https://t.co/CKAtP9qOUT #coronavirus". 
    Answer: negative.
    Reason: The tweet uses explicit negative sentiment towards China through the words such as \
    “dishonest”, “aggressor”, and “abuser”.

    Example 2: "The WH makes a targeted effort to counter China's Military-Fusion Strategy by denying visas to \
    those grad students supporting Beijing’s effort to replace US as the leading global power.Chinese \
    students are still welcome, but not those working for the govt. https://t.co/P8bMTEPMWr". 
    Answer: neutral 
    Reason: The first sentence is a factual statement by itself and the second sentence \
    is a partial comment on the policy with a mixed implicit sentiment.

    Example 3: "@rollcall Unclear why effort to create a scientifically valid vaccine can’t be successful \
    without China/US involvement. Speaks to how wide split btwn the US and China has become and how that split \
    undermines their leadership. US should be the first of the two to join and show leadership". 
    Answer: positive 
    Reason: it shows support for cooperation with China to develop a vaccine.

    Given these examples, value the following tweet: """

    model_output = classify_sentiment(df, question_prompt)
    output_df = pd.DataFrame(model_output, columns=['id', 'model_output'])
    model_df = sampled_df.merge(output_df, on='id', how='inner')
    model_df['outputed_sentiment'] = model_df.model_output.apply(extract_sentiment)

    accuracy = np.mean(model_df['sentiment'] == model_df['outputed_sentiment'])
    print('fThe accuracy of classification on this dataset is : {accuracy}.')

    return model_df