import numpy as np
import openai

import seaborn as sns

def clean_answer(response):
    """
    Function to clean a GPT-3 response.
    Thanks to prompt, the answer is always at the start. Either TRUE or FALSE. 
    """
   
    if "False" in response:
        return False
    return True

def gpt3(tweet):

    prompt = """
    You are a machine that will be given a Tweet from a U.S. politician, and you will be asked to determine its relevancy to one of three countries. If it is relevant, return "True". Otherwise, return "False".
    
    The three possible countries are Canada, Iran, or China. 
    A "relevant" tweet would have:
    - They express sentiment towards the country of interest, either positive or negative.
    - They discuss how only ONE of these countries' governments is having and impact on American politics. 

    EXAMPLE TWEET:
    "They are also an assault on the American-led world order, and a disturbing premonition of an alternative world order—one controlled by the Chinese Communist Party and one that ends in Room 101."

    EXAMPLE ANSWER: 
    True

    EXPLANATION:
    This tweet is clearly about the Chinese government and its impact over American politics. Hence, the return value is True. 

    EXAMPLE TWEET:
    "JUST IN: House votes to block Obama from lifting Iran sanctions https://t.co/EFI5L9WjI4"

    EXAMPLE ANSWER: 
    False

    EXPLANATION:
    This tweet, while mentioning Iran, does not reflect any substance over Iranian political effects on America. Its ONLY focus is on America, not Iranian impacts on America, and is thus irrelvant. 

    EXAMPLE TWEET:
    "The President’s border-crossing permit for the A2A Railway Development Corp is a big boost for efforts to connect Alaska’s rich resources to a global market via freight rail through Canada. https://t.co/baSaeN9Lym"

    EXAMPLE ANSWER: 
    True

    EXPLANATION:
    This tweet is about how Canadian freight reils allow for American political expansion, and is thus relevant about one of our three options.

    EXAMPLE TWEET:
    Months ago, all 100 Senators, Democrats &amp; Republicans alike, passed a bill to stop the influence of the Chinese government-funded Confucius Institute in US schools. The Dem-led House continues to block the legislation.

    Why is Pelosi playing politics with our national security?

    EXAMPLE ANSWER: 
    False

    EXPLANATION: 
    This Tweet, while discussing Chinese political impact on America, does NOT contain sentiment towards China. Instead, it focuses its sentiment towards America. Therefore, it's irrelevant. 

    EXAMPLE TWEET:
    .@grahamblog: If war continues, how will Iran take us seriously re: nuclear program if U.S. does nothing about Assad? #MTP

    EXAMPLE ANSWER:
    False

    EXPLANATION:
    Again, this Tweet mentions Iran-US relations, but the sentiment expressed is towards America, not Iran. Therefore, it's irrelevant. 
    
    TWEET: \n
    """

    prompt += tweet

    try: 
        response = openai.Completion.create(
              model="text-davinci-003",
              prompt=prompt,
              max_tokens=256,
              top_p=1,
              frequency_penalty=0,
              presence_penalty=0
            )
        response = response['choices'][0]['text'].strip()
        return response
    except: 
        return "Error with GPT-3. Try again later. "

def find_relevance(df, api_key):
    """
    Apply GPT-3 completion to each Tweet in the dataset.
    Returns a dataframe with both labels and predictions. 
    """

    openai.api_key = api_key

    df["gpt3_answer"] = df["text"].apply(gpt3)
    df["gpt3_answer"] = df["gpt3_answer"].apply(clean_answer)

    accuracy = np.mean(df["gpt3_answer"] == df["Relevant"])
    print(f"The relevance accuracy on this dataset is: {accuracy}.")

    return accuracy
