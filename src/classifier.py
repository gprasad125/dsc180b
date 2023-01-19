import openai
import pandas as pd
import os

openai.api_key = os.getenv("openai_api")

prompt = "Determine if the following Tweet is True or False: \n"

