import openai
import pandas as pd
import os

openai.api_key = os.get_env("openai_api")

prompt = "Determine if the following Tweet is True or False: \n"

