# Analyzing U.S. Congressional Tweets with OpenAI GPT-3

# Repository for the Spring 2023 Quarter (DSC180b)

This project covers Tweet sentiment analysis for Tweets originating from US Congresspeople as it relates to China.
This is an extension to Quarter 1's project found for each researcher below:

[Annie's Q1 codebase](https://github.com/AnnieeeeeF/DSC180A_Project1)

[Gokul's Q1 codebase](https://github.com/gprasad125/dsc180a_project)

This project continues the work by exploring the same topic through the lens of a Large Language Model (LLM). 

## Necessary Configurations:

You will *need* an API key from OpenAI to utilize the GPT-3 model.
Sign up for an account and get a key [here](https://openai.com/api/)

You can then pass your API Key to our scripts in one of two ways:

1. Export your key by running the following in your command line:

`export OPENAI_API_KEY=...`

2. Create a .env file in the root directory and paste in your key like so:

`OPENAI_API_KEY=...`

## Data Source:

Raw data can be found [here](https://drive.google.com/drive/u/1/folders/1VSYdGh12UNVNhfxbSeHRdANvHr5xF8Ea). 
Download the file `SentimentLabeled_10112022.csv`, and place it inside the `data/raw` directory. 

You can then run the `run.py` file with the following targets:
- `test`: runs the file on man-made test data
- `data` / `all`: runs the file on Twitter-API sourced data.

## Explanation of File Structure:

### 📁 Folders:

#### config
Contains JSON configuration for optimized & group-selected models. 

#### data
Contains the data for and from the project, divided as such:
- raw: the base uncleaned data
- out: the output cleaned data used for visualizations and modeling
- test: test data used to debug the Python scripts
- results: visuals generated from the EDA and modeling, formatted as PNGs

#### notebooks
Contains initial Jupyter Notebooks for EDA / Modeling.
Not entirely cleaned up yet. Cleaned versions of this code will be found inside our `src` folder.

#### src
Contains the Python scripts needed to run the project, divided as such:
- data: 
    - `make_dataset.py` cleans and processes the raw data
- models: 
    - `classifier.py`: GPT-3 powered classifier to find "relevant" Tweets (i.e, Tweets about Chinese governmental impact on America.)
    - `sentiment.py`: GPT-3 powered sentiment scorer to find "emotion" of Tweet (i.e, is a Tweet favorable or negative towards China?)
- visuals: 
    - `eda.py`: Generates summary visuals for the two cleaned dataframes going into modeling. Not the full EDA of the dataset. For that, check under `notebooks/EDA.ipynb`
- notebooks:
    - `nb_functions.py`: All necessary functions for notebook report + visuals. Uses the code from other folders with slight modifications to fit an ipynb environment. 

### 📜 Files:

#### run.py
Baseline Python script to run via CLI with targets.
Current targets include `test` (`all`) and `data`. 

    - Creates cleaned data file
    - Generates exploratory visuals and saves them
    - Runs models on data

### requirements.txt
Necessary Python packages to install via `pip install -r requirements.txt`

