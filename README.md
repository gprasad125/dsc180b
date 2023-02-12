# dsc180b

# Repository for the Spring 2023 Quarter (DSC180b)

This project covers Tweet sentiment analysis for Tweets originating from US Congresspeople as it relates to China.
This is an extension to Quarter 1's project found for each researcher below:
Annie's Q1 codebase
Gokul's Q1 codebase

This project continues the work by exploring the same topic through the lens of a Large Language Model (LLM) like GPT-3. 

## Necessary Configurations:

You will *need* an API key from OpenAI to utilize the GPT-3 model.
Sign up for an account and get a key [here](https://openai.com/api/)

You can then pass your API Key in as a command line argument to our run script like so:
```
python run.py [target_1] [api key value]
```
For example, you can run the code on test data by running this script:
```
python run.py test [api key value]
```

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
- data: `make_dataset.py` cleans and processes the raw data
- models: 
    - classifier.py: GPT-3 powered classifier to find "relevant" Tweets (i.e, Tweets about Chinese governmental impact on America.)
    - sentiment.py: GPT-3 powered sentiment scorer to find "emotion" of Tweet (i.e, is a Tweet favorable or negative towards China?)

### 📜 Files:

#### run.py
Baseline Python script to run via CLI with targets.
Current targets include `test` (`all`) and `data`. 

    - Creates cleaned data file
    - Generates exploratory visuals and saves them
    - Runs models on data

### requirements.txt
Necessary Python packages to install via `pip install -r requirements.txt`

