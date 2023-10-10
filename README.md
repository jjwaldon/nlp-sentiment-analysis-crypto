# Crypto Sentiment Analysis

This repository contains a Python script for performing sentiment analysis on Ethereum and blockchain-related tweets obtained from Twitter using the `sctweet` scraping library. The sentiment analysis is conducted using the `TextBlob` library, and the analyzed data is presented through various visualizations using `matplotlib`.

## Installation and Setup

1. Install Python 3.10+
2. Prepare required Python libraries using pip and the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Clone this repository:
4. Insert the path to your CSV file containing Twitter data in the Python script:

```python
path = r"INSERT PATH HERE"
```

## Usage

1. Update the `path` variable with the correct file path to your CSV file containing Twitter data.
2. Run the Python script using the following command:

```bash
python sentiment_analysis.py
```

3. Run the script, depending on your hardware this can take a moment.


## Script Explanation

- The Python script processes the Twitter data, cleans the tweets, and performs sentiment analysis.
- Cleaned tweets are analyzed for subjectivity and polarity to determine sentiment.
- Results are visualized using scatter plots, pie charts, and bar charts.
- 
![Figure_1](https://github.com/Dviqel/nlp-sentiment-analysis-crypto/assets/147337604/54d2f188-3c31-45c8-9d30-4b2b079c0b58)

![Figure_2](https://github.com/Dviqel/nlp-sentiment-analysis-crypto/assets/147337604/303bca08-cfc7-43ae-80c1-88b1d2306d37)

![Figure_3](https://github.com/Dviqel/nlp-sentiment-analysis-crypto/assets/147337604/c00dd44f-6636-44ab-9657-b4f44cb70a9a)


