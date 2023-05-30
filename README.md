# Real-time Tweets Sentiment Analysis and Emotion Detection

![twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)
![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

This project is an end-to-end pipeline that provides powerful insights about any topic or product through real-time tweets sentiment analysis and emotion detection.

This is a group project for the course **Professional Personal Project** at the **National Institute of Applied Science and Technology**, Tunisia.

## Description

This project explores **sentiment analysis** and **emotion detection** along with **text preprocessing** and **feature engineering methods**. 

We are exploring, through this project, different machine learning techniques to train classifier models and evaluate using a confusion matrix. 
We are also pulling real-time data from Twitter using Twitter API and `tweepy` package, and predict sentiment and emotions to generate insights. Finally, we generate a script for an **automated reporting**, which sends reports to a given set of e-mail addresses.

## Files

The project consists of the following files:

- [auto.py](/auto.py): Contains functions for loading models, vectorizers, data, and performing predictions. These functions are combined to create an automated reporting process.

- [job.py](/job.py): Implements a scheduler to run auto.py daily.

- [twitter_search.py](/twitter_search.py): Contains functions related to data fetching through Tweepy and the Twitter API.
   
- [svm_model.pkl](/svm_model.pkl): A trained SVM model for sentiment analysis.
   
- [tfidf.pkl](/tfidf.pkl): A trained TF-IDF vectorizer for sentiment analysis.

- [requirements.txt](/requirements.txt): Lists all the packages required for the project.

- [Step-by-step-guide.ipynb](/Step-by-step-guide.ipynb): A Jupyter notebook that provides a step-by-step guide for the project. It includes the code and explanations.

- [eng_dataset.csv](/eng_dataset.csv): A dataset containing 7.1k tweets in English, which is used for training the sentiment analysis model.

## Getting Started

To run the project, follow the steps below:

1. clone the repository by using the following command:

    ```bash
    git clone https://github.com/mrdaliselmi/Real-time-Tweets-Sentiment-Analysis-and-Emotion-Detection-Pipeline
    ```

2. Install the required packages listed in [requirements.txt](requirements.txt) using the following command:
    
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure that you have the necessary credentials and access to the Twitter API. If not, you can apply for a developer account [here](https://developer.twitter.com/en/apply-for-access).
   
4. Create a file credentials.py having an enum class with all the credentials

5. Set up the necessary configurations in [job.py](/job.py) to specify the schedule and other parameters for running the automated reporting process.

6. Run [job.py](/job.py) to start the automated reporting process. **Alternatively**, if you prefer to use a **cron job**, you can schedule it using the cron syntax. For example, to run the script every day at 8 AM, you can use the following command:

    ```bash
    0 8 * * * python3 /path/to/auto.py
    ```
## Project overview: Methodology and Approach

### Project Pipeline

The following flowchart illustrates the pipeline of the project:

![flowchart](/flowchart.png)

### Data Pre-processing

The data pre-processing step includes the following steps:
- **Case Normalization**: Convert all text to lowercase.
- **Data Cleaning**: Remove special characters, and ponctuation.
- **Stopwords Removal**: Remove stopwords from the text.
- **Correct Spelling**: Correct spelling mistakes in the text.
- **Stemming**: Reduce words to their root form.

### Feature Engineering

The feature engineering step includes the following steps:
- **TF-IDF Vectorization**: Convert text to a matrix of TF-IDF features.
- **Count Vectorization**: Convert text to a matrix of token counts.

### Model Training and Evaluation

The model training and evaluation step includes the following steps:
- **Train/Test Split**: Split the data into training and testing sets.
- **Model Training**: Train a classifier model using the training set.
- **Model Evaluation**: Evaluate the model using the testing set.
- **Confusion Matrix**: Generate a confusion matrix to evaluate the model's performance.
  
|       | precision | recall | f1-score | support |
|-------|-----------|--------|----------|---------|
|   0   |   0.92    |  0.87  |   0.89   |   340   |
|   1   |   0.83    |  0.96  |   0.89   |   451   |
|   2   |   0.97    |  0.90  |   0.93   |   323   |
|   3   |   0.86    |  0.77  |   0.81   |   307   |
|   accuracy  |           |        |   0.88   |   1421  |
|  macro avg |   0.89    |  0.87  |   0.88   |   1421  |
|weighted avg|   0.89    |  0.88  |   0.88   |   1421  |

### Real-time Data Fetching

- **Twitter API**: Use the Twitter API to fetch real-time tweets.
- **Tweepy**: Use the `tweepy` package to fetch tweets from the Twitter API.
- **Data Pre-processing**: Pre-process the fetched tweets using the same steps as well as the same vectorizer as the data pre-processing step.

### Prediction and Reporting

- **Prediction**: Predict the sentiment and emotions of the fetched tweets using `TextBlob` and the trained model.

- **Reporting**: Generate a report containing the following information:
    - Emotion distribution pie Chart
    - Sentiment distribution pie Chart
    - Statistics about the number of tweets, the number of positive, negative, and neutral tweets, and the number of tweets for each emotion.
    - A stacked bar plot as a combination of the sentiment and emotion distributions.

## Additional Notes

- The project's code is provided in the Jupyter notebook [step-by-step-guide.ipynb](/Step-by-step-guide.ipynb), which contains detailed explanations and code snippets for each step.

- The project utilizes various Python packages such as pandas, NLTK, scikit-learn, Matplotlib, seaborn, and xgboost. Make sure to install these packages, as mentioned in the requirements.txt file.

## Team Members
- [Dali Selmi](https://github.com/mrdaliselmi)
- [Nour Eddine Ben Nejma](https://github.com/Lakhdher)
- [Walid Sboui](https://github.com/walid192)
- [Mehdi Cherif](https://github.com/mehdixlabetix)

## References

- https://www.kaggle.com/datasets/faisalsanto007/isear-dataset
- Akshay Kulkarni, Adarsha Shivananda and Anoosh Kulkarni
- A. Kulkarni et al., Natural Language Processing Projects, https://doi.org/10.1007/978-1-4842-7386-9_2