from joblib import load
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import datetime as dt

from textblob import TextBlob
from twitter_search import *
from nltk.corpus import stopwords

import chart_studio.plotly as py
import plotly as ply
import cufflinks as cf
from plotly.offline import *
from plotly.graph_objs import *
import seaborn as sns

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email import encoders
import os
import smtplib
from credentials import Credentials

def load_vectorizer(filepath):
    """_summary_
        function to load the vectorizer

    Args:
        filepath (string): path to the vectorizer file
    Returns:
        vectorizer (sklearn.feature_extraction.text.CountVectorizer): the vectorizer
    """
    with open(filepath, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_vectorizer

def load_model(filepath):
    """_summary_
        function to load the model
    Args:
        filepath (string): path to the model file
    Returns:
        model (sklearn.svm.SVC): the model
    """
    # Load the model from file using pickle
    with open(filepath, 'rb') as f:
        svm_model = pickle.load(f)
    return svm_model

def load_data(search_phrases=['chatgpt'], time_limit=0.1, max_tweets=100, min_days_old=1, max_days_old=2, geocode='39.8,-95.583068847656,2500km'):
    """_summary_
        function to load the data from twitter and save it to a json file
    Args:
        search_phrases (list, optional): _description_. Defaults to ['chatgpt'].
        time_limit (float, optional): _description_. Defaults to 0.1.
        max_tweets (int, optional): _description_. Defaults to 100.
        min_days_old (int, optional): _description_. Defaults to 1.
        max_days_old (int, optional): _description_. Defaults to 2.
        geocode (str, optional): _description_. Defaults to '39.8,-95.583068847656,2500km'.
    """
    
    
    # loop over search items,
    # creating a new file for each
    for search_phrase in search_phrases:

        print('Search phrase =', search_phrase)

        ''' other variables '''
        name = search_phrase.split()[0]
        
        json_file_root = name + '/'  + name
        os.makedirs(os.path.dirname(json_file_root), exist_ok=True)
        read_IDs = False
            
        # open a file in which to store the tweets
        if max_days_old - min_days_old == 1:
            d = dt.datetime.now() - dt.timedelta(days=min_days_old)
            day = '{0}-{1:0>2}-{2:0>2}'.format(d.year, d.month, d.day)
        else:
            d1 = dt.datetime.now() - dt.timedelta(days=max_days_old-1)
            d2 = dt.datetime.now() - dt.timedelta(days=min_days_old)
            day = '{0}-{1:0>2}-{2:0>2}_to_{3}-{4:0>2}-{5:0>2}'.format(
                d1.year, d1.month, d1.day, d2.year, d2.month, d2.day)
        json_file = json_file_root + '_' + day + '.json'
        if os.path.isfile(json_file):
            print('Appending tweets to file named: ',json_file)
            read_IDs = True
            
        # authorize and load the twitter API
        api = load_api()
            
        # set the 'starting point' ID for tweet collection
        if read_IDs:
            # open the json file and get the latest tweet ID
            with open(json_file, 'r') as f:
                lines = f.readlines()
                max_id = json.loads(lines[-1])['id']
                print('Searching from the bottom ID in file')
        else:
            # get the ID of a tweet that is min_days_old
            if min_days_old == 0:
                max_id = -1
            else:
                max_id = get_tweet_id(api, days_ago=(min_days_old-1))
        # set the smallest ID to search for
        since_id = get_tweet_id(api, days_ago=(max_days_old-1))
        print('max id (starting point) =', max_id)
        print('since id (ending point) =', since_id)
            


        ''' tweet gathering loop  '''
        start = dt.datetime.now()
        end = start + dt.timedelta(hours=time_limit)
        count, exitcount = 0, 0
        while dt.datetime.now() < end:
            count += 1
            print('count =',count)
            # collect tweets and update max_id
            tweets, max_id = tweet_search(api, search_phrase, max_tweets,
                                        max_id=max_id, since_id=since_id,
                                        geocode=geocode)
            # write tweets to file in JSON format
            if tweets:
                write_tweets(tweets, json_file)
                exitcount = 0
            else:
                exitcount += 1
                if exitcount == 3:
                    if search_phrase == search_phrases[-1]:
                        sys.exit('Maximum number of empty tweet strings reached - exiting')
                    else:
                        print('Maximum number of empty tweet strings reached - breaking')
                        break

def preprocess(search_phrases=['chatgpt'], vectorizer_path='tfidf_vectorizer.pkl'):
    """_summary_

    Args:
        search_phrases (list, optional): topics to search for. Defaults to ['chatgpt'].
        vectorizer_path (str, optional): path to vectorizer. Defaults to 'tfidf_vectorizer.pkl'.

    Returns:
        scipy.sparse._csr.csr_matrix: _description_
    """
    
    twt = pd.DataFrame(columns=['date', 'text'])
    for folder in search_phrases:
        files = os.listdir(folder)
        for file in files:
            file_name=folder+'/'+file
            twt1=pd.read_json(file_name, lines=True)
            twt1=twt1[['created_at','text']]
            twt=pd.concat([twt,twt1],ignore_index=True)


    twt['text']=twt['text'].str.lstrip('0123456789')
    #lower casing 
    twt['text']=twt['text'].apply(lambda a: " ".join(a.lower() for a in a.split()))
    #remove punctuation
    twt['text']=twt['text'].str.replace('[^\w\s]','')
    #remove stopwords
    stop = stopwords.words('english')
    twt['text']=twt['text'].apply(lambda a: " ".join(a for a in a.split() if a not in stop))
    Xpredict=twt['text']
    #tf-idf
    tv = load_vectorizer(vectorizer_path)
    predict_tfidf = tv.transform(Xpredict)
    return twt, predict_tfidf

def predict(classifier,predict_tfidf, model_path='svm_model.pkl'):
    """_summary_

    Args:
        classifier (sklearn.svm.SVC): _description_
        predict_tfidf (scipy.sparse._csr.csr_matrix): _description_
        model_path (str, optional): _description_. Defaults to 'svm_model.pkl'.

    Returns:
        _type_: _description_
    """
    return classifier.predict(predict_tfidf)

def predict_sentiments(twt):
    """_summary_
        function to predict the sentiments
    """
    twt['sentiment']=twt['text'].apply(lambda x: TextBlob(x).sentiment[0] )
    def function (value):
        if value['sentiment']>0:
            return 'positive'
        elif value['sentiment']<0:
            return 'negative'
        else:
            return 'neutral'
    twt['sentiment_label']=twt.apply(lambda x: function(x),axis=1)
    return twt

def generate_attachments(twt, search_phrases=['chatgpt']):
    init_notebook_mode(connected=True)
    cf.set_config_file(offline=True, world_readable=True, theme='white')

    sentiment_df=pd.DataFrame(twt['sentiment_label'].value_counts().reset_index())
    sentiment_df.columns=['sentiment','count']

    sentiment_df["percentage"]=100*sentiment_df["count"]/sentiment_df["count"].sum()
    sentiment_Max=sentiment_df.iloc[0,0]
    sentiment_percent=str(round(sentiment_df.iloc[0,2],2))+"%"
    # plot pie chart for the sentiment_df dataframe sentiment vs count
    labels = sentiment_df['sentiment']
    values = sentiment_df['count']
    colors = ['#FEBFB3', '#E1396C', '#96D38C']
    trace = ply.graph_objs.Pie(labels=labels, values=values, textinfo='value',hoverinfo='label+percent',
                    textfont=dict(size=20),
                    marker=dict(colors=colors,
                                line=dict(color='#000000', width=2)))
    layout = ply.graph_objs.Layout(title="Sentiment Distribution of Tweets on "+search_phrases[0])
    fig = ply.graph_objs.Figure(data=[trace], layout=layout)
    # fig.show()
    ply.offline.plot(fig, filename='sentiment_distribution.html')
    
    init_notebook_mode(connected=True)
    cf.set_config_file(offline=True, world_readable=True, theme='white')

    emotion_df=pd.DataFrame(twt['emotion'].value_counts().reset_index())
    emotion_df.columns=['emotion','count']
    emotion_df=pd.DataFrame(emotion_df)
    emotion_df["percentage"]=100*emotion_df["count"]/sentiment_df["count"].sum()
    mapper = {0: 'anger', 1: 'fear', 2: 'joy', 3: 'sadness'}
    emotion_df["emotion"]=emotion_df['emotion'].apply(lambda x: mapper[x])
    emotion_Max=emotion_df.iloc[0,0]
    emotion_percent=str(round(sentiment_df.iloc[0,2],2))+"%"
    fig=emotion_df.iplot(kind="pie",labels="emotion",values="count",pull=.2,hole=.2,colorscale='reds',textposition='outside',colors=['red','green','purple','orange'],title="Emotion Analysis of Tweets on"+search_phrases[0] ,world_readable=True,asFigure=True)
    ply.offline.plot(fig,filename='emotion.html')
    
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    result = pd.crosstab(twt['emotion'], twt['sentiment_label'])
    plt = result.plot.bar(stacked=True, sort_columns = True)
    plt.legend(title='Sentiment_label')
    plt.figure.savefig('sentiment_label.png', dpi=400)
    
def generate_email(emailing_list, search_phrases):
    dir_path = os.getcwd()
    files = ["emotion.html", "sentiment_distribution.html", "sentiment_label.png"]
    company_dict = emailing_list
    password = Credentials.PASSWORD.value
    for value in company_dict:
        subject = 'Emotion Detection and Sentiment Analysis Report'
        from_address = Credentials.EMAIL.value
        to_address = value
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = from_address
        msg['To'] = to_address
        body = "Hi \n Please find the attached report for the Emotion Detection and Sentiment Analysis on "+search_phrases[0]+" \n Thanks"
        msg.attach(MIMEText(body, 'plain'))
        for f in files:
            file_location = os.path.join(dir_path, f)
            attachment = MIMEApplication(open(file_location, "rb").read(), _subtype="txt")
            attachment.add_header('Content-Disposition', "attachment", filename=f)
            msg.attach(attachment)
        stmp = smtplib.SMTP('smtp.gmail.com', 587)
        stmp.connect('smtp.gmail.com', 587)
        stmp.ehlo()
        stmp.starttls()
        stmp.ehlo()
        stmp.login(from_address, password)
        text = msg.as_string()
        stmp.sendmail(from_address, to_address, text)
        stmp.quit()
        print("Email Sent Successfully")

def main(search_phrases=['chatgpt']):
    vectorizer_path = 'tfidf.pkl'
    model_path = 'svm_model.pkl'
    time_limit = 0.016666666666666666
    max_tweets = 50
    min_days_old = 1
    max_days_old = 2
    geocode = '39.8,-95.583068847656,2500km'
    load_data(search_phrases, time_limit, max_tweets, min_days_old, max_days_old, geocode)
    twt , predict_tfidf = preprocess(search_phrases, vectorizer_path)
    classifier = load_model(model_path)
    twt["emotion"]= predict(classifier,predict_tfidf, model_path)
    twt = predict_sentiments(twt)
    generate_attachments(twt)
    generate_email(emailing_list=["daliselmi30@gmail.com"], search_phrases=search_phrases)

if __name__ == '__main__':
    main()