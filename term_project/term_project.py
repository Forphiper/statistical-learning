import pandas as pd
import numpy as np
from tqdm import tqdm

"""Read raw datas"""
stock_data = pd.read_csv("./raw_data/upload_DJIA_table.csv")
stock_data = stock_data[::-1]
stock_data.index = range(len(stock_data))

news_data = pd.read_csv("./raw_data/RedditNews.csv")
news_data = news_data[::-1]
news_data.index = range(len(news_data))

combined_data = pd.read_csv("./raw_data/Combined_News_DJIA.csv")

# Convert NaN values to string
combined_data = combined_data.fillna('NaN')

print(combined_data.shape)
print(combined_data.head(6))


"""Data preprocessing"""
rows = len(combined_data)
cols = 2 + 25 * 3
data = np.zeros((rows, cols))
df = pd.DataFrame(data)

df.iloc[:, 0] = combined_data["Date"]

for row_idx, row in stock_data.iterrows():
    if row[6] - row[1] >= 0:
        df.iloc[row_idx, 1] = 1
    else:
        df.iloc[row_idx, 1] = 0
        
df.iloc[:, 1] = df.iloc[:, 1].astype(int)
df = df.rename(columns={0: "date", 1: "label"})
for i in range(2, 27):
    df = df.rename(columns={i: f"model1_top{i-1}", i+25: f"model2_top{i-1}", i+50: f"model3_top{i-1}"})

#print(df.shape)
#print(df.head(10))
#print(df.columns)


"""Model1: cardiffnlp/twitter-roberta-base-sentiment"""
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []


    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task = 'sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#model.save_pretrained(MODEL)

for row_idx, row in tqdm(combined_data.iterrows(), total=len(combined_data)):
    for col_idx in range(2, 27, 1):
        #text = "100+ Nobel laureates urge Greenpeace to stop opposing GMOs"
        text = row[col_idx][2:-1]
        #print(text)
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        #print(scores)
        curr_score = scores[2] - scores[0]
        #print(curr_score)
        df.iloc[row_idx, col_idx] = round(curr_score, 4)
        """
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")
        """

print(df.iloc[:, 2:27].head())
print(df.iloc[:, 2:27].tail())


"""Model2: cardiffnlp/twitter-roberta-base-sentiment-latest"""
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#model.save_pretrained(MODEL)

for row_idx, row in tqdm(combined_data.iterrows(), total=len(combined_data)):
    for col_idx in range(2, 27, 1):
        text = row[col_idx][2:-1] 
        #text = "Covid cases are increasing fast!"
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        #print(scores)
        curr_score = scores[2] - scores[0]
        #print(curr_score)
        df.iloc[row_idx, col_idx+25] = round(curr_score, 4)
        """
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = config.id2label[ranking[i]]
            s = scores[ranking[i]]
            print(f"{i+1}) {l} {np.round(float(s), 4)}")
        """

print(df.iloc[:, 27:52].head())
print(df.iloc[:, 27:52].tail())

"""Model3: yiyanghkust/finbert-tone"""
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)

for row_idx, row in tqdm(combined_data.iterrows(), total=len(combined_data)):
    for col_idx in range(2, 27, 1):
        sentences = [row[col_idx][2:-1]]
        #sentences = ["growth is strong and we have plenty of liquidity"]
        #sentences = ["The market has gone down a lot"]
        results = nlp(sentences)
        #print(results)  #LABEL_0: neutral; LABEL_1: positive; LABEL_2: negative
        if results[0]["label"] == "Neutral":
            curr_result = 0
        elif results[0]["label"] == "Positive":
            curr_result = 1
        elif results[0]["label"] == "Negative":
            curr_result = -1
        #print(curr_result)
        df.iloc[row_idx, col_idx+50] = curr_result

df.iloc[:, 52:77] = df.iloc[:, 52:77].astype(int)
print(df.iloc[:, 52:77].head())
print(df.iloc[:, 52:77].tail())

df.iloc[:, 2:77] = df.iloc[:, 2:77].shift(1)


"""Add technical indicators"""
from ta import add_all_ta_features
import ta

stock_data_with_indicators = add_all_ta_features(stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False)

stock_data_with_indicators = stock_data_with_indicators.drop(stock_data_with_indicators.columns[:7], axis=1)

for column in stock_data_with_indicators.columns:
    if stock_data_with_indicators[column].isna().sum() > 50:
        stock_data_with_indicators = stock_data_with_indicators.drop(column, axis=1)
print(stock_data_with_indicators.shape)

num_cols = stock_data_with_indicators.shape[1]
mapping = {col: f'indicator{i+1}' for i, col in enumerate(stock_data_with_indicators.columns[:num_cols])}
stock_data_with_indicators = stock_data_with_indicators.rename(columns=mapping)


stock_data_with_indicators['open'] = stock_data['Open']
stock_data_with_indicators['high'] = stock_data['High']
stock_data_with_indicators['low'] = stock_data['Low']
stock_data_with_indicators['close'] = stock_data['Close']

# normalize each column
for column_to_normalize in stock_data_with_indicators.columns:
    min_value = stock_data_with_indicators[column_to_normalize].min()
    max_value = stock_data_with_indicators[column_to_normalize].max()

    stock_data_with_indicators[column_to_normalize] = (stock_data_with_indicators[column_to_normalize] - min_value) / (max_value - min_value)

print(stock_data_with_indicators.columns)
print(stock_data_with_indicators.head(20))

# shift back 1
stock_data_with_indicators.iloc[:, :] = stock_data_with_indicators.iloc[:, :].shift(1)


"""Output the processed data for later use in R"""
concat_df = pd.concat([df, stock_data_with_indicators], axis=1)
print(concat_df.shape)
print(concat_df.head())
concat_df.to_csv("processed_data.csv")


