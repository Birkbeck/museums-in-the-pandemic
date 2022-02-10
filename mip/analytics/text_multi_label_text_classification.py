#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import re
get_ipython().run_cell_magic('python', '-m nltk.downloader stopwords', '')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
stop_words = set(stopwords.words('english'))


# In[6]:


df = pd.read_csv(
    "data/ToxicCommentClassificationChallenge/train.csv", encoding="ISO-8859-1")


# In[34]:


df.sample(10)


# ### Number of comments in each category

# In[7]:


df_toxic = df.drop(['id', 'comment_text'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats


# In[8]:


df_stats.plot(x='category', y='number_of_comments', kind='bar',
              legend=False, grid=True, figsize=(8, 5))
plt.title("Number of comments per category")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('category', fontsize=12)


# ### Multi-Label
#
# How many comments have multiple labels?

# In[9]:


rowsums = df.iloc[:, 2:].sum(axis=1)
x = rowsums.value_counts()

# plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(x.index, x.values)
plt.title("Multiple categories per comment")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('# of categories', fontsize=12)


# Vast majority of the comment texts are not labeled.

# The distribution of the number of words in comment texts.

# In[10]:


lens = df.comment_text.str.len()
lens.hist(bins=np.arange(0, 5000, 50))


# Most of the comment text length are within 500 characters, with some outliers up to 5,000 characters long.

# In[11]:


print('Percentage of comments that are not labelled:')
print(len(df[(df['toxic'] == 0) & (df['severe_toxic'] == 0) & (df['obscene'] == 0) & (
    df['threat'] == 0) & (df['insult'] == 0) & (df['identity_hate'] == 0)]) / len(df))


# There is no missing comment in comment text column.

# In[12]:


print('Number of missing comments in comment text:')
df['comment_text'].isnull().sum()


# Have a peek the first comment, the text needs clean.

# In[13]:


df['comment_text'][0]


# In[14]:


categories = ['toxic', 'severe_toxic', 'obscene',
              'threat', 'insult', 'identity_hate']


# ### Create a function to clean the text

# In[15]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text


# ### Clean up comment_text column

# In[16]:


df['comment_text'] = df['comment_text'].map(lambda com: clean_text(com))


# Much better!

# In[97]:


df['comment_text'][0]


# ### Split to train and test sets

# In[17]:


train, test = train_test_split(
    df, random_state=42, test_size=0.33, shuffle=True)


# In[18]:


X_train = train.comment_text
X_test = test.comment_text
print(X_train.shape)
print(X_test.shape)


# ### Pipeline
#
# scikit-learn provides a Pipeline utility to help automate machine learning workflows. Pipelines are very common in Machine Learning systems, since there is a lot of data to manipulate and many data transformations to apply. So we will utilize pipeline to train every classifier.

# ### OneVsRest multilabel strategy
#
# The Multi-label algorithm accepts a binary mask over multiple labels. The result for each prediction will be an array of 0s and 1s marking which class labels apply to each row input sample.

# ### Naive Bayes
#
# OneVsRest strategy can be used for multilabel learning, where a classifier is used to predict multiple labels for instance. Naive Bayes supports multi-class, but we are in a multi-label scenario, therefore, we wrapp Naive Bayes in the OneVsRestClassifier.

# In[19]:


# Define a pipeline combining a text feature extractor with multi lable classifier
NB_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(MultinomialNB(
        fit_prior=True, class_prior=None))),
])


# In[20]:


for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    NB_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = NB_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(
        accuracy_score(test[category], prediction)))


# ### LinearSVC

# In[21]:


SVC_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
])


# In[22]:


for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(
        accuracy_score(test[category], prediction)))


# ### Logistic Regression

# In[64]:


LogReg_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
])
for category in categories:
    print('... Processing {}'.format(category))
    # train the model using X_dtm & y
    LogReg_pipeline.fit(X_train, train[category])
    # compute the testing accuracy
    prediction = LogReg_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(
        accuracy_score(test[category], prediction)))
