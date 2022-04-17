#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

import numpy as np  
import pandas as pd  


# In[2]:


columns_name = ["target", "ids", "date", "flag", "user", "text"]
df = pd.read_csv("trainingdataset.csv", encoding = 'ISO-8859-1', names = columns_name)
df.dropna()
df.head()


# In[3]:


user_tweets=list(df['text'])
df['target']=df['target'].map({0:0,4:1})
df['target'].value_counts()


# In[4]:


labels=df['target'].values


# In[5]:


import string
import re


# In[6]:


clean_tweets= [i.lower() for i in user_tweets]
clean_tweets= [re.sub('RT @\w+:'," ", i) for i in clean_tweets]
clean_tweets= [re.sub('@(\w+)', " ", i) for i in clean_tweets]
clean_tweets= [re.sub('\d', " ", i) for i in clean_tweets]
clean_tweets= [re.sub('http\s+', " ", i) for i in clean_tweets]
clean_tweets= [i.translate(str.maketrans('', '', string.punctuation)) for i in clean_tweets]
clean_tweets= [re.sub('\s+', " ", i) for i in clean_tweets]
clean_tweets= [re.sub('@', " ", i) for i in clean_tweets]
clean_tweets= [re.sub(':', " ", i) for i in clean_tweets]


# In[7]:


df.head()


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
class_vector = vectorizer.fit_transform(clean_tweets)


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(class_vector,labels,test_size=0.3)


# In[10]:


#X_train.shape


# In[11]:


#y_train.shape


# In[12]:


#y_test.shape


# In[13]:


#X_test.shape


# In[14]:


#df.info()


# In[15]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
class_vector = vectorizer.fit_transform(clean_tweets)


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(class_vector,labels,test_size=0.3)


# In[18]:


df[df['target'] == 0].head(10)


# In[19]:


df[df['target'] == 1].head(10)


# In[20]:


#df['target'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))


# In[21]:


#df['len'] = df['text'].str.len()
#df.head(10)


# In[22]:


#df.groupby('target').describe()


# In[23]:


#from sklearn.feature_extraction.text import CountVectorizer


#cv = CountVectorizer(stop_words = 'english')
#words = cv.fit_transform(df.text)

#sum_words = words.sum(axis=0)

#words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
#words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

#frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

#frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'blue')
#plt.title("Most Frequently Occuring Words - Top 30")


# In[24]:


#from wordcloud import WordCloud

#wordcloud = WordCloud(background_color = 'white', width = 1000, height = 1000).generate_from_frequencies(dict(words_freq))

#plt.figure(figsize=(10,8))
#plt.imshow(wordcloud)
#plt.title("WordCloud - Vocabulary from Reviews", fontsize = 22)


# In[25]:


#negative_words =' '.join([text for text in df['text'][df['target'] == 0]])

#wordcloud = WordCloud(background_color = 'cyan', width=800, height=500, random_state = 0, max_font_size = 110).generate(negative_words)
#plt.figure(figsize=(10, 7))
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis('off')
#plt.title('The Negative Words')
#plt.show()


# In[26]:


#def hashtag_extract(x):
 #   hashtags = []
    
  #  for i in x:
   #     ht = re.findall(r"#(\w+)", i)
    #    hashtags.append(ht)
#
 #   return hashtags


# In[27]:


#HT_regular = hashtag_extract(df['text'][df['target'] == 0])

# extracting hashtags from racist/sexist tweets
#HT_negative = hashtag_extract(df['text'][df['target'] == 1])

# unnesting list
#HT_regular = sum(HT_regular,[])
#HT_negative = sum(HT_negative,[])


# In[28]:


import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[30]:


#a = nltk.FreqDist(HT_regular)
#d = pd.DataFrame({'Hashtag': list(a.keys()),
                 # 'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
#d = d.nlargest(columns="Count", n = 10) 
#plt.figure(figsize=(16,5))
#ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()


# In[31]:


#a = nltk.FreqDist(HT_negative)
#d = pd.DataFrame({'Hashtag': list(a.keys()),
 #                 'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
#d = d.nlargest(columns="Count", n = 10) 
#plt.figure(figsize=(16,5))
#ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
#ax.set(ylabel = 'Count')
#plt.show()


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 5000)
class_vector = cv.fit_transform(clean_tweets)

print(X_test.shape)
print(y_test.shape)


# In[33]:


# splitting the training data into train and valid sets
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(class_vector,labels,test_size=0.3)

print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)


# In[34]:



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)


# In[35]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# In[36]:


clf.fit(X_train, y_train)


# In[37]:


y_pred = clf.predict(X_valid)


# In[47]:


print("Training Accuracy :", clf.score(X_train, y_train))
print("Validation Accuracy :", clf.score(X_valid, y_valid))

# calculating the f1 score for the validation set
print("F1 score :", f1_score(y_valid, y_pred))


# In[48]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))


# In[49]:


cm = confusion_matrix(y_valid, y_pred)
print(cm)


# # DecisionTree Classifier

# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


dtc = DecisionTreeClassifier(max_depth=2, random_state=0)


# In[52]:


dtc.fit(X_train, y_train)


# In[53]:


y_pred = dtc.predict(X_valid)


# In[54]:


print("Training Accuracy :", dtc.score(X_train, y_train))
print("Validation Accuracy :", dtc.score(X_valid, y_valid))

# calculating the f1 score for the validation set
print("f1 score :", f1_score(y_valid, y_pred))


# In[55]:


from sklearn.metrics import classification_report
print(classification_report(y_pred,y_test))


# In[56]:


cm = confusion_matrix(y_valid, y_pred)
print(cm)


# # Support Vector Machine

# In[57]:


from sklearn.svm import SVC


# In[58]:


model = SVC(kernel='rbf', random_state = 1)


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_valid)
