**Twitter-Sentiment-Analysis**

It is a Natural Language Processing Problem where Sentiment Analysis is done by Classifying the Positive tweets from negative tweets by machine learning models for classification, text mining, text analysis, data analysis and data visualization

**Introduction**

Natural Language Processing (NLP) is a hotbed of research in data science these days and one of the most common applications of NLP is sentiment analysis. From opinion polls to creating entire marketing strategies, this domain has completely reshaped the way businesses work, which is why this is an area every data scientist must be familiar with.
Thousands of text documents can be processed for sentiment (and other features including named entities, topics, themes, etc.) in seconds, compared to the hours it would take a team of people to manually complete the same task.
Microblogging sites like Facebook, Twitter are now becoming more popular among the people as it provides a direct platform for the users to flatus their views on any topic. These Microblogs are widely used by users to show their emotions, sentiment for any events like Natural Disasters, Earthquake, Sports etc. and the places they visit and the food they eat. Twitter is one of those microblogging sites and widely used platform for emotions manifestation & flooding the views to the intended community.
We will do so by following a sequence of steps needed to solve a general sentiment analysis problem. We will start with preprocessing and cleaning of the raw text of the tweets. Then we will explore the cleaned text and try to get some intuition about the context of the tweets. After that, we will extract numerical features from the data and finally use these feature sets to train models and identify the sentiments of the tweets.

This is one of the most interesting challenges in NLP so I’m very excited to take this journey with you!

The aim of this project is to create an algorithm that can accurately classify Twitter messages as positive or negative, in relation to a query term. Our 
hypothesis is that to have high accuracy in separating emotions in Twitter messages using machine learning methods. We propose an approach that analyses feeling of 
cricket fans and correlate sentiment to match play. We use data collected on twitter in the message form. To predict the outcome of a cricket match we are not going to rely on a single machine learning algorithm we are using at least two machine learning algorithms to compare the accuracy. We have applied modern classification techniques –Logistics Regression and Random Forest, and conducted a comparative study based on the overall cricket tweets. The project outcome is given in form of webpage giving both analysis and prediction of live tweets using Logistic Regression and older tweets using Random Forest. 

Also, to add on I have applied few more Machine Learning Algorithms to compare the accuracy, I ran the Supervised Linear Model - Support Vector Machine and Supervised Decision Tree Classifier to compare the accuracy among the other Machine Learning algorithms ran by the other. 

Formally, given a training sample of tweets and labels, where label ‘1’ denotes the tweet is racist/sexist and label ‘0’ denotes the tweet is not racist/sexist, your objective is to predict the labels on the given test dataset.

**Methodology**

A. Logistic Regression 

B. Random Forest Algorithm

C. DecisionTree Classifier Model

D. Support Vector Machine

 Performance evaluation used for checking the Classification result as Precision, recall and F-measure. 
True positive (TP)is correctly predicted positive values which mean that value of actual class is yes and predicated class is also yes. True Negative (TN) is correctly predicted negative values which means that the actual class is no and value of predicated also no. False Positive (FP) is actual class is no and predicated class is no. False Negative (FN) is actual class is yes and predicated class is no. 

**Note** 
Precision is the number of true positive review out of total number positively assigned review.  
Recall is the number of true positive out of the actual positive review
The evaluation metric from this practice problem is F1-Score.

**Tweets Preprocessing and Cleaning**

You are searching for a document in this office space. In which scenario are you more likely to find the document easily? Of course, in the less cluttered one because each item is kept in its proper place. The data cleaning exercise is quite similar. If the data is arranged in a structured format then it becomes easier to find the right information.

The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text.

In one of the later stages, we will be extracting numeric features from our Twitter text data. This feature space is created using all the unique words present in the entire data. So, if we preprocess our data well, then we would be able to get a better quality feature space.

Let’s first read our data and load the necessary libraries.

**Story Generation and Visualization from Tweets**
So, we have total 160000 tweets in which 80000 is positive and 80000 is negative tweets. We are using 70% data as a Training data and 30% data as a Testing data, on which we have performed Logistic Regression and Random Forest.

In this section, we will explore the cleaned tweets text. Exploring and visualizing data, no matter whether its text or any other data, is an essential step in gaining insights. Before we begin exploration, we must think and ask questions related to the data in hand. A few probable questions are as follows:

What are the most common words in the entire dataset? What are the most common words in the dataset for negative and positive tweets, respectively? How many hashtags are there in a tweet? Which trends are associated with my dataset? Which trends are associated with either of the sentiments? Are they compatible with the sentiments?

**End Notes**
In this article, we learned how to approach a sentiment analysis problem. We started with preprocessing and exploration of data. Then we extracted features from the cleaned text using Bag-of-Words and TF-IDF. Finally, we were able to build a couple of models using both the feature sets to classify the tweets. 
After extracting features, we perform machine learning algorithms on it. Classification is conducted using Logistic Regression, Random Forest, Decision Tree & Support Vector Machine. Results shows precision, recall, f1-score, accuracy of positive and negative tweets of each classification.


