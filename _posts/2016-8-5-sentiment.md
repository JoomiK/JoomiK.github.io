---
layout: post
title: Sentiment analysis
--- 

Here I used the Twitter API to collect tweets about Trump and Clinton, explored word co-occurrences, and performed sentiment analysis. 

Note: I did not have labeled data here, so I used short movie reviews to train my model. This is not expected to lead to accurate predictions of sentiment in tweets (especially of political nature), since this wouold not capture things like sarcasm (of which there is not shortage on Twitter). However, I did it anyway, to try it out, and have a workflow ready for when I do have the opportunity to work with labeled data.

[Part I](https://github.com/JoomiK/Trump_Clinton_Tweets/blob/master/Trump_Clinton_tweets.ipynb)
Data cleanup/pre-processing and word counts.

[Part II](https://github.com/JoomiK/Trump_Clinton_Tweets/blob/master/Trump_Clinton_Tweets_2.ipynb)
Classification/sentiment analysis of tweets.


