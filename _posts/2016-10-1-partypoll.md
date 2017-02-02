---
layout: post
title: Predicting political leanings from text
---

Data: Twitter API  
Techniques: NLP, logistic regression, latent Dirichlet allocation, scraping

The web application can be found here: [PartyPoll](http://partypoll.co/)  
Try it out!

---

### Rationale:
Can we predict which way a person leans, politically, from what they say on social media? Even if they are not talking about politics?

### Approach:
Data Source:  
I collected tweets using the TwitterAPI, from the followers of two "third party" political parties in the U.S., the Green Party, and the Libertarian Party, and used these as proxies for Green and Libertarian sympathizers. 

Preprocessing:  
1. I collected the last 200 tweets for each follower (this was the maximum allowable I could collect using the API).  
2. I removed stop words, links, as well as "political talk." Why did I do that? Well, I wanted to see if we could predict which way someone leaned even if they didn't give out overt cues about their political inclinations. In other words, do the things that Green and Libertarian party followers talk about, differ enough so that we can distinguish between them (do they care about different topics)?  
3. I TF-IDF vectorized the tweets.  
4. I used these to train a logistic regression classifier that would predict political leaning based on text data. It achieved about 84% accuracy in predicting political party.

### Why does this work at all?
Why is it that we can distinguish between Green and Libertarian party followers?  

I used Latent Dirichlet Allocation to explore topics of third party followers. Turns out, they talk about different things (though there is a lot of overlap):

[Explore Topics](http://partypoll.co/topics)
