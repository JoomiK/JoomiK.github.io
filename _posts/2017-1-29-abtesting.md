---
layout: post
title: AB testing teaching methods with PYMC3  
---

Data: Student test scores  
Techniques: Bayesian estimation, MCMC

Code:  
[AB Testing Teaching Methods](https://github.com/JoomiK/AB-testing-teaching-methods/blob/master/AB_Testing_teaching_methods.ipynb)  

---


### Summary  

I participated in a study that looked at whether the order of presenting materials in a high school biology class made a difference in test scores.

Specifically, students were split into two groups; in Group 1, Mendelian genetics was taught before any in-depth discussion of the molecular biology underpinning genetics. In Group 2, the molecular biology was taught before teaching Mendelian genetics. Some teachers have hypothesized that the second method would be better for students; we looked at the evidence with this study.

Here I look at exam score data for the two groups- this exam specifically focused on the conceptual understanding of genetics.

I use Bayesian methods to estimate how different the scores were between the two groups, and estimate the uncertainty associated with that difference.


```python
%matplotlib inline
import numpy as np
import pymc3 as pm
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
sns.set(color_codes=True)

np.random.seed(20090425)
```



### The data:


```python
scores = pd.read_excel('test_scores.xlsx')
```

In group1, students were taught the more "traditional" way; they were taught Mendelian genetics before molecular biology. In group2, the order was reversed.


```python
scores.head()
```


![png](/images/test_scores.png)


Both groups had 93 students, and the mean for group2 (81.8) is 2.8 points higher than the mean for group1 (79). We can also see that group2's min and standard deviation were lower than group1's min and standard deviation.

### How significant is the difference?


```python
scores.describe()
```


![png](/images/score_describe.png)


```python
y1 = np.array(scores['group1'])
y2 = np.array(scores['group2'])

y = pd.DataFrame(dict(value=np.r_[y1, y2], group=np.r_[['group1']*len(y1), ['group2']*len(y2)]))

y.hist('value', by='group');
```


![png](/images/output_10_0.png)


I'll use a t-distribution (this is less sensitive to outliers compared to the normal distribution) to describe the distribution of scores for each group, with each having its own mean and standard deviation parameter. I'll use the same ν (the degrees of freedom parameter) for the two groups- so here we are making an assumption that the degree of normality is roughly the same for the two groups.

So the description of the data uses five parameters: the means of the two groups, the standard deviations of the two groups, and ν.

I'll apply broad normal priors for the means. The hyperparameters are arbitrarily set to the pooled empirical mean of the data and 2 times the pooled empirical standard deviation; this just applies very "diffuse" information to these quantities.

### Sampling from the posterior:


```python
μ_m = y.value.mean()
μ_s = y.value.std() * 2

with pm.Model() as model:
    """
    The priors for each group.
    """
    group1_mean = pm.Normal('group1_mean', μ_m, sd=μ_s)
    group2_mean = pm.Normal('group2_mean', μ_m, sd=μ_s)
```

I'll give a uniform(1,20) prior for the standard deviations.


```python
σ_low = 1
σ_high = 20

with model:
    group1_std = pm.Uniform('group1_std', lower=σ_low, upper=σ_high)
    group2_std = pm.Uniform('group2_std', lower=σ_low, upper=σ_high)
```

    Applied interval-transform to group1_std and added transformed group1_std_interval_ to model.
    Applied interval-transform to group2_std and added transformed group2_std_interval_ to model.


For the prior for ν, an exponential distribution with mean 30 was selected because it balances near-normal distributions (where ν > 30) with more thick-tailed distributions (ν < 30). In other words, this spreads credibility fairly evenly over nearly normal or heavy tailed data. Other distributions that could have been used were various uniform distributions, gamma distributions, etc.


```python
with model:
    """
    Prior for ν is an exponential (lambda=29) shifted +1.
    """
    ν = pm.Exponential('ν_min_one', 1/29.) + 1

sns.distplot(np.random.exponential(30, size=10000), kde=False);
```

    Applied log-transform to ν_min_one and added transformed ν_min_one_log_ to model.



![png](/images/output_18_1.png)



```python
with model:
    """
    Transforming standard deviations to precisions (1/variance) before
    specifying likelihoods.
    """
    λ1 = group1_std**-2
    λ2 = group2_std**-2

    group1 = pm.StudentT('group1', nu=ν, mu=group1_mean, lam=λ1, observed=y1)
    group2 = pm.StudentT('group2', nu=ν, mu=group2_mean, lam=λ2, observed=y2)
```

Now we'll look at the difference between group means and group standard deviations.


```python
with model:
    """
    The effect size is the difference in means/pooled estimates of the standard deviation.
    The Deterministic class represents variables whose values are completely determined
    by the values of their parents.
    """
    diff_of_means = pm.Deterministic('difference of means',  group2_mean - group1_mean)
    diff_of_stds = pm.Deterministic('difference of stds',  group2_std - group1_std)
    effect_size = pm.Deterministic('effect size',
                                   diff_of_means / np.sqrt((group2_std**2 + group1_std**2) / 2))


```


```python
with model:
    trace = pm.sample(2000, njobs=2)
```

### Summarize the posterior distributions of the parameters:


```python
pm.plot_posterior(trace[1000:],
                  varnames=['group1_mean', 'group2_mean', 'group1_std', 'group2_std', 'ν_min_one'],
                  color='#87ceeb');
```


![png](/images/output_24_0.png)


Let's look at the group differences (group2_mean = group1_mean), setting ref_val=0, which displays the percentage below and above zero. For the difference in means, 1.9% of the posterior probability is less than zero, while 98.1% is greater than zero. So there is a very small chance that the mean for group1 is larger or equal to the mean for group2, but there a much larger chance that group2's mean is larger than group1's.

It also looks like the variability in scores for group2 was somewhat lower than for group1- perhaps switching the order that genetics was taught not only increased scores, but brought some of the outlier students (particularly the ones that would have scored most poorly) closer to the mean?


```python
pm.plot_posterior(trace[1000:],
                  varnames=['difference of means', 'difference of stds', 'effect size'],
                  ref_val=0,
                  color='#87ceeb');
```


![png](/images/output_26_0.png)


For comparison we can do a t-test, which in this case is consistent with our Bayesian analysis.


```python
ttest_ind(y1,y2, equal_var=False)
```




    Ttest_indResult(statistic=-2.0683211848534517, pvalue=0.040018677966763901)

