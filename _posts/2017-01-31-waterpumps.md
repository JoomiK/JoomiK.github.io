---
layout: post
title: Predicting non-functional water pumps in Tanzania
---
Data: From a competition for [drivendata.org](https://www.drivendata.org/competitions/7/)  
Techniques: Classification, random forest, imputation, PCA   

[Part I- EDA and cleanup](https://github.com/JoomiK/PredictingWaterPumps/blob/master/WaterPumps.ipynb)    
[Part II- Modeling](https://github.com/JoomiK/PredictingWaterPumps/blob/master/WaterPumpsII.ipynb)  


---

## Part I- EDA and cleanup  
Using data on water pumps in communities across Tanzania, can we predict the pumps that are functional, need repairs, or don't work at all?  

There are 39 features in total. They are described [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/#features_list).

Among the features are:  

* Construction year  
* Total static head (amount water available to waterpoint)  
* Funder of the well  
* Installer of the well  
* Location 


```python
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
import sys

import sklearn as sk
from sklearn import metrics
from sklearn.model_selection import train_test_split, ShuffleSplit, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from IPython.core.display import display, HTML
import warnings

rcParams['figure.figsize'] = 12, 4
print('Python version: %s.%s.%s' % sys.version_info[:3])
print('numpy version:', np.__version__)
print('pandas version:', pd.__version__)
print('scikit-learn version:', sk.__version__)
warnings.filterwarnings("ignore")
```

    Python version: 3.5.2
    numpy version: 1.12.0
    pandas version: 0.19.2
    scikit-learn version: 0.18.1



```python
def label_map(y):
    """
    Converts string labels to integers
    """
    if y=="functional":
        return 2
    elif y=="functional needs repair":
        return 1
    else:
        return 0
    

def isNan(num):
    """
    Function to test for Nan. Returns True for NaNs, False otherwise.
    """
    return num != num
```




```python
# Load the data
train = pd.read_csv('training.csv', index_col='id')
labels = pd.read_csv('train_labels.csv', index_col='id')
test = pd.read_csv('test_set.csv', index_col='id')
```

```python
print("Train data labels:",len(labels))
print("Train data rows, columns:",train.shape)
print("Test data rows, columns:", test.shape)
```

    Train data labels: 59400
    Train data rows, columns: (59400, 39)
    Test data rows, columns: (14850, 39)

### Visualizing water pumps:  
Apparently responsibility for water and sanitation service provision is decentralized, so local governments are responsible for water resource management. Luckily, we have information on which regions the water pumps are in. Perhaps this will be a good predictor.

![png](/images/WellMap.png)
[Interactive Map](https://joomik.carto.com/builder/3227f55e-d6ac-11e6-832f-0e3ebc282e83/embed)  

There is some "clumpiness" here; in the southeast you'll notice that there seems to be a higher proportion of non-functional pumps (red) than near Iringa, where you see a lot of green (functional).

### Exploring the data:  

    # Pairwise plots
    sns.set(style='whitegrid',context='notebook')
    cols=['amount_tsh','gps_height','num_private','population','status_group']
    sns.pairplot(merged[cols],size=2.5)
    plt.show()

![png](/images/waterpumps_pairwise.png)
The variables plotted here are:  
* amount_tsh: total static head (amount water available to waterpoint)  
* gps_height: altitude of well  
* num_private: (this feature is not described)  
* population: population around the well  

It looks like the water pumps with high static head tend to be functional (label 2). (It would also be worth looking into whether the really high tsh values are realistic.)

### Funders:

Apparently water resource management in Tanzania relies heavily on external donors (according to my rudimentary research). We have 1898 different funders in this dataset, with a lot of missing values. There are also many funders with too few observations to have any real impact on model fit (many funders that only occur once). 


```python
len(train.funder.value_counts(dropna=False))
```




    1898




```python
train.funder.value_counts(dropna=False)
```




    Government Of Tanzania            9084
    NaN                               3636
    Danida                            3114
    Hesawa                            2202
    Rwssp                             1374
    World Bank                        1349
    Kkkt                              1287
    World Vision                      1246
    Unicef                            1057
    Tasaf                              877
    District Council                   843
    Dhv                                829
    Private Individual                 826
    Dwsp                               811
    0                                  776
    Norad                              765
    Germany Republi                    610
    Tcrs                               602
    Ministry Of Water                  590
    Water                              583
    Dwe                                484
    Netherlands                        470
    Hifab                              450
    Adb                                448
    Lga                                442
    Amref                              425
    Fini Water                         393
    Oxfam                              359
    Wateraid                           333
    Rc Church                          321
                                      ... 
    Alia                                 1
    Kamata Project                       1
    Mbozi Hospital                       1
    Ballo                                1
    Natherland                           1
    Pankrasi                             1
    Mzee Mkungata                        1
    Cocu                                 1
    H4ccp                                1
    Redekop Digloria                     1
    Luke Samaras Ltd                     1
    John Gileth                          1
    Samwel                               1
    St Elizabeth Majengo                 1
    Serikaru                             1
    Municipal Council                    1
    Dae Yeol And Chae Lynn               1
    Rotary Club Kitchener                1
    Camartec                             1
    Chuo                                 1
    Okutu Village Community              1
    Kipo Potry                           1
    Dmd                                  1
    Maswi Drilling Co. Ltd               1
    Lee Kang Pyung's Family              1
    Grazie Grouppo Padre Fiorentin       1
    Rudep /dwe                           1
    Artisan                              1
    Afriican Reli                        1
    Makondakonde Water Population        1
    Name: funder, dtype: int64



In addition to the 3635 NaNs, there are 777 that have the entry "0".  
I'm assuming these are missing values too, so altogether we have 4412 missing values.


### Some other problems with this data:  
A summary of other things to deal with in this dataset:

- installer: has NaNs, zeros, and many low frequency levels  
- scheme_name: has NaNs and many low frequency levels  
- scheme_management has NaNs
- population: has zeros and many low frequency levels*  
- construction_year: has zeros  
- permit: has NaNs  
- public_meeting: has NaNs  


### Some options for dealing with this missing data:  
1. Drop the rows (I don't want to do this in this case because this would be a large proportion of the data).** 
2. Mean/median/mode imputation (crude, can severely distort the distribution of the variable).  
3. Predict (in order to do this, features need to be correlated or related somehow).
4. Create new feature like "no funder data." 


- For the missing categorical variables, I'm going to use option 4.  
- For the missing population, I'll leave the zeros as is, for the first pass at building a model, but at some point later on I might try to predict these using (for example) clustering or knn.  
- For the missing construction_year, I'll add a 'missing_construction_year' feature, and also run my model with and without imputation with the median. In a later iteration, maybe I'll try to impute using a fancier algorithmn.

### * Note on population: 
If I were really thorough I would try to fill in missing population data using some kind of census data. Also I looked into using the (previously dropped) subvillage feature here to predict missing population values, but it does not seem like a good idea, because it appears that the population they report here (presumably some small radius around the water pump) is not just the subvillage population. That is, when I looked at population values for subvillages, entries with the same subvillage often had wildly different populations.  


### ** Note on missing values: 
It's easy to imagine here that some of these NaNs may not be missing at random (we can imagine that year info is missing more from older wells, for example).


```python
# Prep- convert the zeros in 'funder','installer', and 'population' to NaNs. 
merged_clean = merged.replace({'funder':0, 'installer':0, 'population':0}, np.nan)
merged_clean = merged.replace({'funder':'0', 'installer':'0'}, np.nan)
```

### Dealing with low frequency/many levels:  
For these cases I will take low frequency levels (those that occur 20 times or less) and set to "other." The 20 here is totally arbitrary; in the interest of time I won't test out different thresholds or ways to bin, but ideally I would use cross validation to try out different methods and look at the effect on model performance. 


```python
# Lump low frequency levels in funder, installer, scheme_name into 'other'
exempt=['amount_tsh',  'gps_height',  'num_private',
       'basin', 'region', 'region_code', 'district_code', 'lga', 'population',
       'public_meeting', 'scheme_management',  'permit',
       'construction_year', 'extraction_type', 'extraction_type_group',
       'extraction_type_class', 'management', 'management_group', 'payment',
       'payment_type', 'water_quality', 'quality_group', 'quantity',
       'quantity_group', 'source', 'source_type', 'source_class',
       'waterpoint_type', 'waterpoint_type_group']

merged_clean = merged_clean.apply(lambda x: x.mask(x.map(x.value_counts())<20, 'other') if x.name not in exempt else x)
```

### Construction year:  

Replace zeros with NaNs. 
It might also be a good idea to bin the construction_year feature, but I will leave it as it is for this first pass.


```python
merged_clean = merged.replace({'construction_year':0}, np.nan)
merged_clean = merged.replace({'construction_year':'0'}, np.nan)
```


```python
# Note: this changed year values to floats
merged_clean.construction_year.value_counts(dropna=False)
```




    NaN        20709
     2010.0     2645
     2008.0     2613
     2009.0     2533
     2000.0     2091
     2007.0     1587
     2006.0     1471
     2003.0     1286
     2011.0     1256
     2004.0     1123
     2012.0     1084
     2002.0     1075
     1978.0     1037
     1995.0     1014
     2005.0     1011
     1999.0      979
     1998.0      966
     1990.0      954
     1985.0      945
     1980.0      811
     1996.0      811
     1984.0      779
     1982.0      744
     1994.0      738
     1972.0      708
     1974.0      676
     1997.0      644
     1992.0      640
     1993.0      608
     2001.0      540
     1988.0      521
     1983.0      488
     1975.0      437
     1986.0      434
     1976.0      414
     1970.0      411
     1991.0      324
     1989.0      316
     1987.0      302
     1981.0      238
     1977.0      202
     1979.0      192
     1973.0      184
     2013.0      176
     1971.0      145
     1960.0      102
     1967.0       88
     1963.0       85
     1968.0       77
     1969.0       59
     1964.0       40
     1962.0       30
     1961.0       21
     1965.0       19
     1966.0       17
    Name: construction_year, dtype: int64



#### Adding feature for missing construction year data:


```python
merged_clean['missing_construction_year'] = merged_clean['construction_year'].apply(lambda x: isNan(x))
```





### Summary of the number of missing values for each column:


```python
merged_clean.isnull().sum()
```




    amount_tsh                       0
    funder                        3636
    gps_height                       0
    installer                     3656
    num_private                      0
    basin                            0
    region                           0
    region_code                      0
    district_code                    0
    lga                              0
    population                       0
    public_meeting                3334
    scheme_management             3877
    scheme_name                  28166
    permit                        3056
    construction_year            20709
    extraction_type                  0
    extraction_type_group            0
    extraction_type_class            0
    management                       0
    management_group                 0
    payment                          0
    payment_type                     0
    water_quality                    0
    quality_group                    0
    quantity                         0
    quantity_group                   0
    source                           0
    source_type                      0
    source_class                     0
    waterpoint_type                  0
    waterpoint_type_group            0
    status_group                     0
    missing_construction_year        0
    dtype: int64



Interesting to note that the number of NaNs for both the funder and installer columns is close. If we compare just these two columns we see that the NaNs often appear in the same rows:


```python
pd.concat([merged_clean.funder, merged_clean.installer], axis=1)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>funder</th>
      <th>installer</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69572</th>
      <td>Roman</td>
      <td>Roman</td>
    </tr>
    <tr>
      <th>8776</th>
      <td>Grumeti</td>
      <td>GRUMETI</td>
    </tr>
    <tr>
      <th>34310</th>
      <td>Lottery Club</td>
      <td>World vision</td>
    </tr>
    <tr>
      <th>67743</th>
      <td>Unicef</td>
      <td>UNICEF</td>
    </tr>
    <tr>
      <th>19728</th>
      <td>Action In A</td>
      <td>Artisan</td>
    </tr>
    <tr>
      <th>9944</th>
      <td>Mkinga Distric Coun</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>19816</th>
      <td>Dwsp</td>
      <td>DWSP</td>
    </tr>
    <tr>
      <th>54551</th>
      <td>Rwssp</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>53934</th>
      <td>Wateraid</td>
      <td>Water Aid</td>
    </tr>
    <tr>
      <th>46144</th>
      <td>Isingiro Ho</td>
      <td>Artisan</td>
    </tr>
    <tr>
      <th>49056</th>
      <td>Private</td>
      <td>Private</td>
    </tr>
    <tr>
      <th>50409</th>
      <td>Danida</td>
      <td>DANIDA</td>
    </tr>
    <tr>
      <th>36957</th>
      <td>World Vision</td>
      <td>World vision</td>
    </tr>
    <tr>
      <th>50495</th>
      <td>Lawatefuka Water Supply</td>
      <td>Lawatefuka water sup</td>
    </tr>
    <tr>
      <th>53752</th>
      <td>Biore</td>
      <td>WEDECO</td>
    </tr>
    <tr>
      <th>61848</th>
      <td>Rudep</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>48451</th>
      <td>Unicef</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>58155</th>
      <td>Unicef</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>34169</th>
      <td>Hesawa</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>18274</th>
      <td>Danida</td>
      <td>Danid</td>
    </tr>
    <tr>
      <th>48375</th>
      <td>Twe</td>
      <td>TWE</td>
    </tr>
    <tr>
      <th>6091</th>
      <td>Dwsp</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>58500</th>
      <td>Unicef</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>37862</th>
      <td>Isf</td>
      <td>ISF</td>
    </tr>
    <tr>
      <th>51058</th>
      <td>African Development Bank</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>22308</th>
      <td>Government Of Tanzania</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>55012</th>
      <td>Sobodo</td>
      <td>Kilolo Star</td>
    </tr>
    <tr>
      <th>20145</th>
      <td>Hesawa</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>19685</th>
      <td>Government Of Tanzania</td>
      <td>District council</td>
    </tr>
    <tr>
      <th>69124</th>
      <td>Lawatefuka Water Supply</td>
      <td>Lawatefuka water sup</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>14796</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20387</th>
      <td>Netherlands</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>29940</th>
      <td>Tlc</td>
      <td>TLC</td>
    </tr>
    <tr>
      <th>15233</th>
      <td>Rudep</td>
      <td>Distri</td>
    </tr>
    <tr>
      <th>49651</th>
      <td>Rwssp</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>50998</th>
      <td>Government Of Tanzania</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>34716</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>43986</th>
      <td>Government Of Tanzania</td>
      <td>Government</td>
    </tr>
    <tr>
      <th>38067</th>
      <td>Rc</td>
      <td>ACRA</td>
    </tr>
    <tr>
      <th>58255</th>
      <td>Do</td>
      <td>DO</td>
    </tr>
    <tr>
      <th>30647</th>
      <td>Roman</td>
      <td>Roman</td>
    </tr>
    <tr>
      <th>67885</th>
      <td>Mkinga Distric Coun</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>47002</th>
      <td>Ces(gmbh)</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>44616</th>
      <td>World Bank</td>
      <td>World bank</td>
    </tr>
    <tr>
      <th>72148</th>
      <td>Concern</td>
      <td>CONCERN</td>
    </tr>
    <tr>
      <th>34473</th>
      <td>Jaica</td>
      <td>JAICA CO</td>
    </tr>
    <tr>
      <th>34952</th>
      <td>Adb</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>26640</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>72559</th>
      <td>Kidep</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>30410</th>
      <td>Co</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>13677</th>
      <td>Rudep</td>
      <td>DWE</td>
    </tr>
    <tr>
      <th>44885</th>
      <td>Government Of Tanzania</td>
      <td>Government</td>
    </tr>
    <tr>
      <th>40607</th>
      <td>Government Of Tanzania</td>
      <td>Government</td>
    </tr>
    <tr>
      <th>48348</th>
      <td>Private</td>
      <td>Private</td>
    </tr>
    <tr>
      <th>11164</th>
      <td>World Bank</td>
      <td>ML appro</td>
    </tr>
    <tr>
      <th>60739</th>
      <td>Germany Republi</td>
      <td>CES</td>
    </tr>
    <tr>
      <th>27263</th>
      <td>Cefa-njombe</td>
      <td>Cefa</td>
    </tr>
    <tr>
      <th>37057</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31282</th>
      <td>Malec</td>
      <td>Musa</td>
    </tr>
    <tr>
      <th>26348</th>
      <td>World Bank</td>
      <td>World</td>
    </tr>
  </tbody>
</table>
<p>59400 rows × 2 columns</p>
</div>



Something to notice here: the funder and installer is often the same entitiy.  

When they are not the same, it's usually because 'DWE' appears as the installer. It seems like 'DWE' could be some code for something (like "no information"), instead of an actual installer. Anyway, I will leave these alone in the absence of more information on what the acronyms mean.  

Also there are some cases where the same entitiy was probably entered differently- for example "Danida" and "Danid" or "Jaica" and "JAICA CO" or "Government of Tanzania" and "Government" or "Wateraid" and "Water Aid" I won't try to fix these for now, but this is something one would do ideally.

### Dealing with categorical variables:
Convert categorical variables into dummy/indicator variables. At the same time, we'll be adding columns to indicate NaNs. 


```python
merged_clean_dum = pd.get_dummies(merged_clean, dummy_na=True)
```


```python
merged_clean_dum.shape
```




    (59400, 7069)



I now have 7069 features.


```python
merged_clean_dum.to_csv('merged_clean_dum.csv')
```

### A note on other features of interest  for this project:
It's easy to imagine that we could improve our predictions with other sources of data here. Some that come to mind in this case are:  
- Presence of other utilities nearby  
- Rate of previous breakdowns  
- Distance to major road  
- Whether the water pump is in a hazardous or flooding area
- Crime rate in the near vicinity  


## Part II - Model selection and evaluation


I will try modeling two ways: (1) on one-hot encoded data, with and without dimensionality reduction, and (2) on the original data without one-hot encoding.


```python
# Load the one-hot encoded data
data = pd.read_csv('merged_clean_dum.csv', index_col='id')
```



```python
print("Data rows, columns:",data.shape)
```

    Data rows, columns: (59400, 7069)


Note: region_code and district_code are integers, but categorical. I could convert these to dummy variables too, but I will leave these as is for the first pass.


```python
# Getting just the training data by dropping the labels
train = data.drop('status_group',1)

# Getting labels
labels = data.status_group
```

### Model preprocessing and selection:


```python
X = train.as_matrix()
y = labels.tolist()

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
```

### I. GridSearchCV w/the one-hot encoded data:
Imputation, dimensionality reduction, and selection of hyperparameters for a random forest classifier (Note- I'm not testing hyperparameters exhaustively for now):


```python
imputer = Imputer(strategy='median')
pca = decomposition.PCA()
rf = RandomForestClassifier(class_weight='balanced', 
                            n_jobs=-1, 
                            n_estimators=300)

steps = [('imputer', imputer),
        ('pca', pca),
        ('random_forest', rf)]

pipeline = Pipeline(steps)
parameters = dict(
                pca__n_components=[40, 100, 300], 
                random_forest__max_features=['auto', 'log2']
                )

gs = GridSearchCV(pipeline, param_grid=parameters)
gs.fit(X_train, y_train)
```




    GridSearchCV(cv=None, error_score='raise',
           estimator=Pipeline(steps=[('imputer', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)), ('pca', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)), ('random_forest', RandomForestClassifier(bootstrap=True, class_weight=..._jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False))]),
           fit_params={}, iid=True, n_jobs=1,
           param_grid={'random_forest__max_features': ['auto', 'log2'], 'pca__n_components': [40, 100, 300]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=0)




```python
print(gs.best_score_)
print(gs.best_params_)
```

    0.773400673401
    {'random_forest__max_features': 'log2', 'pca__n_components': 100}


### Classification report:
The model has an f1-score of about 0.8 for labels 0 and 2 (non-functional and functional), but does poorly on label 1- functional but needs repairs.


```python
rf = gs.best_estimator_
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

                 precision    recall  f1-score   support
    
              0       0.82      0.76      0.79      6875
              1       0.41      0.39      0.40      1333
              2       0.80      0.85      0.82      9612
    
    avg / total       0.78      0.78      0.78     17820
    


### Steps:
1. Median imputation  
2. PCA with n_components=100  
3. Random Forest w/max_features=log2  
(Note: I could refine hyperparameters further but I'll move ahead for now). 


```python
rf.steps
```




    [('imputer',
      Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)),
     ('pca',
      PCA(copy=True, iterated_power='auto', n_components=100, random_state=None,
        svd_solver='auto', tol=0.0, whiten=False)),
     ('random_forest',
      RandomForestClassifier(bootstrap=True, class_weight='balanced',
                  criterion='gini', max_depth=None, max_features='log2',
                  max_leaf_nodes=None, min_impurity_split=1e-07,
                  min_samples_leaf=1, min_samples_split=2,
                  min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
                  oob_score=False, random_state=None, verbose=0,
                  warm_start=False))]




```python
rf.steps[2][1]
```




    RandomForestClassifier(bootstrap=True, class_weight='balanced',
                criterion='gini', max_depth=None, max_features='log2',
                max_leaf_nodes=None, min_impurity_split=1e-07,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=-1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
importances = rf.steps[2][1].feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.steps[2][1].estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
```


```python
feature_names = np.array(list(train.columns.values))
feature_names
```




    array(['amount_tsh', 'gps_height', 'num_private', ...,
           'waterpoint_type_group_improved spring',
           'waterpoint_type_group_other', 'waterpoint_type_group_nan'], 
          dtype='<U58')




```python
important_names = feature_names[importances>np.mean(importances)]
print(important_names)
```

    ['amount_tsh' 'gps_height' 'region_code' 'district_code' 'population'
     'construction_year' 'missing_construction_year' 'funder_0'
     'funder_A/co Germany' 'funder_Aar' 'funder_Abas Ka' 'funder_Abasia'
     'funder_Abc-ihushi Development Cent' 'funder_Abd' 'funder_Abdala'
     'funder_Abood' 'funder_Abs' 'funder_Aco/germany' 'funder_Acord Ngo'
     'funder_Act' 'funder_Act Mara' 'funder_Africa Project Ev Germany'
     'funder_Africaone Ltd' 'funder_Aqua Blues Angels' 'funder_Area']


    /Users/jkim/anaconda/envs/py35/lib/python3.5/site-packages/ipykernel/__main__.py:1: VisibleDeprecationWarning: boolean index did not match indexed array along dimension 0; dimension is 7068 but corresponding boolean dimension is 100
      if __name__ == '__main__':


### Top features:

Some of the top features are: amount_tsh (total static head, or amount water available to waterpoint), gps_height (altitude of the well), region and district codes, population, construction year, missing construction year, and funder.

### Some caveats:  
This is not unique to using random forests for feature selection, but applies to most model based feature selection methods: if the dataset has correlated features, once one of them is used as a predictor, the importance of the others is significantly reduced. This is ok if we just want to use feature selection to avoid overfitting, but if we are using it to interpret the data, we have to be careful. 

### Problems with one-hot encoding:
When we one-hot encoded categorical variables, the resulting sparsity makes continuous variables assigned higher feature importance. Moreover, a single level of a categorical variable must meet a very high bar to be selected for splitting early in the tree building, which can degrade predictive performance. Lastly, by one-hot encoding, we created many binary variables, and they were all seen as independent (from the splitting algorithm's point of view). 

### Confusion matrix:


```python
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
confmat
```




    array([[5216,  217, 1442],
           [ 211,  523,  599],
           [ 911,  539, 8162]])




```python
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,
               s=confmat[i,j],
               va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()
```


![png](/images/waterpump_CM.png)


The model has the most trouble classifying water pumps in class 1 (pumps that need repair).

### II. H2O implementation of random forest without one-hot encoding:

The H2O random forest implementation lets you input categorical data without one-hot encoding. It also treats missing values differently than sklearn's (as a separate category).

Start up a local H2O cluster:


```python
h2o.init(max_mem_size = "2G", nthreads=-1)
```

    Checking whether there is an H2O instance running at http://localhost:54321..... not found.
    Attempting to start a local H2O server...
      Java Version: java version "1.8.0_40"; Java(TM) SE Runtime Environment (build 1.8.0_40-b27); Java HotSpot(TM) 64-Bit Server VM (build 25.40-b25, mixed mode)
      Starting server from /Users/jkim/anaconda/envs/py35/lib/python3.5/site-packages/h2o/backend/bin/h2o.jar
      Ice root: /var/folders/5d/ftxntrp16l5g04c049nr03pr0000gp/T/tmp_jb85xnp
      JVM stdout: /var/folders/5d/ftxntrp16l5g04c049nr03pr0000gp/T/tmp_jb85xnp/h2o_jkim_started_from_python.out
      JVM stderr: /var/folders/5d/ftxntrp16l5g04c049nr03pr0000gp/T/tmp_jb85xnp/h2o_jkim_started_from_python.err
      Server is running at http://127.0.0.1:54321
    Connecting to H2O server at http://127.0.0.1:54321... successful.



<div style="overflow:auto"><table style="width:50%"><tr><td>H2O cluster uptime:</td>
<td>02 secs</td></tr>
<tr><td>H2O cluster version:</td>
<td>3.10.0.10</td></tr>
<tr><td>H2O cluster version age:</td>
<td>2 months and 9 days </td></tr>
<tr><td>H2O cluster name:</td>
<td>H2O_from_python_jkim_cz0i75</td></tr>
<tr><td>H2O cluster total nodes:</td>
<td>1</td></tr>
<tr><td>H2O cluster free memory:</td>
<td>1.778 Gb</td></tr>
<tr><td>H2O cluster total cores:</td>
<td>8</td></tr>
<tr><td>H2O cluster allowed cores:</td>
<td>8</td></tr>
<tr><td>H2O cluster status:</td>
<td>accepting new members, healthy</td></tr>
<tr><td>H2O connection url:</td>
<td>http://127.0.0.1:54321</td></tr>
<tr><td>H2O connection proxy:</td>
<td>None</td></tr>
<tr><td>Python version:</td>
<td>3.5.2 final</td></tr></table></div>


### Import data:  
This is a cleaned up version of the data without one-hot encoding.


```python
data2 = h2o.import_file(os.path.realpath('merged_clean.csv'))
```

    Parse progress: |█████████████████████████████████████████████████████████| 100%


### Encode response variable:  
Since we want to train a classification mode, we must ensure that the response is coded as a factor.


```python
data2['status_group']=data2['status_group'].asfactor() # encode response as a factor
data2['status_group'].levels() # show the levels
```




    [['0', '1', '2']]



### Partition data:  


```python
# Partition data into 70%, 15%, 15% chunks
# Setting a seed will guarantee reproducibility
train, valid, test = data2.split_frame([0.7, 0.15], seed=1234)
```


```python
X = data2.col_names[1:-1] # All columns except first (id) and last (reponse variable) 
y = data2.col_names[-1] # Response variable
```

### Model:  

I'll quickly build a model for now and come back to tuning hyperparameters later. 


```python
rf_v1 = H2ORandomForestEstimator(
        model_id="rf_v1",
        balance_classes=True,
        ntrees=300,
        stopping_rounds=2,
        score_each_iteration=True,
        seed=1000000)
```


```python
rf_v1.train(X, y, training_frame=train, validation_frame=valid)
```

    drf Model Build progress: |███████████████████████████████████████████████| 100%



```python
rf_v1
```

    Model Details
    =============
    H2ORandomForestEstimator :  Distributed Random Forest
    Model Key:  rf_v1
    Model Summary: 



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>number_of_trees</b></td>
<td><b>number_of_internal_trees</b></td>
<td><b>model_size_in_bytes</b></td>
<td><b>min_depth</b></td>
<td><b>max_depth</b></td>
<td><b>mean_depth</b></td>
<td><b>min_leaves</b></td>
<td><b>max_leaves</b></td>
<td><b>mean_leaves</b></td></tr>
<tr><td></td>
<td>32.0</td>
<td>96.0</td>
<td>3629520.0</td>
<td>20.0</td>
<td>20.0</td>
<td>20.0</td>
<td>1433.0</td>
<td>3759.0</td>
<td>2650.0938</td></tr></table></div>


    
    
    ModelMetricsMultinomial: drf
    ** Reported on train data. **
    
    MSE: 0.24663297395089073
    RMSE: 0.496621560094697
    LogLoss: 0.7351344533214206
    Mean Per-Class Error: 0.34878276470320757
    Confusion Matrix: vertical: actual; across: predicted
    



<div style="overflow:auto"><table style="width:50%"><tr><td><b>0</b></td>
<td><b>1</b></td>
<td><b>2</b></td>
<td><b>Error</b></td>
<td><b>Rate</b></td></tr>
<tr><td>17622.0</td>
<td>91.0</td>
<td>4808.0</td>
<td>0.2175303</td>
<td>4,899 / 22,521</td></tr>
<tr><td>3197.0</td>
<td>5680.0</td>
<td>13685.0</td>
<td>0.7482493</td>
<td>16,882 / 22,562</td></tr>
<tr><td>1683.0</td>
<td>136.0</td>
<td>20758.0</td>
<td>0.0805687</td>
<td>1,819 / 22,577</td></tr>
<tr><td>22502.0</td>
<td>5907.0</td>
<td>39251.0</td>
<td>0.3488028</td>
<td>23,600 / 67,660</td></tr></table></div>


    Top-3 Hit Ratios: 



<div style="overflow:auto"><table style="width:50%"><tr><td><b>k</b></td>
<td><b>hit_ratio</b></td></tr>
<tr><td>1</td>
<td>0.6511971</td></tr>
<tr><td>2</td>
<td>0.8973396</td></tr>
<tr><td>3</td>
<td>0.9999999</td></tr></table></div>


    
    ModelMetricsMultinomial: drf
    ** Reported on validation data. **
    
    MSE: 0.1594342154717807
    RMSE: 0.399292143012833
    LogLoss: 0.54388162010831
    Mean Per-Class Error: 0.39999447107732106
    Confusion Matrix: vertical: actual; across: predicted
    



<div style="overflow:auto"><table style="width:50%"><tr><td><b>0</b></td>
<td><b>1</b></td>
<td><b>2</b></td>
<td><b>Error</b></td>
<td><b>Rate</b></td></tr>
<tr><td>2494.0</td>
<td>18.0</td>
<td>897.0</td>
<td>0.2684072</td>
<td>915 / 3,409</td></tr>
<tr><td>92.0</td>
<td>87.0</td>
<td>446.0</td>
<td>0.8608</td>
<td>538 / 625</td></tr>
<tr><td>325.0</td>
<td>16.0</td>
<td>4477.0</td>
<td>0.0707763</td>
<td>341 / 4,818</td></tr>
<tr><td>2911.0</td>
<td>121.0</td>
<td>5820.0</td>
<td>0.2026661</td>
<td>1,794 / 8,852</td></tr></table></div>


    Top-3 Hit Ratios: 



<div style="overflow:auto"><table style="width:50%"><tr><td><b>k</b></td>
<td><b>hit_ratio</b></td></tr>
<tr><td>1</td>
<td>0.7973340</td></tr>
<tr><td>2</td>
<td>0.9536828</td></tr>
<tr><td>3</td>
<td>1.0</td></tr></table></div>


    Scoring History: 



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>timestamp</b></td>
<td><b>duration</b></td>
<td><b>number_of_trees</b></td>
<td><b>training_rmse</b></td>
<td><b>training_logloss</b></td>
<td><b>training_classification_error</b></td>
<td><b>validation_rmse</b></td>
<td><b>validation_logloss</b></td>
<td><b>validation_classification_error</b></td></tr>
<tr><td></td>
<td>2017-01-17 12:03:16</td>
<td> 0.018 sec</td>
<td>0.0</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:18</td>
<td> 2.216 sec</td>
<td>1.0</td>
<td>0.5951179</td>
<td>8.5421026</td>
<td>0.3883801</td>
<td>0.5236415</td>
<td>6.8440286</td>
<td>0.2817442</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:19</td>
<td> 2.854 sec</td>
<td>2.0</td>
<td>0.5879322</td>
<td>8.0989405</td>
<td>0.3799788</td>
<td>0.4518493</td>
<td>3.2032095</td>
<td>0.2403977</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:20</td>
<td> 3.387 sec</td>
<td>3.0</td>
<td>0.5767514</td>
<td>7.1878886</td>
<td>0.3760123</td>
<td>0.4255624</td>
<td>1.8792577</td>
<td>0.2220967</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:20</td>
<td> 3.919 sec</td>
<td>4.0</td>
<td>0.5565996</td>
<td>5.8767437</td>
<td>0.3620462</td>
<td>0.4141192</td>
<td>1.2854569</td>
<td>0.2159964</td></tr>
<tr><td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td>
<td>---</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:32</td>
<td>15.659 sec</td>
<td>28.0</td>
<td>0.4966274</td>
<td>0.7603848</td>
<td>0.3469849</td>
<td>0.3994254</td>
<td>0.5534639</td>
<td>0.2005197</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:32</td>
<td>16.154 sec</td>
<td>29.0</td>
<td>0.4960718</td>
<td>0.7481101</td>
<td>0.3463494</td>
<td>0.3992440</td>
<td>0.5441000</td>
<td>0.2001808</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:33</td>
<td>16.635 sec</td>
<td>30.0</td>
<td>0.4969369</td>
<td>0.7459109</td>
<td>0.3492019</td>
<td>0.3996000</td>
<td>0.5445013</td>
<td>0.2021012</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:33</td>
<td>17.124 sec</td>
<td>31.0</td>
<td>0.4965813</td>
<td>0.7387624</td>
<td>0.3488915</td>
<td>0.3994825</td>
<td>0.5445578</td>
<td>0.2015364</td></tr>
<tr><td></td>
<td>2017-01-17 12:03:34</td>
<td>17.625 sec</td>
<td>32.0</td>
<td>0.4966216</td>
<td>0.7351345</td>
<td>0.3488028</td>
<td>0.3992921</td>
<td>0.5438816</td>
<td>0.2026661</td></tr></table></div>


    
    See the whole table with table.as_data_frame()
    Variable Importances: 



<div style="overflow:auto"><table style="width:50%"><tr><td><b>variable</b></td>
<td><b>relative_importance</b></td>
<td><b>scaled_importance</b></td>
<td><b>percentage</b></td></tr>
<tr><td>lga</td>
<td>127887.0</td>
<td>1.0</td>
<td>0.1876133</td></tr>
<tr><td>quantity</td>
<td>62685.6757812</td>
<td>0.4901646</td>
<td>0.0919614</td></tr>
<tr><td>quantity_group</td>
<td>46434.4179688</td>
<td>0.3630894</td>
<td>0.0681204</td></tr>
<tr><td>region</td>
<td>36214.4062500</td>
<td>0.2831750</td>
<td>0.0531274</td></tr>
<tr><td>funder</td>
<td>33745.8203125</td>
<td>0.2638722</td>
<td>0.0495059</td></tr>
<tr><td>---</td>
<td>---</td>
<td>---</td>
<td>---</td></tr>
<tr><td>amount_tsh</td>
<td>4199.7519531</td>
<td>0.0328396</td>
<td>0.0061611</td></tr>
<tr><td>management_group</td>
<td>4159.1743164</td>
<td>0.0325223</td>
<td>0.0061016</td></tr>
<tr><td>source_class</td>
<td>3242.0720215</td>
<td>0.0253511</td>
<td>0.0047562</td></tr>
<tr><td>missing_construction_year</td>
<td>1844.0565186</td>
<td>0.0144194</td>
<td>0.0027053</td></tr>
<tr><td>num_private</td>
<td>580.7127075</td>
<td>0.0045408</td>
<td>0.0008519</td></tr></table></div>


    
    See the whole table with table.as_data_frame()





    



Just like in the sklearn implementation, from looking at the confusion marix we can see that the error is highest with label 1. It does relatively well with labels 0 and 2. The hit_ratio is the ratio of the number of times a correct prediction was made, to the total number of predictions.

The top features are also different from what we saw with the sklearn implementation, which upweighted the importance of continuous variables. In the H2O implementation the top features are all categorical.




