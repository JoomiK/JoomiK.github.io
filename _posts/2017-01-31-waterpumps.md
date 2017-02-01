---
layout: post
title: Predicting non-functional water pumps in Tanzania
---
Data: From a competition for [drivendata.org](https://www.drivendata.org/competitions/7/)  
Techniques: Classification, random forest, imputation, PCA   

### Links to Code:  
[Part I- EDA and cleanup](https://github.com/JoomiK/PredictingWaterPumps/blob/master/WaterPumps.ipynb)  
- The data  
- Visualizing water pumps and regions  
- Dropping some features  
- Preprocessing labels  
- Looking at features  

[Part II- Modeling](https://github.com/JoomiK/PredictingWaterPumps/blob/master/WaterPumpsII.ipynb)  
- Model selection and evaluation

---

### Summary of Project:  
Using data on water pumps in communities across Tanzania, can we predict the pumps that are functional, need repairs, or don't work at all?  
There are 39 features in total. They are described [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/#features_list).

Among the features are:  
* Construction year  
* Total static head (amount water available to waterpoint)  
* Funder of the well  
* Installer of the well  
* Location 

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

#### Visualizing water pumps
Apparently responsibility for water and sanitation service provision is decentralized, so local governments are responsible for water resource management. Luckily, we have information on which regions the water pumps are in. Perhaps this will be a good predictor.

![png](/images/WellMap.png)
[Interactive Map](https://joomik.carto.com/builder/3227f55e-d6ac-11e6-832f-0e3ebc282e83/embed)  

There is some "clumpiness" here; in the southeast you'll notice that there seems to be a higher proportion of non-functional pumps (red) than near Iringa, where you see a lot of green (functional).

#### Exploring the data  
Looking at some continuous variables:

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

#### Funders:

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


#### Some other problems with this data
A summary of other things to deal with in this dataset:

- installer: has NaNs, zeros, and many low frequency levels  
- scheme_name: has NaNs and many low frequency levels  
- scheme_management has NaNs
- population: has zeros and many low frequency levels*  
- construction_year: has zeros  
- permit: has NaNs  
- public_meeting: has NaNs  


#### Some options for dealing with this missing data 
1. Drop the rows (I don't want to do this in this case because this would be a large proportion of the data).** 
2. Mean/median/mode imputation (crude, can severely distort the distribution of the variable).  
3. Predict (in order to do this, features need to be correlated or related somehow).
4. Create new feature like "no funder data." 


- For the missing categorical variables, I'm going to use option 4.  
- For the missing population, I'll leave the zeros as is, for the first pass at building a model, but at some point later on I might try to predict these using (for example) clustering or knn.  
- For the missing construction_year, I'll add a 'missing_construction_year' feature, and also run my model with and without imputation with the median. In a later iteration, maybe I'll try to impute using a fancier algorithmn.

#### * Note on population: 
If I were really thorough I would try to fill in missing population data using some kind of census data. Also I looked into using the (previously dropped) subvillage feature here to predict missing population values, but it does not seem like a good idea, because it appears that the population they report here (presumably some small radius around the water pump) is not just the subvillage population. That is, when I looked at population values for subvillages, entries with the same subvillage often had wildly different populations.  


#### ** Note on missing values: 
It's easy to imagine here that some of these NaNs may not be missing at random (we can imagine that year info is missing more from older wells, for example).


```python
# Prep- convert the zeros in 'funder','installer', and 'population' to NaNs. 
merged_clean = merged.replace({'funder':0, 'installer':0, 'population':0}, np.nan)
merged_clean = merged.replace({'funder':'0', 'installer':'0'}, np.nan)
```

#### Dealing with low frequency/many levels
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

#### Construction year  

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



#### Adding feature for missing construction year data


```python
merged_clean['missing_construction_year'] = merged_clean['construction_year'].apply(lambda x: isNan(x))
```


```python
merged_clean.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>funder</th>
      <th>gps_height</th>
      <th>installer</th>
      <th>num_private</th>
      <th>basin</th>
      <th>region</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>lga</th>
      <th>...</th>
      <th>quality_group</th>
      <th>quantity</th>
      <th>quantity_group</th>
      <th>source</th>
      <th>source_type</th>
      <th>source_class</th>
      <th>waterpoint_type</th>
      <th>waterpoint_type_group</th>
      <th>status_group</th>
      <th>missing_construction_year</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69572</th>
      <td>6000.0</td>
      <td>Roman</td>
      <td>1390</td>
      <td>Roman</td>
      <td>0</td>
      <td>Lake Nyasa</td>
      <td>Iringa</td>
      <td>11</td>
      <td>5</td>
      <td>Ludewa</td>
      <td>...</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>spring</td>
      <td>spring</td>
      <td>groundwater</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8776</th>
      <td>0.0</td>
      <td>Grumeti</td>
      <td>1399</td>
      <td>GRUMETI</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Mara</td>
      <td>20</td>
      <td>2</td>
      <td>Serengeti</td>
      <td>...</td>
      <td>good</td>
      <td>insufficient</td>
      <td>insufficient</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>34310</th>
      <td>25.0</td>
      <td>Lottery Club</td>
      <td>686</td>
      <td>World vision</td>
      <td>0</td>
      <td>Pangani</td>
      <td>Manyara</td>
      <td>21</td>
      <td>4</td>
      <td>Simanjiro</td>
      <td>...</td>
      <td>good</td>
      <td>enough</td>
      <td>enough</td>
      <td>dam</td>
      <td>dam</td>
      <td>surface</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>67743</th>
      <td>0.0</td>
      <td>Unicef</td>
      <td>263</td>
      <td>UNICEF</td>
      <td>0</td>
      <td>Ruvuma / Southern Coast</td>
      <td>Mtwara</td>
      <td>90</td>
      <td>63</td>
      <td>Nanyumbu</td>
      <td>...</td>
      <td>good</td>
      <td>dry</td>
      <td>dry</td>
      <td>machine dbh</td>
      <td>borehole</td>
      <td>groundwater</td>
      <td>communal standpipe multiple</td>
      <td>communal standpipe</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19728</th>
      <td>0.0</td>
      <td>Action In A</td>
      <td>0</td>
      <td>Artisan</td>
      <td>0</td>
      <td>Lake Victoria</td>
      <td>Kagera</td>
      <td>18</td>
      <td>1</td>
      <td>Karagwe</td>
      <td>...</td>
      <td>good</td>
      <td>seasonal</td>
      <td>seasonal</td>
      <td>rainwater harvesting</td>
      <td>rainwater harvesting</td>
      <td>surface</td>
      <td>communal standpipe</td>
      <td>communal standpipe</td>
      <td>2</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



#### Summary of the number of missing values for each column


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

#### Dealing with categorical variables.
Convert categorical variables into dummy/indicator variables. At the same time, we'll be adding columns to indicate NaNs. 


```python
merged_clean_dum = pd.get_dummies(merged_clean, dummy_na=True)
```


```python
merged_clean_dum.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>amount_tsh</th>
      <th>gps_height</th>
      <th>num_private</th>
      <th>region_code</th>
      <th>district_code</th>
      <th>population</th>
      <th>construction_year</th>
      <th>status_group</th>
      <th>missing_construction_year</th>
      <th>funder_0</th>
      <th>...</th>
      <th>waterpoint_type_improved spring</th>
      <th>waterpoint_type_other</th>
      <th>waterpoint_type_nan</th>
      <th>waterpoint_type_group_cattle trough</th>
      <th>waterpoint_type_group_communal standpipe</th>
      <th>waterpoint_type_group_dam</th>
      <th>waterpoint_type_group_hand pump</th>
      <th>waterpoint_type_group_improved spring</th>
      <th>waterpoint_type_group_other</th>
      <th>waterpoint_type_group_nan</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69572</th>
      <td>6000.0</td>
      <td>1390</td>
      <td>0</td>
      <td>11</td>
      <td>5</td>
      <td>109</td>
      <td>1999.0</td>
      <td>2</td>
      <td>False</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8776</th>
      <td>0.0</td>
      <td>1399</td>
      <td>0</td>
      <td>20</td>
      <td>2</td>
      <td>280</td>
      <td>2010.0</td>
      <td>2</td>
      <td>False</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34310</th>
      <td>25.0</td>
      <td>686</td>
      <td>0</td>
      <td>21</td>
      <td>4</td>
      <td>250</td>
      <td>2009.0</td>
      <td>2</td>
      <td>False</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>67743</th>
      <td>0.0</td>
      <td>263</td>
      <td>0</td>
      <td>90</td>
      <td>63</td>
      <td>58</td>
      <td>1986.0</td>
      <td>0</td>
      <td>False</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19728</th>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>2</td>
      <td>True</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 7069 columns</p>
</div>




```python
merged_clean_dum.shape
```




    (59400, 7069)



I now have 7069 features.


```python
merged_clean_dum.to_csv('merged_clean_dum.csv')
```

#### A note on other features of interest  for this project
It's easy to imagine that we could improve our predictions with other sources of data here. Some that come to mind in this case are:  
- Presence of other utilities nearby  
- Rate of previous breakdowns  
- Distance to major road  
- Whether the water pump is in a hazardous or flooding area
- Crime rate in the near vicinity  

#### Up next:
For Part II - dealing with missing values + model selection and evaluation


```python

```


