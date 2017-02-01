---
layout: post
title: Predicting non-functional water pumps in Tanzania
---
Data: From a competition for [drivendata.org](https://www.drivendata.org/competitions/7/)  
Techniques: Classification, random forest, imputation, PCA

Using data on water pumps in communities across Tanzania, can we predict the pumps that are functional, need repairs, or don't work at all?
 

#### Links to Code:  
[Part I- EDA and cleanup](https://github.com/JoomiK/PredictingWaterPumps/blob/master/WaterPumps.ipynb)  
- The data  
- Visualizing water pumps and regions  
- Dropping some features  
- Preprocessing labels  
- Looking at features  

[Part II- Modeling](https://github.com/JoomiK/PredictingWaterPumps/blob/master/WaterPumpsII.ipynb)  
- Model selection and evaluation

### Summary of Project:  
There are 39 features in total. They are described [here](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/#features_list).

Among the features are:  
* Construction year  
* Total static head (amount water available to waterpoint)  
* Funder of the well  
* Installer of the well  
* Location  

#### Visualizing water pumps
Apparently responsibility for water and sanitation service provision is decentralized, so local governments are responsible for water resource management. Luckily, we have information on which regions the water pumps are in. Perhaps this will be a good predictor.
![png](/images/WellMap.png)
[Interactive Map](https://joomik.carto.com/builder/3227f55e-d6ac-11e6-832f-0e3ebc282e83/embed)  
There is some "clumpiness" here; in the southeast you'll notice that there seems to be a higher proportion of non-functional pumps (red) than near Iringa, where you see a lot of green (functional).

#### Problems with the data
* Missing data  
* Categorical data with many levels  
