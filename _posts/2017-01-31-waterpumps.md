---
layout: post
title: Predicting non-functional water pumps in Tanzania
---
Data: From a competition for [drivendata.org](https://www.drivendata.org/competitions/7/)  
Techniques: Random forest, classification, PCA, imputation

Using data on water pumps in communities across Tanzania, can we predict the pumps that are functional, need repairs, or don't work at all?
The rich feature space is derived from a mix of continuous and categorical variables; examples include information about what kind of pump is operating, when it was installed, the manager, the region, etc. 

### Outline:  
[Part I- EDA and cleanup](https://github.com/JoomiK/PredictingWaterPumps/blob/master/WaterPumps.ipynb)  
- The data  
- Visualizing water pumps and regions  
- Dropping some features  
- Preprocessing labels  
- Looking at features  

[Part II- Modeling](https://github.com/JoomiK/PredictingWaterPumps/blob/master/WaterPumpsII.ipynb)  
- Model selection and evaluation
