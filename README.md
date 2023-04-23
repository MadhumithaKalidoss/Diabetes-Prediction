# Diabetes-Prediction
## Overview

Diabetes is a chronic disease that affects millions of people worldwide. According to the World Health Organization, the global prevalence of diabetes among adults over 18 years of age has risen from 4.7% in 1980 to 8.5% in 2014. This disease can cause various health complications, such as heart disease, stroke, kidney failure, blindness, and amputations. Therefore, early detection and prevention are essential to reduce the burden of diabetes and its associated complications.

## Problem Definition

In this project, we aim to build a diabetes prediction system that can help identify individuals who are at high risk of developing diabetes. We will use machine learning algorithms to analyse various factors, such as glucose level, cholesterol levels and other relevant medical information, to accurately predict the likelihood of an individual developing diabetes

The purpose of this project is to provide a useful tool to identify patients who may benefit from early intervention and lifestyle modifications to prevent or delay the onset of diabetes. 

## Contributors

Praveena Vijayan - Data Preparation and Random Forest

Kalidoss Madhumitha - Data Visualisation of Categorical Variables and Logistic Regression

Nanditha Kumar - Data Visualisation of Numerical Variables and Support Vector Machine

## Data Collection & Preparation

In this project, we collected three datasets to train and evaluate our diabetes prediction model. 
The first two datasets, Big Data 1 and Big Data 2, contain various medical and demographic information about patients who have or have not been diagnosed with diabetes. These datasets contain categorical (binary) predictor variables such as physical activity and smoking habits.

To increase the sample size and diversity of our data, we concatenated both Big  Data 1 and Big Data 2 and then appended a smaller dataset, Small Data, containing only numerical predictor variables such as pregnancies and glucose. 

After cleaning the data by dropping unnecessary variables, removing duplicates and zeros in some of the numerical variables, we had a large dataset with over 100,000 data points and 18 variables.

To reduce the computational burden and improve model performance, we randomly sampled 1000 data points from this dataset. However, the process of appending big datasets to the smaller dataset resulted in some missing values in our data. To handle these missing values, we used the K-Nearest Neighbors (KNN) imputation technique to predict the null values based on the values of the k-nearest neighbours.

Final cleaned dataset (1000 data points and 17 predictor variables): 
Binary response variable: Diabetes_binary
Predictor numerical variables: BMI, MentHlth, PhysHlth, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction
Predictor categorical variables: HighChol, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, HvyAlcoholConsump, DiffWalk, AgeLevel

To be taken note: Since our project uses random sampling of data points, there can be slight differences in results and analysis as data points in the final cleaned dataset differ for every run. To address this, the code was run multiple times and our analysis is based on the consistency of results across different runs.

## Exploratory Data Analysis (EDA)

In this project, we performed extensive exploratory data analysis (EDA) to gain insights into our dataset and identify the most important predictor variables for our diabetes prediction model. Our dataset contains a mix of numerical and categorical variables, and the response variable is binary, indicating whether a patient has diabetes or not.

For the numerical variables, we calculated summary statistics such as mean, median, standard deviation, and range. We also visualised the distribution of each numerical variable using box plots, histograms, and violin plots. Additionally, we used joint plots to examine the relationship between each numeric variable and the response variable, and we plotted histograms separately for each level of the response variable.  We also used correlation matrix to visualise the correlation between the numerical predictor variables and the response variable, Diabetes_binary. This allowed us to observe any potential differences in the distribution of each variable between patients with and without diabetes.

For the categorical variables, we used count plots, heatmaps, and point plots to visualise the relationship between each categorical variable and the response variable. We also conducted a chi-square test to assess the association between each categorical variable and the response variable, and we calculated the chi-square statistic and p-value for each variable.

Based on the results of our EDA, we decided to focus on six variables for our machine learning model: Glucose, BMI, SkinThickness, DiffWalk, HeartDiseaseorAttack, and HighChol. We selected these variables based on either their correlation with the response variable or their association with diabetes according to the chi-square test. 

Overall, our EDA process allowed us to gain a better understanding of our dataset and select the most relevant variables for our diabetes prediction model.

## Machine Learning

We used three different machine learning models to predict whether a person has diabetes or not: logistic regression, random forest, and support vector machine (SVM). We selected these models because they can handle both categorical binary variables and numerical variables, as well as the binary response variable.

*Logistic Regression*

Logistic regression is a statistical model that analyses the relationship between the predictor variables and the binary response variable. It uses a logistic function to model the probability of the response variable being a particular outcome based on the predictor variables. In our case, the logistic regression model predicts the probability of a person having diabetes given their values for the predictor variables.

*Random Forest*

Random forest is an ensemble learning model that uses decision trees to create a forest of trees. It works by constructing multiple decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. In our case, the random forest model predicts whether a person has diabetes based on their values for the predictor variables.

*Support Vector Machine (SVM)*

Support vector machine (SVM) is a supervised learning model that analyses data for classification and regression analysis. It works by finding the hyperplane that maximally separates the classes. In our case, the SVM model predicts whether a person has diabetes based on their values for the predictor variables.

After building each of the models, we evaluated their accuracy and effectiveness through various metrics such as plotting confusion matrices for each model. We used these to assess the performance of the models and identify any areas for improvement.

Overall, our machine learning process involved selecting appropriate models, fitting the models to the data, and evaluating their performance.

## Conclusion
In this project, we built a model that can predict whether a person has diabetes or not based on their BMI, Glucose, SkinThickness, DiffWalk, HeartDiseaseorAttack, and HighChol. We collected and prepared data, explored the data using univariate, bivariate, and multivariate analysis, and selected the most appropriate predictor variables for our diabetes prediction model.

We used three different machine learning models: logistic regression, random forest, and support vector machine (SVM) to predict whether a person has diabetes or not. We evaluated each model's accuracy and effectiveness using confusion matrices and other metrics.

Overall, our model achieved good accuracy in predicting whether a person has diabetes or not. Our machine learning process showed that logistic regression was the most effective model for predicting diabetes in our dataset. 

This project demonstrates the value of data science and machine learning in healthcare, providing insights into factors that may contribute to diabetes and helping healthcare professionals make better decisions to improve patient outcomes.

## Data - Driven Insights

Explored new ways of visualising and analysing data

  -Point Plots
  
  -Chi Square Statistic

Explored new machine learning models

  -Logistic Regression
  
  -Random Forest
  
  -Support Vector Machine
  
 Newfound appreciation for model evaluation
 
  - A model producing the correct outcome may not immediately mean that it is the best model to use for our problem


## References

https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

https://www.kaggle.com/datasets/vikasukani/diabetes-data-set

https://analyticsindiamag.com/understanding-the-basics-of-svm-with-example-and-python-implementation/

**NTU Resources**

Coursematerials for SC1015:

M2 Basic Statistics

M2 Exploratory Analysis

M4 Classification Tree 
