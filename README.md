# PySparkMLRegression

### Overview

This repository contains example code for creating ML models using PySpark.

These examples were initially created to demonstrate how to find the threshold at which diastolic and systolic blood pressure result in a diagnosis of hypertension using scikit-learn. I thought it would be useful to see how to do this using SparkML, so I ported the example to use the PySparkML library. Beacuse I don't expect everyone to have clinical data available, I also provided a workbook, "Generate_Dummy_BP_data.ipynb", to randomly generate bmi, systolic, and diastolic data. This script doesn't accuratly reflect the distribution and correlation between these variables, it's there to provide a way to get the script running. I also checked in a file, "hypertensive1.csv", that contains 10,000 randomly generated rows, so you don't have to run the script to get the example working. 

I thought it might also be useful to provide examples based on more publicly available dataset with more interesting trends and correlations, so I ported a few examples to use the gapminder dataset, a csv file that provides historical life expectency, per capity gdp, and population growth for various countries between the 1957 and 2007. 

The dataset used here, "gapminder_all_binary.csv", is based on a dataset used for the software carpentry lesson "Plotting and Programming in Python" (https://swcarpentry.github.io/python-novice-gapminder/). I made one alteration - I added a binary field, "Over 65", indicating whether the life expectency for a particular country in 2007 was over or under 65 (1 for yes, 0 for no). I use this field for examples where I train a random forest (among other tree-based regressions) to determine the threshold for when this field should map to 1 or 0 (because this is a contrived example where I specifically coded these values to a threshold of 65, we know in advance that the ML model should return not too far off that predetermined threshold). 

### Workbooks

* Pyspark_continuous_regression.ipynb provides an example for predicting life expectancy based on GDP as a continuous variable (using linear regression and decision trees). 

* PySpark_binary_regression.ipynb uses a decision tree and a random forest to predict a binary value (whether the life expectency is above 65).

* PySpark_CSV_SQL demonstrates how to run a SQL command on a dataframe created through a CSV import into a SQLContext.

* PySpark_binary_regression_BP.ipynb uses PySpark tree based regression to predict the probabilty that a patient is coded as hypertensive based on systolic and diastolic blood pressure readings, along with some analysis and interpretation of the model.

* Generate_Dummy_BP_Data.ipynb is a script to generate randomized bmi, systolic, and diastolic readings. These values are generated as uncorrelated variables uniformly distributed between a range. In other words, this dummy data doesn't reflect any of the real-world intricacies of blood pressure data. 

* Pyspark_scikit_learn_RF_binary_regression.ipynb is based on an example written by Hunter Mills at UCSF that uses scikit-learn in a PySpark3 kernel to find the threshold for systolic and/or diastolic blood pressure readings coded as hypertension. This approach allows you to use PySpark in a clustered environment for creating dataframes from SQL queries against a very large database, taking full advantage of the distributed environment. This example then switches to pandas and scikit-learn to build, fit, run, and access results on a random forest regression. This approach is useful when you need a spark cluster to filter and prepare data, but can switch to a single-node approach for training and running an ML model.

