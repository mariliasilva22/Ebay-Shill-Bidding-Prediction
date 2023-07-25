# Ebay-Shill-Bidding-Prediction
The main goal of this project is to use classification algorithms to predict shill bids in the future. 

## 1. Introduction
In this project, I will be working with the dataset Shill biddig available at: https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset.

Customers throughout the world have the option of bidding on eBay. A 'bid' is the price offered by anyone participating in this procedure. Shill bidding occurs when a seller deliberately inflates the final price of his or her auction by faking bids. In the data set, normal bids are labelled as '0', while abnormal bids are labelled as '1'.

The main goal is to use classification algorithms to predict bids in the future. The difference between regression and classification models is the nature of the dependent variable which can be qualitative, in the case of classification, and continuous in the case of regression.

For this purpose, I will divide this project into Data Understanding, EDA, Data Preparation, PCA vs LDA, Class Imbalance, Modelling, Comparation between models chosen and Conclusion.

**Data Dictionary:**

![image-2.png](attachment:image-2.png)

## 2. Data Understanding

### 2.1 Importing Libraries
The first step of data preparation is to import the libraries necessary in the cell bellow. I imported the libraries for dataframe, visualisation, preprocessing, modelling and measurement.
### 2.2 Reading 
Then I will be uploading the dataset and reading the first and last 5 rows of the data set. Checking the size of the data set. Using the function .info(), I want to see if we have any missing values and what type of observations we have. Checking the information about the dataset, we can affirm that there are 12 features, which 8 are float type, 3 as integer type and 1 as object type. And a total of 6321 observations.
### 2.3 Measures of Variability
Using the function .describe() to obtain the distribution of variables, including mean, median, min, max, and the different quartiles. Transpose used to provide a better visualisation. 
We can see that the data set consists primarily of numerical data and most of the values range between 0 to 1. This does not happen with the variable 'Auction_Duration', 'Bidding_Ratio' and 'Auction_Bids'. The variable 'Auction_Duration' seem to follow a normal distribution and this will be analysed further ahead. We can also observe how the variable 'Class' is more represented by 0, as well as the variables 'Winning_Ratio' and 'Successive_Outbidding' seem to have plenty of observations clustering around either 0 or 1 values.

## 3. Exploratory Data Analysis
Exploratory Data Analysis is important to visualize and manipulate the data as well as summarize it with its main characteristics, using data visualization methods. It helps us to manipulate the data to take the best decision regards to it and to get answers for what we need.

### 3.1 Missing Values
We do not observe any missing values, which means we will not have to handle them.
Using the function .nunique() to see "the number of unique values for each column".

### 3.2 Treating Outliers
One of the most important step while preparing the data set is to detect and to deal with Outliers. An outlier is any extrem value in the data set that can be an error or not. It can affect the accurancy in the machine learning models.

### 3.3 IQR (Interquartile Range)
IQR tells us the variation and spread of all data points within one quartile of the average. Any value beyond the range of -1.5 x IQR to 1.5 x IQR is considered an outiler. The IQR is calculated by subtracting Q1 from Q3. 
I will be using this technique to remove the outiliers.

### 3.4 Distribution of Variables
The variables Bidder_tendency and Bidding_ratio are right skewed.
By seeing the distributions we can have an idea how skewed are these features, and also to observe that Successive_Outbidding variable it is not outlier since it is represented by only 3 values 0, 0.5 and 1.

### 3.5 Distribution of Bidder Tendency and Bidding Ratio
### 3.6 Correlation Among Variables
Number of variables with > 0.90 correlation:  1
The heatmap above shows that there are a few variables with strong correlation, others with moderate correlation and others with almost no correlation. For instance, the variable Last_bidding has more than 90% of correlation with variable Early_Bidding.

### 3.7 ScatterPlots
As informed previously, the heatmap showed how strong is the correlation among variables. The variables Last_Bidding and Early_Bidding has showed a strong positive correlation (0.96). It is clear, by looking at the scatterplot below, that there is a pattern between the variables.

### 3.8 Distribution of Class
We can notice how imbalanced is the original dataset. Most of the observations are normal behaviour bidding (89%). It is likely that I will get a lot of errors if I use this dataframe for predictive models and analysis, since the algorithms will "assume" that most bids are normal. However, I will do experiments and see the model's accuracy both in the balanced and unbalanced dataset in the Modelling part, and observe if it will affect my results or not.

I will be balancing the dataset in the 6. Class Imbalance session.

## 4. Data Preparation
In this part of the project, Preprocessing techniques were used to prepare the dataset for modelling process. It was performed Min Max scaler after removing the outliers, even though most of the values are between 0 to 1, except for the variable Auction_duration I decided to scale because some algoriths such as LDA, PCA, Logistic Regression, SVM and k nearest neighbor converge faster when features are relatively smaller or closer to normal distribution. (Gogia, 2019)

The variables to be analysed were stored in X and the variable target was stored in y. After that was applied PCA and LDA both in unbalanced and balanced data set using SMOTE technique.

### 4.1 Features Importances
For bidding class prediction is important to know wich features are the most important. To evaluate features on a classification task, this method uses a forest of trees. (scikit-learn, n.d.) The bars show the feature importances of the tree.
We see that four the most important features for predicting the target variable are: Successive_Outbidding; Auction_Duration; Winning_Ratio, and Last_Bidding.

And the less important is Starting_Price_Average.

### 4.2 Feature Scaling
Although most of the features seems to be already scaled, I will rescale them in order to obtain a normalized data set using the MinMaxScaler. Thus, the entire data set will range between 0 to 1. "This method subtract the min value and divide by the max minus the min." (Aurélien Géron, 2017, p.93)
Now I have all the values ranging from 0 to 1 and my data set is ready for the next step.

### 4.3 Spliting in train and test
The model is not created from the entire data set. Some data are randomly selected and some kept aside for checking the accuracy of the model. The testing data represents the data being tested, and the training data represents the data on which the model will be built.

## 5. PCA vs LDA
Principal Component Analysis (PCA) is a dimentionally reduction technique that reduce the number of features (principal componentes), the dimentionality of a dataset and maximize the variance. These features are projected in a lower dimensional space. (Pratap Dangeti, 2017, pp.320)

Linear Discriminant Analysis (LDA) is a technique to reduce dimentionality findind the direction that maximizes difference between two classes. (Pratap Dangeti, 2017, pp.283)

Although we apply both PCA and LDA in order to reduce the number of features and the dimensionality of a dataset, LDA is actually a classification algorithm based on strength of relationship between independent and dependent variable and is a very good technique to apply before running classification models.

### 5.1 PCA
Since we have an unbalanced data set I will perform PCA both in the original data set and after balancing, using SMOTE technique.
### 5.2 LDA
The classes can be well-separated in this two-dimensional space. Comparing the PCA graph with LDA graph we can conclud that using LDA for this data set we can have a better result regards to classification problem.

## 6. Class Imbalance
When the classes are not represented equally, this characterizes an unbalanced dataset. It is a problem for classification problems because the classification models are likely to predict everything as the majority class. Learning from highly imbalanced datasets was often considered a problem in the past. (Brownlee, 2020)

To balance the data set I decided to apply oversampling technique instead of undersampling as the undersampling process involves eliminating some observations of the majority class. This can be a good choice when we have a very bid data (around millions of rows) which is not the case, as this data set is not that big. So I decided not to apply undersampling to balance the data set since it could remove valuable information that could lead to underfitting. (Boyle, 2019)

### 6.1 SMOTE technique
SMOTE is a technique for unbalanced dataset that oversample the minority class. It works by drawing lines between close examples in feature space and picking a random point on the line as the new instance. I will import the librarie and then I will apply it only in the train set and use the test set to evaluate it. When we use any sampling technique we should divide our data first and apply synthetic sampling only on the training data if not doing that we could apply the synthetic data on the test set and our model will simply memorize and cause overfitting. (Boyle, 2019)

## 7. Model Building
There are two main types of machine learning models: Supervised, and Unsupervised. In supervised learning there are labels, which means, a labeled data set is used and we know what the output values for our samples should be. The feature Class in our data set is a label indicating if a bidding is fake or not. In the data set, normal bids are labeled as '0', while abnormal bids are labeled as '1'. Using a group of other features we can predict if a new bid is fake or not. Problems of this type are known as classification problems. In unsupervised learning labels are not available and we make predictions based on patterns in the dataset. (Patel, 2019, pp.25, 30)

As I am aware that the dataset is highly imbalanced, my first step is to check the performance of the imbalanced dataset, then I will implement SMOTE technique for balancing it and then I will check the performance again. Finally, I will compare the performance of each model.

### 7.1 Modelling on the original dataset
#### 7.1.1 Logistic Regression
Logistic Regression is a Supervised machine learning used for classification tasks. I chose this algorithm mainly because logistic regression is intended for binary (two-class) classification problems. (Brownlee, 2016)
To tune the parameters I will use GridSearchCV. GridSearchCV is as search over specified parameter values for an estimator. In order to determine the best combination of parameters, the algorithm runs through all the parameters that are entered into the parameter grid.(Okamura, 2020)

#### 7.1.2 Decision Tree Classifier
"The decision tree is one of the most intuitive and popular data mining methods, especially as it provides explicit rules for classification and copes well with heterogeneous data, missing data and non-linear effects." (Stéphane Tufféry, 2011)

Decision Tree is a Supervised Machine Learning Algorithm that by splitting the data into subsets, the algorithm builds a tree for each subset and then combines those trees into a single tree. The idea is to split the dataset into yes/no questions and then isolate data points belonging to each class until we obtain all of the data points. A node is added to the tree every time we ask a question. After asking a question, the dataset is split into new nodes based on the value of a feature. As a result, only one class can be assigned to the data points in each leaf node. (Aurélien Géron, 2019)

#### 7.1.3 K-nearest Neighbors
K-nearest Neighbors is a Supervised Machine learning algorithm that assumes that similiar things are near to each other by calculating the distance between points on a graph. According to Harrison (2018) there are many ways to calcule these distances. However, the Euclidean distance is the most popular way.

I chose this algorithm to work with because KNN works very well with lower dimensional data (as we applied LDA) and perform better than other techniques. We will see this at the end, in Compararing Models Session. (Brownlee, 2016a)

### 7.2 Modelling after SMOTE

## 8. Comparing Models

### 8.1 Comparing Predictions vs Real Values

## 9. Conclusion
This project was divided into sections. In the first steps was done data understanding, EDA, which involved data cleaning, data preprocessing and plotting some graphs to visualize some characteristics. Then, was done Data preparation, defining the most important features that can be used in the future for data mining and feature scaling using MinMaxScaler and splitting the data into train and test sets.

After that, through the application of two dimensionality reduction techniques, I was able to compare the results. It was possible to observe through visualisations that for this classification problem, LDA performs better than PCA due to the separation of classes. Although we apply both PCA and LDA to reduce the number of features and the dimensionality of a dataset, LDA is a classification algorithm based on strength of the relationship between an independent and dependent variable and is a very good technique to apply before running classification models.

It was possible to observe that the data set was highly imbalanced with 89% of the observations in Class classified as 'Normal'. So it was applied oversampling technique (SMOTE) to balance the dataset and see how the models would perform after this.

Finally, the last section was to perform the models. As the dataset chosen presented a linear relationship between some variables, has a binary class and also is a lower-dimensional data (as we applied LDA), I chose to perform the Logistic Regression, Decision Tree Classifier and K-nearest Neighbor models both in the original data set and in the balanced data set with SMOTE.

I tuned hyperparameters and compared performance of all models and we could observe that they performed very well and satisfied the accuracy requirements on both the original data set and on the data set after applying SMOTE, as shown by the accuracy of the sets below:

image.png

It was possible to observed that after applying SMOTE, the training and test scores increased in all models, except in Logistic Regression which was slightly overfitting since the training set score is higher than the test set score which confirms the overfitting.

It appears that for this specific problem, KNN and Decision Tree may be a better choice of model to predict the bids in the future.

Suggestions for Future Research:

A similar experiment can be conducted by balancing the data set with Random Under Sample for comparison purposes.

## 10. Reference list
1. archive.ics.uci.edu. (n.d.). UCI Machine Learning Repository: Shill Bidding Dataset Data Set. [online] Available at: https://archive.ics.uci.edu/ml/datasets/Shill+Bidding+Dataset [Accessed 29 Apr. 2022].  


2. Aurélien Géron (2017). Hands-on machine learning with Scikit-Learn and TensorFlow : concepts, tools, and techniques to build intelligent systems. Sebastopol, Ca: O’reilly Media, pp.93, 273.  


3. Aurélien Géron (2019). Hands-on machine learning with Scikit-Learn and TensorFlow concepts, tools, and techniques to build intelligent systems. O’Reilly Media, Inc.


4. Boyle, T. (2019). Methods for Dealing with Imbalanced Data. [online] Medium. Available at: https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18 [Accessed 15 May 2022].


5. Brownlee, J. (2016a). K-Nearest Neighbors for Machine Learning. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/#:~:text=Lower%20Dimensionality%3A%20KNN%20is%20suited [Accessed 15 May 2022].


6. Brownlee, J. (2016b). Logistic Regression for Machine Learning. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/logistic-regression-for-machine-learning/.


7. Brownlee, J. (2016). Logistic Regression for Machine Learning. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/logistic-regression-for-machine-learning/.


8. Brownlee, J. (2020). Imbalanced Classification with the Fraudulent Credit Card Transactions Dataset. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/imbalanced-classification-with-the-fraudulent-credit-card-transactions-dataset/ [Accessed 15 May 2022].


9. Gogia, N. (2019). Why Scaling is Important in Machine Learning? [online] Analytics Vidhya. Available at: https://medium.com/analytics-vidhya/why-scaling-is-important-in-machine-learning-aee5781d161a [Accessed 3 May 2022].


10. Grant, P. (2022). Using Python to Find Outliers With IQR: A How-To Guide. [online] Medium. Available at: https://towardsdatascience.com/using-python-to-find-outliers-with-iqr-a-how-to-guide-1197f2929a12 [Accessed 2 May 2022]. 


11. Harrison, O. (2018). Machine Learning Basics with the K-Nearest Neighbors Algorithm. [online] Medium. Available at: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761 [Accessed 15 May 2022].


12. pandas.pydata.org. (n.d.). pandas.DataFrame.describe — pandas 1.3.4 documentation. [online] Available at: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html [Accessed 29 Apr. 2022].  


13. Patel, A.A. (2019). Hands-On Unsupervised Learning Using Python : how to build applied machine learning solutions from unlabeled data. Sebastopol, Ca O’reilly Media, pp.25, 30.



14. Pratap Dangeti (2017). Statistics for machine learning build supervised, unsupervised, and reinforcement learning models using both Python and R. Birmingham ; Mumbai Packt Publishing July, pp.11, 60, 320, 328.  


15. scikit-learn developers (2019). sklearn.neighbors.KNeighborsClassifier — scikit-learn 0.22.1 documentation. [online] Scikit-learn.org. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html [Accessed 17 May 2022].


16. scikit-learn. (n.d.). Feature importances with a forest of trees. [online] Available at: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html [Accessed 10 May 2022].  


17. Scikit-learn.org. (2009). sklearn.tree.DecisionTreeClassifier — scikit-learn 0.21.3 documentation. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.


18. Scikit-learn.org. (2014). sklearn.linear_model.LogisticRegression — scikit-learn 0.21.2 documentation. [online] Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html [Accessed 13 May 2022].  


19. Sharma, N. (2018). Ways to Detect and Remove the Outliers. [online] Towards Data Science. Available at: https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba [Accessed 7 May 2022].  


20. Stéphane Tufféry (2011). Data mining and statistics for decision making. Chichester, West Sussex ; Hoboken, Nj.: Wiley, p.302.  


21. www.w3schools.com. (n.d.). Pandas DataFrame nunique() Method. [online] Available at: https://www.w3schools.com/python/pandas/ref_df_nunique.asp#:~:text=The%20nunique()%20method%20returns [Accessed 28 Apr. 2022].
