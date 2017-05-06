
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*

# Welcome to the second project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Please specify WHICH VERSION OF PYTHON you are using when submitting this notebook. Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

# ----
# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'income'`, will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.

# In[1]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))


# ### Implementation: Data Exploration
# A cursory investigation of the dataset will determine how many individuals fit into either group, and will tell us about the percentage of these individuals making more than \$50,000. In the code cell below, you will need to compute the following:
# - The total number of records, `'n_records'`
# - The number of individuals making more than \$50,000 annually, `'n_greater_50k'`.
# - The number of individuals making at most \$50,000 annually, `'n_at_most_50k'`.
# - The percentage of individuals making more than \$50,000 annually, `'greater_percent'`.
# 
# **Hint:** You may need to look at the table above to understand how the `'income'` entries are formatted. 

# In[2]:

# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = np.count_nonzero(data['income'] == '>50K')

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = np.count_nonzero(data['income'] == '<=50K')

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (float(n_greater_50k)/float(n_records))*100

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)


# ----
# ## Preparing the Data
# Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

# ### Transforming Skewed Continuous Features
# A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 
# 
# Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed.

# In[3]:

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)


# For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.
# 
# Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed. 

# In[4]:

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)


# ### Normalizing Numerical Features
# In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.
# 
# Run the code cell below to normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.

# In[5]:

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 1))


# ### Implementation: Data Preprocessing
# 
# From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.
# 
# |   | someFeature |                    | someFeature_A | someFeature_B | someFeature_C |
# | :-: | :-: |                            | :-: | :-: | :-: |
# | 0 |  B  |  | 0 | 1 | 0 |
# | 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
# | 2 |  A  |  | 1 | 0 | 0 |
# 
# Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. In code cell below, you will need to implement the following:
#  - Use [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to perform one-hot encoding on the `'features_raw'` data.
#  - Convert the target label `'income_raw'` to numerical entries.
#    - Set records with "<=50K" to `0` and records with ">50K" to `1`.

# In[6]:

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
categorical = ['workclass', 'education_level', 'marital-status', 'occupation', 
               'relationship' ,'race', 'sex', 'native-country']
features = pd.get_dummies(features_raw[categorical])

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 0 if x == '<=50K' else 1)

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# See the encoded feature names
print encoded


# ### Shuffle and Split Data
# Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
# 
# Run the code cell below to perform this split.

# In[7]:

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, 
                                                    test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# ----
# ## Evaluating Model Performance
# In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a *naive predictor*.

# ### Metrics and the Naive Predictor
# *CharityML*, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that *does not* make more than \$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).
# 
# Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, *CharityML* would identify no one as donors. 

# ### Question 1 - Naive Predictor Performace
# *If we chose a model that always predicted an individual made more than \$50,000, what would that model's accuracy and F-score be on this dataset?*  
# **Note:** You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.

# In[8]:

# TODO: Calculate accuracy
from sklearn.metrics import accuracy_score
y_pred_naive = np.ones(n_records, dtype=np.int)
accuracy = accuracy_score(income, y_pred_naive)

# TODO: Calculate F-score using the formula above for beta = 0.5
from sklearn.metrics import fbeta_score
fscore = fbeta_score(income, y_pred_naive, beta = 0.5)

# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)


# ###  Supervised Learning Models
# **The following supervised learning models are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent Classifier (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Question 2 - Model Application
# List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
# - *Describe one real-world application in industry where the model can be applied.* (You may need to do research for this — give references!)
# - *What are the strengths of the model; when does it perform well?*
# - *What are the weaknesses of the model; when does it perform poorly?*
# - *What makes this model a good candidate for the problem, given what you know about the data?*

# **Answer: **
# In order to choose the appropriate supervised learning models i should find the important characteristics of the problem which i want to solve. These characteristics are:
# - I have enough data (45222).
# - I want to solve a **classification** problem.
# - My data are labeled. I know the correct output of every record.
# - My data are not text data, so i will not use Gaussian Naive Bayes.
# 
# After these observations i choose the following models to try:
# - Support Vector Machines (SVC)
# - K-Nearest Neighbors
# - Logistic Regression
# 
# More models to check:
# - Decision Trees
# - Ensemble Method (Gradient Boosting)
# 
# **Note**: It's not available for this project but i believe that a well-defined and well-trained neural network will performe pretty good.
# 
# **SVM**
# - Real-world applications:
#     - SVMs have demonstrated highly competitive performance in numerous real-world applications, such as **bioinformatics**, **text mining**, **face recognition** and **image processing**, which has established SVMs as one of the state-of- the-art tools for machine learning and data mining.
#     - Source: https://pdfs.semanticscholar.org/ed91/7e043ac071379de2f5890ef9ca51aa961039.pdf
#     
# - Advantages:
#     - It has a regularization parameter (C), which makes the user think about avoiding over-fitting. 
#     - It uses the kernel trick, so you can build in expert knowledge about the problem via engineering the kernel. With kernel you can seperate easier the data in different classes.
#     - An SVM is defined by a convex optimization problem (no local minima) for which there are efficient methods (e.g. SMO). So you model will not stuck to a local minimum. 
#     - It is an approximation to a bound on the test error rate.
#     - Effective in high dimensional spaces (many features).
#     - Still effective in cases where number of dimensions is greater than the number of samples.
#     - Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
#     - Sources:
#         - https://stats.stackexchange.com/questions/24437/advantages-and-disadvantages-of-svm
#         - http://scikit-learn.org/stable/modules/svm.html
# - Disadvantages:
#     - If the number of features is much greater than the number of samples, the method is likely to give poor performances. In this problem there isn't this danger (98 total features after one-hot encoding and 45222 available samples.
#     - SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.
#     - Training and prediction time are bigger than the corresponding time  of other models.
#     - Sources:
#         - http://scikit-learn.org/stable/modules/svm.html
# 
# **K-Nearest Neighbors**
# - Real-world application:
#     - Herta Security uses deep learning algorithms to generate feature vectors representing people’s faces. They then use **k-NN** to identify a person by compare the face to their watchlist(face recognition). The reason? k-NN is good enough and it’d be impractical to train a separate classifier for each person on the watchlist.
#     - Source: https://www.quora.com/What-are-industry-applications-of-the-K-nearest-neighbor-algorithm
# - Advantages:
#     - Robust to noisy training data. Out datasets hasn't noisy data.
#     - Effective for big training data. We have enoough data.
#     - Flexible to feature / distance choices
#     - The cost of the learning process is zero.
#     - No assumptions about the characteristics of the concepts to learn have to be done.
#     - Complex concepts can be learned by local approximation using simple procedures.
#     - Source:
#         - http://people.revoledu.com/kardi/tutorial/KNN/Strength%20and%20Weakness.htm
#         - http://www.cs.upc.edu/~bejar/apren/docum/trans/03d-algind-knn-eng.pdf
# - Disadvantages:
#     - Distance based learning is not clear which type of distance to use and which attribute to use to produce the best results. 
#     - High computation cost because we need to compute distance of each query instance to all training samples.
#     - Performance depends on the number of dimensions that we have (curse of dimensionality) =⇒ Attribute Selection
#     - Source:
#         - http://people.revoledu.com/kardi/tutorial/KNN/Strength%20and%20Weakness.htm
#         - http://www.cs.upc.edu/~bejar/apren/docum/trans/03d-algind-knn-eng.pdf
#         
# **Logistic Regression**
# - Real-world application:
#     - Image Segmentation and Categorization
#     - Geographic Image Processing
#     - Handwriting recognition
#     - Healthcare : Analyzing a group of over million people for myocardial infarction within a period of 10 years is an application area of logistic regression.
#     - Prediction whether a person is depressed or not based on bag of words from the corpus seems to be conveniently solvable using logistic regression and SVM.
#     - Source:
#         - https://www.quora.com/What-are-applications-of-linear-and-logistic-regression
# - Advantages:
#     - Fast algorithm, small run time.
#     - Is intrinsically simple, it has low variance and so is **less prone to over-fitting** compared to other models.
#     - Can accept a large number of independent variables.
#     - 
#     - Source:
#         - https://www.quora.com/What-are-the-advantages-of-logistic-regression-over-decision-trees
# - Disadvantages:
#     - Has an implicit assumption of linearity in terms of the logit function versus the independent variable. Sometimes the relationship isn't linear and the model performs poorly.
#     - Binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal. Logistic regression assumes linearity of independent variables and log odds.  Whilst it does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.  Otherwise the test underestimates the strength of the relationship and rejects the relationship too easily, that is being not significant (not rejecting the null hypothesis) where it should be significant. That's ok in our problem.
#     - The model should be fitted correctly.  Neither over fitting nor under fitting should occur.  That is only the **meaningful variables** should be included, but also all meaningful variables should be included.  A good approach to ensure this is to use a stepwise method to estimate the logistic regression. So it's very important to use meaningful features. In our problem the features seem to be significant (more than others).
#     -  The error terms need to be independent.  Logistic regression requires each observation to be independent.  That is that the data-points should not be from any dependent samples design, e.g., before-after measurements, or matched pairings.
#     - Source:
#         - http://www.statisticssolutions.com/assumptions-of-logistic-regression/
# 

# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
# In the code block below, you will need to implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
#    - Record the total prediction time.
#  - Calculate the accuracy score for both the training subset and testing set.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!

# In[9]:

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta = 0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta = 0.5)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results


# ### Implementation: Initial Model Evaluation
# In the code cell, you will need to implement the following:
# - Import the three supervised learning models you've discussed in the previous section.
# - Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
#   - Use a `'random_state'` for each model you use, if provided.
#   - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
# - Calculate the number of records equal to 1%, 10%, and 100% of the training data.
#   - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.
# 
# **Note:** Depending on which algorithms you chose, the following implementation may take some time to run!

# In[10]:

# TODO: Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# TODO: Initialize the three models
clf_A = KNeighborsClassifier()
clf_B = LogisticRegression()
clf_C = SVC(random_state = 0)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = X_train.shape[0]/100
samples_10 = X_train.shape[0]/10
samples_100 = X_train.shape[0]

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


# ----
# ## Improving Results
# In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 

# ### Question 3 - Choosing the Best Model
# *Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000.*  
# **Hint:** Your answer should include discussion of the metrics, prediction/training time, and the algorithm's suitability for the data.

# **Answer: ** --- Models Overall Performance ---
# - SVC:
#     - Advantages:
#         - High Accuracy Score on testing set (>0.8)
#         - High F-Score on Testing Set (>0.6)
#         - Very good performance on training set
#     - Disadvantages:
#         - Slowest training time (>90 sec)
#         - Slow prediction time (>15 sec)
#         
# - Logistic Regression:
#     - Advantages:
#         - Highest Accuracy Score on testing set (>0.8)
#         - Highest F-score on testing score (>0.6)
#         - Very fast on training and prediction session.
#     - Disadvantages:
#         - Best performance (better than SVC performance and KNN performance). That's dangerous because of possible overfitting.
#         
# - KNN:
#     - Advantages:
#         - Good performance on training and testing set (appr. 0.8 accuracy and appr. 0.6 f-score)
#     - Disadvantages:
#         - Very big prediction time. This is critical because after training you want the response of the model to be fast.
# 
# **Summary:**
# After these observations i am thinking to choose between SVC and LogisticRegression classifier. The decision is tough. SVM generally perform really good in classification and you can easily control the overfitting via C parameter. The back-step is that you should wait until the training is finished, but after that, if the model is well-defined, the performance will be satisfactorily. It's important that we have enough data to train the model good. In addition in SMV we can use different types of kernels. So, based on the evaluation and the above analysis i believe that the most appropriate model for the task of identifying individuals that make more than $50,000 is **SVC** classifier.
# 
# **Note:**
# - I focused mainly on the results that came out of the whole training and testing set.
# - I believe that the LogisticRegression classifier deserves a shot.

# ### Question 4 - Describing the Model in Layman's Terms
# *In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical or technical jargon, such as describing equations or discussing the algorithm implementation.*

# **Answer: **
# - Basic idea of SVM: Effectively, what SVMs do is take your data and draw the line (or hyperplane) that divides your dataset into groups of positive (>50K) and negative (<=50K) observations. Then, when you feed it a new data point, the algorithm figures out which side of the line or hyperplane the data point is on and spits back the predicted classification!
# - Training: The model takes every data and creates the Separating surfaces (hyperplane) between different classes. In our problem we want to separate two classes. One class is contained the people who earn more than 50K and the other the people who don't.
# - Prediction: The model classifies new data by determining which side of the hyperplane they lie on.
# 
# source: http://andybromberg.com

# ### Implementation: Model Tuning
# Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
# - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
# - Initialize the classifier you've chosen and store it in `clf`.
#  - Set a `random_state` if one is available to the same state you set before.
# - Create a dictionary of parameters you wish to tune for the chosen model.
#  - Example: `parameters = {'parameter' : [list of values]}`.
#  - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
# - Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
# - Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
# - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.
# 
# **Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!

# In[12]:

# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = SVC(random_state = 0)

# TODO: Create the parameters list you wish to tune
# parameters = {'penalty':('l1', 'l2'), 'C':[1, 10], 'verbose':[0,5]}
parameters =  [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
               {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},]

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(estimator = clf, param_grid = parameters, scoring = scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))


# ### Question 5 - Final Model Evaluation
# _What is your optimized model's accuracy and F-score on the testing data? Are these scores better or worse than the unoptimized model? How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  
# **Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.

# #### Results:
# 
# |     Metric     | Benchmark Predictor | Unoptimized Model | Optimized Model |
# | :------------: | :-----------------: | :---------------: | :-------------: | 
# | Accuracy Score |       0.2478        |       0.8242      |      0.8231     |
# | F-score        |       0.2917        |       0.6432      |      0.6410     |
# 

# **Answer: **
# - Compared to naive predictor the performance of the model is extraordinary.
# - The optimized model performs a little bit less good (really small - negligible difference in scores) than the unoptimized performance. We should consider that grid search checks the model in order to avoid overfitting. 

# ----
# ## Feature Importance
# 
# An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.
# 
# Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.

# ### Question 6 - Feature Relevance Observation
# When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data.  
# _Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?_

# **Answer:**
# - All features (98 total features after one-hot encoding):
#     - age (39, 50, 38, etc.)
#     - workclass (State-gov, self-emp, private, etc.)
#     - **educational_level** (Bachelors, Masters, Doctorate, 9th, etc.)
#     - education-num (13.0, 9.0, 14.0, etc.)
#     - marital-status (Never-married, Divorced, Married-civ-spouse, etc.)
#     - **occupation** (Prof-specialty, Handlers-cleaners, Exec-managerial, etc.)
#     - relationship (Not-in-family, Husband, Wife, etc.)
#     - race (White, Black)
#     - **sex** (Male, Female)
#     - **capital-gain** (2174.0, 0.0, 14084.0, etc.)
#     - capital-loss (0.0, 2042.0, etc.)
#     - **hours-per-week** (40.0, 13.0, 45.0, etc.)
#     - native-country (United States, Cuba, Jamaica, etc.)
# - Most important for prediction (ranking):
#     - occupation
#         - Some jobs are generally more well-paid than others.
#     - educational-level
#         - People with higher educational level (Masters or PhD) tend to earn more money.
#     - capital-gain
#         - When your capital grows, you probably earn more money.
#     - hours-per-week
#         - The more you work, the more you earn. Someone who works many hours has a tendency to earn more than 50K.
#     - sex
#         - Based on researches male tend to earn better job positions in businesses. Unfortunately racism is still out there.

# ### Implementation - Extracting Feature Importance
# Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
# 
# In the code cell below, you will need to implement the following:
#  - Import a supervised learning model from sklearn if it is different from the three used earlier.
#  - Train the supervised model on the entire training set.
#  - Extract the feature importances using `'.feature_importances_'`.

# In[14]:

# TODO: Import a supervised learning model that has 'feature_importances_'
clf = AdaBoostClassifier()

# TODO: Train the supervised model on the training set 
model = clf.fit(X_train, y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
vs.feature_plot(importances, X_train, y_train)


# ### Question 7 - Extracting Feature Importance
# 
# Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  
# _How do these five features compare to the five features you discussed in **Question 6**? If you were close to the same answer, how does this visualization confirm your thoughts? If you were not close, why do you think these features are more relevant?_

# **Answer:**
# - The five most relevant features for predicting if an individual makes at most or above $50,000 are:
#     - sex_Female
#     - occupation_Prof-specialty
#     - occupation_Exec-managerial
#     - relationship_Husband
#     - occupation_Other-service
# - I am close to the same answer. Based on model feature extraction the most important features are sex(5th in my ranking), occupation(1st in my ranking) and relationship(not in my ranking).
# - I don't know why the model thinks that relationship feature is important. Maybe a husband has a more stable life with good job. 

# ### Feature Selection
# How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 

# In[15]:

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print "Final Model trained on full data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))
print "\nFinal Model trained on reduced data\n------"
print "Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5))


# ### Question 8 - Effects of Feature Selection
# *How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?*  
# *If training time was a factor, would you consider using the reduced data as your training set?*

# **Answer:**
# - The performance of the model is not so good.
#     - Final Model trained on full data:
#         - Acc. --> 0.8231
#         - F-score --> 0.6410
#     - Final Model trained on reduced data:
#         - Acc. --> 0.8062
#         - F-score --> 0.5882
# - The performance of the model trained on reduced is still good. If the training time is critical i would consider using the reduced data as my training set. It's all about a trade-off. If you want to gain maximum performance you should train your model really good, but it's very important to be careful with the overfitting problem.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
