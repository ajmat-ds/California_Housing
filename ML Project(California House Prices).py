#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# <a id='import'></a>

# In[1]:


# Common imports
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# random seed to make output stable across runs
np.random.seed(42)


# <a id='data'></a>

# # **California housing Dataset**
# 
# The dataset we will use is the "California Housing Prices" dataset from the statlib repository, which is based on data from the 1990 census. This dataset offers great opportunities for learning. The prediciton task for this dataset wil be to predict housing prices based on several features.

# ### Get the data:

# In[1]:


import os
import tarfile
from six.moves import urllib

root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
path = os.path.join("datasets", "housing")
source = root + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=source, path=path):
    if not os.path.isdir(path):
        os.makedirs(path)
    tgz_path = os.path.join(path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=path)
    housing_tgz.close()


# In[2]:


fetch_housing_data()


# In[4]:


import pandas as pd

def load_data(housing_path=path):
    csv = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv)


# <a id='bigpop'></a>

# # TAKE A QUICK LOOK AT THE DATA STRUCTURE
# 
# We are asked to build a machine learning model that predicts the median house value for a given "district". This data has metrics like population, median income, median housing price for each block group in califorania. Block groups are smallest geographical unit for which the US Census Bureu publishes data. 
# 
# Our problem is clearly a supervised learning task, because we have labeled input data. It is also clearly a regression task since we have to predict a numeric value. 

# In[5]:


housing = load_data()
housing.head()


# We can see the top 5 rows of the dataset with the "head()" method. 
# 
# Each row  represents one district. The dataset has 10 attributes: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, median_house_value and ocean_proximity.

# In[6]:


housing.info()


# Now we can see a brief description of the data using the "info()" method above.
# 
# Here we see that there are 20,640 entries in the whole dataset. Also notice that the total_bedrooms attribute has only 20,433 non-null values which means that 207 districts don't have this feature. We can also see that ocean_proximity is not numerical and probably a categorical attribute.We can find out how many districts belong to each attribute by using the "value_counts()" method.

# In[7]:


housing["ocean_proximity"].value_counts()


# Now I will use the "describe()" method to show a summary of the numerical attributes:

# In[8]:


housing.describe()


# The count, mean, min and max rows are portray same meanings as their name suggests. Here we see that all null values are ignored for example(count of total_bedrooms is 20,433 not 20,640). The std row shows you the standard deviation, which quantifies the amount of variation between the values. The 25%, 50%, 75% are the percentiles. For an example 75% of the districts have housing_median_age of lower than 37 while 50% are lower than 29 and 25% are lower than 18. These are often called the 25th percentile (or 1st quartile), the median and the 75th percentile (or 3rd quartile).
# 
# Another good way to explore a given dataset is to plot a histogrom of each numerical attribute. A histogram shows the number of entries (vertical axes) and the number of the given value range (horizontal axes).

# In[9]:


import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))


# Few noticeable things in these histograms::
# 
# * The attributes have varying scales.
# * Many of the histograms are "tail heavy" which means that they extend further to the right of the median than to the left which makes it harder for an algortihm to detect patterns. We will transform this later on.
# * The median income attribute does not look like it is expressed in US dollars (USD). After checking with the team that collected the data, you are told that the data has been scaled and capped at 15(actually 15.0001) for higher median incomes, and at 0.5(actually 0.4999) for lower median incomes. This is called a "preprocessed attribute" and is common within machine learning, but you should understand how the data was preprocessed.
# * The housing_median_age and the median_house_value attributes are also capped. That the median_house_value is capped could be a serious problem, because this is our label (what we want to predict) and our model could learn that the price never goes beyond that limit. In this case we only have the option to remove the capped one or to collect the right labels for them.
# 
# 

# <a id='split'></a>

# # Train/Test Split
# 
# It is important that we now set a part of the data aside.Our brain is an amazing pattern detection system, which means that it is highly prone to overfitting: if you look at the test set, you may stumble upon some seemingly interesting pattern in the test data that leads you to select a particular kind of Machine Learning model. When you estimate the generalization error using the test set, your estimate will be too optimistic and you will launch a system that will not perform as well as expected. This is called data snooping bias.
# Creating a test set is theoretically quite simple: just pick some instances randomly, typically 20% of the dataset, and set them aside.

# In[21]:


#We will use the straight forward "train_test_split()" method from sklearn to split our data into train and test subsets.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(len(train_set), "Train Instances +", len(test_set), "Test Instances")


# Now we splitted our data purely random, which is fine if you have a large dataset, but if it is not, you could have a sampling bias. When a company decides to call 10,000 people because of a survey, they want to make sure that these people represent the whole population. For an example the US population consists out of 49% male and 51% female, so a well conducted survey would try to maintain this ratio, which is called **stratified sampling**.  The population is divided into homogeneous subgroups, called **strata** and the right number of instances is sampled from each **stratum** to makes sure that the data really represents the population. 
# 
# If they used purely random sampling, there would be about 12% chance of sampling a skewed test set with either less than 49% female or more than 54% female. Either way, the survey results would be significantly biased.
# 
# Imagine that you talked with real estate experts and they tell you, that the median_icome attribute is very important, when you want to predict housing prices. If that's the case, you want a test set that is really representative of the income categories of the dataset. Because median_income is continous, you need to convert it into a categorical attribute. If you look Most median income values are clustered around 2–5 (tens of thousands of dollars), but some median incomes go far beyond 6.
# It is very important that you don't have too many strata and that each stratum should have a sufficient number of instances. If this is not the case, the estimate of the stratums importance may be biased and your model could think that a stratum is less important. 
# 
# The code below transforms the median_income attribute into a categorical one by dividing the median income by 1.5 to limit the number of income categories and rounds it up using "np.ceil()" to have discrete categories. It merges all the categories that are greater than 5 into category 5. The categories are represented in the histogram below the code.

# In[39]:


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[40]:


plt.hist(housing["income_cat"])
fig = plt.gcf()


# Last but not least you need to do stratified sampling based on the income categories. You can use sklearn's "StratifiedShuffleSplit" class:

# In[57]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for trainIndex, testIndex in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[trainIndex]
    strat_test_set = housing.loc[testIndex]


# Let’s see if this worked as expected. You can start by looking at the income category proportions 
# in the full housing dataset:

# In[58]:


housing["income_cat"].value_counts() / len(housing)


# In[59]:


strat_train_set.info()


# In[60]:


strat_test_set.info()


# In[62]:


# Now we want to remove the income_categoreis attribute because we don't need it anymore.
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# <a id='explo'></a>

# # CREATE A TEST SET
# 
# So far you have only taken a quick glance at the data to get a general understanding of 
# the kind of data you are manipulating. 
# Now the goal is to go a little bit more in depth.
# First, make sure you have put the test set aside and you are only exploring the training 
# set. Also, if the training set is very large, you may want to sample an exploration set, to 
# make manipulations easy and fast. 
# In our case, the set is quite small so you can just work directly on the full set. Let’s create 
# a copy so you can play with it without harming the training set:
# 

# In[63]:


housing = strat_train_set.copy()


# Let us work on the following points:
# 1. Visualizing Geographical Data
# 2. Looking for Correlations
# 3. Experimenting with Attribute Commbinations

# In[66]:


# Setting the alpha option to 0.1 makes it much easier to visualize the places where there is a high density of data points.
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# Here we can clearly see the high density areas, namely the Bay Area and around Los Angeles and San Diego, plus a long line of fairly high density in the Central Valley, in particular around Sacramento and Fresno.

# # Visualisation

# In[70]:


import seaborn as sns


# In[71]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,\
             s=housing["population"]/100, label="population",c="median_house_value", \
             cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()


# On the scatterplot we can see, that the housing prices are related to the location (close to the ocean) and to the population density, but we know that the housing prices of coastal districts are not that high in Northern California, so we can't make that rule as simple as that.  

# <a id='corre'></a>

# #  Correlation
# 
# The housing dataset isn't that large and therefore we can easily compute the correlations between every attribute using the "corr()" method. We will start by looking how much each attribute is correlated to the median house value.

# In[73]:


corr_matrix = housing.loc[ : , housing.columns!='ocean_proximity'].corr()
#the full correlation matrix
corr_matrix


# In[74]:


#Correlation for a single variable, let us use median_income 
corr_matrix['median_income'].sort_values()


# The coefficient of the correlation ranges from 1 to -1. The closer it is to 1 the more correlated it is and vice versa.  Correlations that are close to 0, means that there is no correlation, neither negative or positive. You can see that the median_income is correlated the most with the median house value. Because of that, we will generate a more detailed scatterplot below:

# # Scatter Matrix

# In[75]:


from pandas.plotting import scatter_matrix
attributes=["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# These graphs show the distribution of the feature variables along the main diagonal and the scatter plots show the relationship between pairs of vairables.
# We can even make these scatter plots separately as follows:

# In[76]:


housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)


# The scatterplot reveals, that the correlation is indeed very strong because we can clearly see an upward trend and the points are not to dispersed. We can also clearly see the price-cap, we talked about earlier, at 500 000 as a horizontal line. Other less obvious lines are around 450 000, 350 000 and 280 000. We may have to remove the corresponding districts to prevent the model from learning to reproduce these data faults. 

# # Attribute Combination

# Before we now actually prepare the data to fed it into the model, we should think about combinating a few attributes. For example, the number of rooms within a district is of course not very helpful, if you don't know how many households are within that district. You want the number of roms per household. The number of bedrooms isn't that helpful for the same reason, but it would make sense to compare it with the total number of rooms within a household. Also the population per household would be an interesting attribute. I will create these new attributes in the code below and then we will look at the correlation matrix again.

# In[77]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[79]:


corr_matrix = housing.loc[ : , housing.columns!='ocean_proximity'].corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# Now we see, that the bedrooms_per_room attribute is more correlated with the median house value than the total number of rooms or bedrooms. Houses with a lower bedroom/room ratio tend to be more expensive. The rooms_per_household attribute is also better than the total number of rooms in a district. Obviously the larger the house, the higher the price.
# 

# # Data Preparation
# 
# Now it is time to prepare the data so that our model can process it. We will write functions that do this instead of doing it manually. The reason for this is, that you can reuse these on a new dataset and at new projects you will work on. But first, let's revert to a clean training set by copying start_train_set and let's separate the predictors and the labels since we don't necessarily want to apply the same tranformations to the predictors and the target values.

# In[80]:


housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()


# We noticed earlier that the total_bedrooms attribute has some missing values. Most Machine Learning algorithms can't work with datasets that have missing values. 
# 
# There are 3 ways to solve this problem:
# 
# 1.) You could remove the whole attribute
# 
# 2.) You could get rid of the districts that contain missing values.
# 
# 3.) You could replace them with zeros, the median or the mean
# 
# We chose option 3 and will compute the median on the training set. Sklearn provides you with "Imputer" to do this. You first need to specify an Imputer instance, that specifies that you want to replace each attributes missing values with the median of that attribute. Because the median can only be computed on numerical attributes, we need to make a copy of the data without the ocean_proximity attribute that contains text and no numbers.

# In[82]:


median = housing["total_bedrooms"].median()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop('ocean_proximity', axis=1)


# Now we can fit the Imputer instance to the training data using the "fit()" method.

# In[83]:


imputer.fit(housing_num)


# Now that we have trained the Imputer we can use it to impute values.
# After imputation we get the results in the form of numpy array so we have to convert it back to Pandas dataframe.

# In[84]:


X=imputer.transform(housing_num)
housing_tr=pd.DataFrame(X, columns=housing_num.columns, index = list(housing.index.values))


# In[85]:


housing_tr.head()


# <a id='categ'></a>

# # Handling Categorical Attributes
# 
# Like I already mentioned most of the machine learning algorithms can just work with numerical data. The ocean_proximity attribute is still a categorical feature and we will convert it now. We can use Pandas' factorize() method to convert this string categorical feature to an integer categorical feature.
# 

# In[86]:


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housingCat = housing['ocean_proximity']
housingCat1hot = encoder.fit_transform(housingCat)


# In[87]:


housingCat1hot
#this is a numpy array


# # Transformation Pipelines

# In[88]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[91]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[92]:


housing_num_tr


# In[93]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[94]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('one_hot_encoder', OneHotEncoder(sparse=False)),
    ])


# In[95]:


# Now we combine the two pipelines
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[96]:


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# # Train Models
# 
# We looked at the big picture of your problem and framed it, we explored and visualized the data and preprocessed it. Now it is time to select and train a Machine Learning model. The hard part were the previous steps. What we do now, is going to be much simpler and easier.
# 
# First, let's test whether a Linear Regression model gives us a satisfying result:

# In[97]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[98]:


ex = housing.iloc[:3]
ex_labels = housing_labels.iloc[:3]
ex_data_prepared = full_pipeline.transform(ex)

print("Predictions:", lin_reg.predict(ex_data_prepared))
print("Labels:",list(ex_labels))


# Let's use [RMSE (Root Mean Squared Error)](http://www.statisticshowto.com/rmse/) to judge the quality of our predictions:

# In[99]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# This is clearly not a great score. Since most districts median_housing_values range between 120,000 and 265,000 dollar, a prediction error of $68,376 is not very satisfying and also an example of a model underfitting the data. This either means that the features do not provide  enough information to make proper predictions, or that the model is just not powerful enough.
# 
# 
# The main ways to fix underfitting are:
# 
# 1.) feed the model with better features
# 
# 2.) select a more powerful model
# 
# 3.) reduce the constraints on the model
# 
# 
# First let's try out a more powerful model since we just only tested one.

# Let's use a DecisionTreeRegressor, which can find complex nonlinear relationships in the data:

# In[100]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[102]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# This gives you no error at all, which means that we strongly overfitted our data. How can we be sure ? As we allready discussed earlier, you don't want to use the test set until you are confident about your model. But how can we test how our model performs if we can't use the test data ? One way to do this is using **K-Fold Cross-Validation**, which uses part of the training set for training and a part for validation. The following code randomly splits the training set into 10 subset called **folds**. Then it trains and evaluates 10 times, using every fold once for either training or validation:

# In[103]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# Let's look at the result:

# In[104]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# These values indicate that Regression tree is slightly overfitting the data. Let us try RandomForestRegressor

# In[105]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[106]:


housing_pred=forest_reg.predict(housing_prepared)
forest_mse=mean_squared_error(housing_labels, housing_pred)
forest_rmse=np.sqrt(forest_mse)
forest_rmse


# In[107]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,\
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse= np.sqrt(-forest_scores)
display_scores(forest_rmse)


# This is a lot better. Note that the score on the training set is still much lower, than on the validation set, which indicates that the model is overfitting the training set and that we should optimize the model to solve this problem. 

# # Fine Tuning Of Parameters
# 
# 

# In[108]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[109]:


grid_search.best_params_


# This tells us that the best solution would be by setting the max_features to 8 and the n_estimators to 30.

# # Evaluation
# 
# Now that we have our best parameter values, we can fit a final model using these.

# In[110]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[111]:


final_rmse


# We now have a final prediciton error of $47,873.

# <a id='summary'></a>

# # Summary
# 
# **Problem Framing & Data Exploration:** Learned to frame machine learning problems, analyze data through techniques like histograms, and handle data imbalances using stratified sampling.  
# **Feature Engineering:** Created new features, processed categorical data using one-hot encoding, handled missing values, and scaled features for optimal model performance.  
# **Model Building & Evaluation:** Built and evaluated various regression models (Linear Regression, Decision Tree Regressor, Random Forest Regressor) using techniques like K-fold cross-validation and grid search to combat overfitting and underfitting.  
# **Essential Tools & Techniques:** Utilized the scikit-learn library's pipeline class, employed RMSE for model evaluation, and gained insights into hyperparameter tuning.

# In[ ]:




