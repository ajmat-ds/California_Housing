# California_Housing

## Libraries required
` Pandas
Numpy
Matplotlib
Seaborn
os
tarfile
urllib
SimpleImputer
 LabelBinarizer
 BaseEstimator, TransformerMixin
 Pipeline
StandardScaler
OneHotEncoder
FeatureUnion`
"""
## Machine Learning Model used in this project
"""
LinearRegression
DecisionTreeRegressor
RandomForestRegressor

"""

## Take a Look At The Dataset
`longitude,	latitude,	housing_median_age,	total_rooms,	total_bedrooms,	population,	households,	median_income,	median_house_value,	ocean_proximity`
'Data columns (total 10 columns)'

## Train/Test Split
--from sklearn.model_selection import train_test_split
## Visualizing Geographical Data
"""
Here we can clearly see the high density areas, namely the Bay Area and around Los Angeles and San Diego, 
plus a long line of fairly high density in the Central Valley, in particular around Sacramento and Fresno."""

"""
On the scatterplot we can see, that the housing prices are related to the location (close to the ocean) and to the population density,
but we know that the housing prices of coastal districts are not that high in Northern California, so we can't make that rule as simple as that.
"""
## Looking for Correlations
"""
The coefficient of the correlation ranges from 1 to -1. The closer it is to 1 the more correlated it is and vice versa. 
Correlations that are close to 0, means that there is no correlation, neither negative or positive.
You can see that the median_income is correlated the most with the median house value. 
"""
## Experimenting with Attribute Commbinations
## Data Preparation
## Handling Categorical Attributes
"""
Like I already mentioned most of the machine learning algorithms can just work with numerical data. 
The ocean_proximity attribute is still a categorical feature and we will convert it now. 
We can use Pandas' factorize() method to convert this string categorical feature to an integer categorical feature.
"""
## Transformation Pipelines
## Train Models
## Fine Tuning Of Parameters
## Finally Evaluation








