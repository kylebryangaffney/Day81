import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


pd.options.display.float_format = '{:,.2f}'.format
data = pd.read_csv('boston.csv', index_col=0)

## What is the shape of `data`? 
print(data.shape)
## How many rows and columns does it have?
print(data.describe())
## What are the column names?
print(data.columns)
## Are there any NaN values or duplicates?
print(data.duplicated().any())
print(data.isna().any())

## How many students are there per teacher on average?
print(data["PTRATIO"].mean())
## What is the average price of a home in the dataset?
print(data["PRICE"].mean() * 1000)

## What is the CHAS feature?
## the chas is a boolean describing if the residence is on the Charles river or not
## What are the minimum and the maximum value of the CHAS and why?
## the values are either 0 or 1 because it is a binary function

## What is the maximum and the minimum number of rooms per dwelling in the dataset?
print(data["RM"].max()) ## == 8.78 which actually means the most rooms is 9
print(data["RM"].min()) ## == 3.561 which actually means the minimum rooms is 3

## make seaborn box plot of price
sns.displot(data['PRICE'], bins=50, aspect=2,kde=True, color='blue')
## update the labels
plt.title(f'Averge Cost of Boston Homes in 1970s: ${(data["PRICE"].mean() * 1000):.7}')
plt.xlabel('Average Cost')
plt.ylabel('Amount of Homes')
plt.show()

## make seaborn box plot of number of rooms per unit
sns.displot(data['RM'], bins=50, aspect=2,kde=True, color='green')
## update the labels
plt.title(f'Averge number of rooms per Boston Home in 1970s: {(data["RM"].mean()):.3}')
plt.xlabel('Average Room amount')
plt.ylabel('Amount of Homes')
plt.show()

## make a box plot of DIS: the weighted distance to the 5 Boston employment centres--the estimated commute
sns.displot(data['DIS'], bins=50, aspect=2,kde=True, color='red')
## update the labels
plt.title(f'Averge weighted distance to the 5 Boston employment centres--the estimated commute in 1970s: {(data["DIS"].mean()):.3}')
plt.xlabel('Average Commute')
plt.ylabel('Amount of Homes')
plt.show()

## box plot of RAD: the index of accessibility to highways. 
sns.displot(data['RAD'], bins=50, aspect=2,kde=True, color='yellow')
## update the labels
plt.title(f'Averge Accesability of Highways from homes in Boston during 1970s: {(data["RAD"].mean()):.3}')
plt.xlabel('Average Highway Accessability')
plt.ylabel('Amount of Homes')
plt.show()

## bar graph describing how many homes are on the charles river
charles_homes = data["CHAS"].value_counts()
charles_bar = px.bar(x=["NO", "YES"], 
                     y=charles_homes.values, 
                     color=charles_homes.values, 
                     color_continuous_scale=px.colors.sequential.Plasma, 
                     title="Boston Homes on The Charles River in 1970s")

charles_bar.update_layout(xaxis_title="Homes With Charles River Access",
                          yaxis_title="Amount of Homes",
                          coloraxis_showscale=False)
charles_bar.show()


## build a pair plot on the original data
sns.pairplot(data)
## add in regression line to show best fit between the amount of homes and other variable
sns.pairplot(data, kind='reg', plot_kws={'line_kws':{'color': 'red'}})
plt.show()

## build join plots to compare different columns
## DIS and NOX
with sns.axes_style('darkgrid'):
  sns.jointplot(x=data['DIS'], 
                y=data['NOX'], 
                height=8, 
                kind='scatter',
                color='red', 
                joint_kws={'alpha':0.8})

plt.show()
## INDUS vs NOX
with sns.axes_style('darkgrid'):
  sns.jointplot(x=data['INDUS'], 
                y=data['NOX'], 
                height=8, 
                kind='scatter',
                color='blue', 
                joint_kws={'alpha':0.8})

plt.show()
## LSTAT vs RM
with sns.axes_style('darkgrid'):
  sns.jointplot(x=data['LSTAT'], 
                y=data['RM'], 
                height=8, 
                kind='scatter',
                color='green', 
                joint_kws={'alpha':0.8})

plt.show()

## LSTAT vs PRICE
with sns.axes_style('darkgrid'):
  sns.jointplot(x=data['LSTAT'], 
                y=data['PRICE'], 
                height=8, 
                kind='scatter',
                color='yellow', 
                joint_kws={'alpha':0.8})
plt.show()

## RM vs PRICE
with sns.axes_style('darkgrid'):
  sns.jointplot(x=data['RM'], 
                y=data['PRICE'], 
                height=8, 
                kind='scatter',
                color='purple', 
                joint_kws={'alpha':0.8})
plt.show()


## split training from test data based on the price variable
target = data['PRICE']
features = data.drop('PRICE', axis=1)
## provide the input of the price data column and everything else
## output training and test data split in an 80-20 test_size = 0.2
## set random state to ensure the same split happens each time the program is ran
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)


## determine how much of the data is used for training
train_data_pct = len(x_train) / len(features) * 100
print(f'Training data = {train_data_pct:.3}% of total data.')

## determine how much of the data will be used for testing
test_data_pct = len(x_test) / len(features) * 100
print(f'Test data = {test_data_pct:0.3}% of total data.')

## Use sklearn to run the regression on the training data Determine how well other data points can predict price
regression = LinearRegression()
regression.fit(x_train, y_train)
rsquared = regression.score(x_train, y_train)

print(f'Training data r-squared: {rsquared:.2}')

## Evaluate the Coefficients of the Model
regr_coef = pd.DataFrame(data=regression.coef_, index=x_train.columns, columns=['Coefficient'])
regr_coef

## Is RM's coefficient also positive?
## yes 3.11

## What is the sign on the LSAT coefficient? 
## -0.58
## Does it match your intuition and the scatter plot above?
## i thought it would be a stronger correlation 
## Based on the coefficients, how much more expensive is a room with 6 rooms compared 
## to a room with 5 rooms? According to the model, what is the 
## premium you would have to pay for an extra room? 

extra_room_cost = regr_coef.loc["RM"].values[0] * 1000 ## ~ $3,110
print(f"An extra room adds: ${extra_room_cost:.6} to the value")

## Analyse the Estimated Values & Regression Residuals
## find the residuals or the difference in predicted vs actual
predicted_values = regression.predict(x_train)
residuals = (y_train - predicted_values)

## build scatter plots of actual when compared to predicted
# Original Regression of Actual vs. Predicted Prices
plt.figure(dpi=100)
plt.scatter(x=y_train, y=predicted_values, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.title(f'Actual and Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
plt.xlabel('Actual prices 1000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 1000s $\hat y _i$', fontsize=14)
plt.show()

# Residuals and Predicted values
plt.figure(dpi=100)
plt.scatter(x=predicted_values, y=residuals, c='indigo', alpha=0.6)
plt.title('Residuals vs Predicted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()

## Calculate the mean and the skewness of the residuals. 
residuals_mean = round(residuals.mean(), 2)
residuals_skew = round(residuals.skew(), 2)

## Again, use Seaborn's `.displot()` to create a histogram and superimpose the Kernel Density Estimate (KDE)
sns.displot(residuals, kde=True, color='indigo')

plt.title(f'Residuals Skew ({residuals_skew}) Mean ({residuals_mean})')
plt.show()
## Is the skewness different from zero? If so, by how much? yes 1.46
## Is the mean different from zero? no mean is 0

## make a graph of the price data using skew instead of linear regression
target_skew = data['PRICE'].skew()
sns.displot(data['PRICE'], kde='kde', color='purple')
plt.title(f'Normal Prices. Skew is {target_skew:.3}')
plt.show()

## make another graph showing how running the log function on the y axis modifies the model
y_log = np.log(data['PRICE'])
sns.displot(y_log, kde=True)
plt.title(f'Log Prices. Skew is {y_log.skew():.3}')
plt.show()

## build a new graph but run the log function on the price data to compress the values
plt.figure(dpi=150)
plt.scatter(data["PRICE"], np.log(data["PRICE"]))
plt.title('Mapping the Original Price to a Log Price')
plt.ylabel('Log Price')
plt.xlabel('Actual Price in $1000s')
plt.show()

## make a new data set based on the log prices, not the exact
log_target = np.log(data['PRICE']) 
log_features = data.drop('PRICE', axis=1)

log_x_train, log_x_test, log_y_train, log_y_test = train_test_split(log_features, log_target, 
                                                                    test_size=0.2, random_state=10)

log_regression = LinearRegression()
log_regression.fit(log_x_train, log_y_train)
log_rsquared = log_regression.score(log_x_train, log_y_train)

log_predictions = log_regression.predict(log_x_train)
log_residuals = (log_y_train - log_predictions)

print(f'Training log(data[Prices]) r-squared: {log_rsquared:.2}')

## display the coefficients of the new model
log_df_coef = pd.DataFrame(data=log_regression.coef_, index=log_x_train.columns, columns=['coef'])
log_df_coef

## Graph of Actual vs. Predicted Log Prices
plt.scatter(x=log_y_train, y=log_predictions, c='navy', alpha=0.6)
plt.plot(log_y_train, log_y_train, color='cyan')
plt.title(f'Actual vs Predicted Log Prices: $y _i$ vs $\hat y_i$ (R-Squared {log_rsquared:.2})', fontsize=17)
plt.xlabel('Actual Log Prices $y _i$', fontsize=14)
plt.ylabel('Prediced Log Prices $\hat y _i$', fontsize=14)
plt.show()

# Original Regression of Actual vs. Predicted Prices
plt.scatter(x=y_train, y=predicted_values, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.title(f'Original Actual vs Predicted Prices: $y _i$ vs $\hat y_i$ (R-Squared {rsquared:.3})', fontsize=17)
plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
plt.show()

# Residuals vs Predicted values (Log prices)
plt.scatter(x=log_predictions, y=log_residuals, c='navy', alpha=0.6)
plt.title('Residuals vs Fitted Values for Log Prices', fontsize=17)
plt.xlabel('Predicted Log Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()

# Residuals vs Predicted values
plt.scatter(x=predicted_values, y=residuals, c='indigo', alpha=0.6)
plt.title('Original Residuals vs Fitted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()

## Calculate the mean and the skew for the residuals using log prices. 
# show the Residuals of log prices - if it makes sense
log_resid_mean = round(log_residuals.mean(), 2)
log_resid_skew = round(log_residuals.skew(), 2)

sns.displot(log_residuals, kde=True, color='navy')
plt.title(f'Log price model: Residuals Skew ({log_resid_skew}) Mean ({log_resid_mean})')
plt.show()

sns.displot(residuals, kde=True, color='indigo')
plt.title(f'Original model: Residuals Skew ({residuals_skew}) Mean ({residuals_mean})')
plt.show()
## Are the mean and skew closer to 0 for the regression using log prices? skew is now 0.9, which is an improvement from 1.46


## comparing the models on the test data to determine accuracy
print(f'Original Model Test Data r-squared: {regression.score(x_test, y_test):.2}')
print(f'Log Model Test Data r-squared: {log_regression.score(log_x_test, log_y_test):.2}')

## Predict a Property's Value using the Regression Coefficients
features = data.drop(['PRICE'], axis=1)
average_vals = features.mean().values
property_stats = pd.DataFrame(data=average_vals.reshape(1, len(features.columns)), 
                              columns=features.columns)
property_stats

## Predict how much the average property is worth using the stats above. What is the log price estimate and what is the dollar estimate?
## Make prediction
log_estimate = log_regression.predict(property_stats)[0]
print(f'The log price estimate is ${log_estimate:.3}')

## Convert Log Prices to Acutal Dollar Values
dollar_est = np.exp(log_estimate) * 1000
print(f'The property is estimated to be worth ${dollar_est:.6}')

## predict cost of property 
# Define Property Characteristics
next_to_river = True
nr_rooms = 8
students_per_classroom = 20 
distance_to_town = 5
pollution = data["NOX"].quantile(q=0.75) # high
amount_of_poverty =  data["LSTAT"].quantile(q=0.25) # low

property_stats['RM'] = nr_rooms
property_stats['PTRATIO'] = students_per_classroom
property_stats['DIS'] = distance_to_town

if next_to_river:
    property_stats['CHAS'] = 1
else:
    property_stats['CHAS'] = 0

property_stats['NOX'] = pollution
property_stats['LSTAT'] = amount_of_poverty

log_estimate = log_regression.predict(property_stats)[0]
print(f'The log price estimate is ${log_estimate:.3}')

# Convert Log Prices to Acutal Dollar Values
dollar_est = np.exp(log_estimate) * 1000
print(f'The property is estimated to be worth ${dollar_est:.6}')