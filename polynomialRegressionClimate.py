# https://towardsdatascience.com/machine-learning-polynomial-regression-with-python-5328e4e8a386
# https://towardsdatascience.com/time-series-analysis-and-climate-change-7bb4371021e

# Import the Pandas library
import pandas as pd
import numpy as np
# Import Matplotlib
import matplotlib.pyplot as plt
import io, requests
import calendar
from datetime import datetime
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Fitting Linear Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Function definition
def populate_df_with_anomolies_from_row(row):
	year = row['Year']
	# Anomaly values (they seem to be a mixture of strings and floats)
    # data.iloc[<row_selection>,<column_selection>]
	monthly_anomolies = row.iloc[1:]
	# Abbreviated month names (index names)
	months = monthly_anomolies.index
	for month in monthly_anomolies.index:
		# Get the last day for each month 
		last_day = calendar.monthrange(year,datetime.strptime(month, '%b').month)[1]
		# construct the index with which we can reference our new DataFrame (to populate) 
		date_index = datetime.strptime(f'{year} {month} {last_day}', '%Y %b %d')
		# Populate / set value @ above index, to anomaly value
		t.loc[date_index] = monthly_anomolies[month]
		
def clean_anomaly_value(raw_value):
	try:
		return float(raw_value)
	except:
		return np.NaN

def viz_polymonial(X,y):
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Climate (Polynomial Regression)')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.show()
    return
    
def addTestValues(X, y):
    numero = len(X)
    ultimo = X[numero-1][0]
    result = np.arange('2021-01', '2040-12', dtype='datetime64[M]')
    contador = 0
    for i in result:
        if contador > 0:
            X_new = np.datetime64(i, 'D') - np.timedelta64(1, 'D')
            arrayX = np.array([X_new])
            X = np.append(X, arrayX).reshape(len(X)+1,1)
            y = np.append(y, np.empty([1,1], dtype=object)) 
        contador = contador + 1
    viz_polymonial(X, y)

    
def getX(temp_X):
    result = []
    for i in temp_X:
        temp = []
        temp.append(i)
        result.append(temp)
    return result

# Read in the raw temperature and emissions datasets (they are in CSV format)
url = 'https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv'
s = requests.get(url).content
raw_t = pd.read_csv(io.StringIO(s.decode('utf-8')), skiprows=1)
#raw_t = pd.read_csv('./data/GLB.Ts+dSST.csv')

# Create new dataframe with an index for each month
# Last day of each month from 31/01/1880 to 31/12/2018
date_rng = pd.date_range(start='1/1/1880', end='1/03/2019', freq='M')

# Next create the empty DataFrame, which we will populate using the actual data
# each line contains an element of date_rng
# index increment from 1
t = pd.DataFrame(date_rng, columns=['date'])

# Create a column for the anomoly values
# index increment from 1
t['Avg_Anomaly_deg_C'] = None

# Set the index of the DataFrame to the date column (DateTime index)
t.set_index('date', inplace=True)

# data.iloc[<row_selection>,<column_selection>]
#,:13 - da primeira a 13a coluna
# https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
raw_t = raw_t.iloc[:,:13]

# Apply function to each row of raw data 
_ = raw_t.apply(lambda row: populate_df_with_anomolies_from_row(row), axis=1)

# Apply above function to all anomaly values in DataFrame
t['Avg_Anomaly_deg_C'] = t['Avg_Anomaly_deg_C'].apply(lambda raw_value: clean_anomaly_value(raw_value))

# 'Forward fill' to take care of NaN values
t.fillna(method='ffill', inplace=True)

y=np.array(t['Avg_Anomaly_deg_C'].values, dtype=float)
temp=np.array(pd.to_datetime(t['Avg_Anomaly_deg_C'], format='%Y%m%d', errors='ignore').index.values)
tempX = getX(list(temp))
X=np.array(tempX)

poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=0)
lin_reg = LinearRegression().fit(X_train, y_train)

pred_train = lin_reg.predict(X_train)
pred_test = lin_reg.predict(X_test)
print('Root mean squared error (train): %.2f' % np.sqrt(mean_squared_error(y_train, pred_train)))
print('Coefficient of determination (train): %.2f (1 is perfect) ' % r2_score(y_train, pred_train))
print('Root mean squared error (test): %.2f' % np.sqrt(mean_squared_error(y_test, pred_test)))
print('Coefficient of determination (test): %.2f (1 is perfect) ' % r2_score(y_test, pred_test))
print('R-squared score (training): ', lin_reg.score(X_train, y_train))
print('R-squared score (test): ', lin_reg.score(X_test, y_test))

addTestValues(X,y)