#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt


# ## Preliminary Analysis

# ###  This code visualizes the trend of average weekly earnings over time using a line plot, providing insights into the earnings pattern over the given time period.

# In[14]:


file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)
# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Earning'], marker='o', color='red')
plt.title('Average Weekly Earnings Over Time')
plt.xlabel('Date')
plt.ylabel('Earning')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# ## "Monthly and Yearly Average Trends in Average Weekly Earnings

# ### This code calculates and displays both monthly and yearly averages of the provided average weekly earning data, offering a comprehensive summary of the data distribution over time.

# In[15]:


file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)
# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Aggregate data by month
monthly_data = data.resample('M', on='Date').mean()

# Calculate yearly averages
yearly_averages = monthly_data.resample('Y').mean()

# Display the monthly data and yearly averages in a tabular format
print("Monthly Averages:")
print(monthly_data)
print("\nYearly Averages:")
print(yearly_averages)


# ### Visual representation of the trend and seasonality components in the 'Earning' time series data using moving averages.

# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Create empty Series for MA7
MA7 = pd.Series(index=data.index)

# Fill series with MA7
for i in np.arange(3, len(data) - 3):
    MA7[i] = np.mean(data['Earning'][(i-3):(i+4)])

# Create empty Series for MA2x12
MA2x12 = pd.Series(index=data.index)

# Fill series with MA2x12
for i in np.arange(6, len(data) - 6):
    MA2x12[i] = np.sum(data['Earning'][(i-6):(i+7)] * np.concatenate([[1/24], np.repeat(1/12, 11), [1/24]]))

# Plot original time series
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Earning'], color='pink', label='Original')
plt.plot(MA7.index, MA7, color='red', label='Seasonality')
plt.plot(MA2x12.index, MA2x12, color='green', label='Trend')
plt.title('Moving Averages')
plt.xlabel('Date')
plt.ylabel('Earning')
plt.legend()
plt.grid(True)
plt.show()


# ## Time Series Decomposition of Average Weekly Earnings

# ### This code decomposes the time series of average weekly earnings into trend, seasonal, and residual components using the additive model and visualizes each component separately.

# In[17]:


import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)
# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(data['Earning'], model='additive')

# Plot the decomposed components
plt.figure(figsize=(10, 8))
plt.subplot(4, 1, 1)
plt.plot(data.index, data['Earning'], label='Original')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(data.index, decomposition.trend, label='Trend')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(data.index, decomposition.seasonal, label='Seasonal')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(data.index, decomposition.resid, label='Residual')
plt.legend()

plt.tight_layout()
plt.show()


# ## Coorelation

# ### Correlation matrix

# ### This code snippet generates a scatter matrix plot for all numeric variables in the dataset, computes the correlation matrix for these variables, and prints it, providing insights into their pairwise relationships and correlations.

# In[18]:


# Read the data
data = pd.read_excel("K54Ddata_34884157.xlsx")

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Calculate correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Correlation Coefficient')
plt.title('Correlation Matrix')
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.tight_layout()
plt.show()


# Correlation matrix for all numeric variables
CorrelationMatrix = data.corr()
print(CorrelationMatrix)


# ## Scatter plot

# ###  This code generates a scatter plot showing the relationship between earnings and years, providing a visual representation of how earnings have changed over time.

# In[19]:


file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)
# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')
# Extract year from 'Date' column
data['Year'] = data['Date'].dt.year

# Plot scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data['Year'], data['Earning'], color='blue', alpha=0.5)
plt.title('Scatter Plot of Earning Over Time')
plt.xlabel('Year')
plt.ylabel('Earning')
plt.grid(True)
plt.tight_layout()
plt.show()


# ## Autocorrelation

# 
# ### The code plots the Autocorrelation Function (ACF) for the 'Earning' data, showing the correlation between each observation and its lagged values up to 50 lags.

# In[20]:


from statsmodels.graphics.tsaplots import plot_acf
file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(data['Earning'], lags=50, ax=plt.gca(), color = "green")
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.grid(True)
plt.show()


# ## Model

# ### This code fits Holt-Winters exponential smoothing models (both additive and multiplicative) to the average weekly earning data and visualizes the actual data along with the fitted forecasts from both models.

# In[21]:


from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
file_path = 'K54Ddata_34884157.xlsx'
data = pd.read_excel(file_path)

# Convert 'Date' column to datetime format and set as index
data['Date'] = pd.to_datetime(data['Date'], format='%Y %b')
data.set_index('Date', inplace=True)

# Fit Holt-Winters model with additive seasonality
fit1 = ExponentialSmoothing(data['Earning'], seasonal='add').fit()

# Fit Holt-Winters model with multiplicative seasonality
fit2 = ExponentialSmoothing(data['Earning'], seasonal='mul').fit()

# Forecasting for the next 1 year (assuming monthly data)
forecast_additive = fit1.forecast(12)
forecast_multiplicative = fit2.forecast(12)



# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(data.index, fit1.fittedvalues, label='HW Additive Forecast', linestyle='--', color='green')
plt.plot(data.index, fit2.fittedvalues, label='HW Multiplicative Forecast', linestyle='-.', color='orange')
plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast_additive, label='HW Additive Forecast (Next 1 Year)', linestyle='--', color='blue')
plt.plot(pd.date_range(start=data.index[-1], periods=13, freq='M')[1:], forecast_multiplicative, label='HW Multiplicative Forecast (Next 1 Year)', linestyle='-.', color='red')
plt.xlabel('Date')
plt.ylabel('Earning')
plt.title('Average Weekly Earnings')
plt.legend()
plt.show()


# 
# ### This code extracts and prints the smoothing parameters (alpha and gamma) for both the additive and multiplicative Holt-Winters models.

# In[22]:


# Parameters for the additive model
alpha_add = fit1.params['smoothing_level']
gamma_add = fit1.params['smoothing_seasonal']

# Parameters for the multiplicative model
alpha_mul = fit2.params['smoothing_level']
gamma_mul = fit2.params['smoothing_seasonal']

print("Additive Model Parameters:")
print("Alpha:", alpha_add)
print("Gamma:", gamma_add)

print("\nMultiplicative Model Parameters:")
print("Alpha:", alpha_mul)
print("Gamma:", gamma_mul)


# 
# ### The code calculates the root mean squared error (RMSE) for both the additive and multiplicative Holt-Winters models to assess their forecasting performance.

# In[23]:


from statsmodels.tools.eval_measures import rmse

# Calculate RMSE for the additive model
rmse_additive = rmse(data['Earning'], fit1.fittedvalues)

# Calculate RMSE for the multiplicative model
rmse_multiplicative = rmse(data['Earning'], fit2.fittedvalues)

print("RMSE for Additive Model:", rmse_additive)
print("RMSE for Multiplicative Model:", rmse_multiplicative)


# ### The code generates forecasts until December 2024 using the multiplicative Holt-Winters model and prints the forecasted values in a DataFrame.

# In[24]:


# Generate forecasts until December 2024
forecast_index = pd.date_range(start=data.index[-1], periods=12, freq='W')
forecast = fit2.forecast(len(forecast_index))  # Using the multiplicative model for forecasting

# Print the forecasted values till December 2024
forecast_df = pd.DataFrame({'Forecast': forecast})
print("Forecasted Earnings till December 2024 (Multiplicative Method):")
print(forecast_df)


# # 

# In[ ]:





# In[ ]:





# In[ ]:




