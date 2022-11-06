#!/usr/bin/env python
# coding: utf-8

# # Forecasting Net Prophet
# 
# You’re a growth analyst at [MercadoLibre](http://investor.mercadolibre.com/investor-relations). With over 200 million users, MercadoLibre is the most popular e-commerce site in Latin America. You've been tasked with analyzing the company's financial and user data in clever ways to make the company grow. So, you want to find out if the ability to predict search traffic can translate into the ability to successfully trade the stock.
# 
# Instructions
# 
# This section divides the instructions for this Challenge into four steps and an optional fifth step, as follows:
# 
# * Step 1: Find unusual patterns in hourly Google search traffic
# 
# * Step 2: Mine the search traffic data for seasonality
# 
# * Step 3: Relate the search traffic to stock price patterns
# 
# * Step 4: Create a time series model with Prophet
# 
# * Step 5 (optional): Forecast revenue by using time series models
# 
# The following subsections detail these steps.
# 
# ## Step 1: Find Unusual Patterns in Hourly Google Search Traffic
# 
# The data science manager asks if the Google search traffic for the company links to any financial events at the company. Or, does the search traffic data just present random noise? To answer this question, pick out any unusual patterns in the Google search data for the company, and connect them to the corporate financial events.
# 
# To do so, complete the following steps:
# 
# 1. Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?
# 
# 2. Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?
# 
# ## Step 2: Mine the Search Traffic Data for Seasonality
# 
# Marketing realizes that they can use the hourly search data, too. If they can track and predict interest in the company and its platform for any time of day, they can focus their marketing efforts around the times that have the most traffic. This will get a greater return on investment (ROI) from their marketing budget.
# 
# To that end, you want to mine the search traffic data for predictable seasonal patterns of interest in the company. To do so, complete the following steps:
# 
# 1. Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).
# 
# 2. Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?
# 
# 3. Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?
# 
# ## Step 3: Relate the Search Traffic to Stock Price Patterns
# 
# You mention your work on the search traffic data during a meeting with people in the finance group at the company. They want to know if any relationship between the search data and the company stock price exists, and they ask if you can investigate.
# 
# To do so, complete the following steps:
# 
# 1. Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.
# 
# 2. Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (`2020-01` to `2020-06` in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?
# 
# 3. Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:
# 
#     * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility
# 
#     * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis
# 
# 4. Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?
# 
# ## Step 4: Create a Time Series Model with Prophet
# 
# Now, you need to produce a time series model that analyzes and forecasts patterns in the hourly search data. To do so, complete the following steps:
# 
# 1. Set up the Google search data for a Prophet forecasting model.
# 
# 2. After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?
# 
# 3. Plot the individual time series components of the model to answer the following questions:
# 
#     * What time of day exhibits the greatest popularity?
# 
#     * Which day of the week gets the most search traffic?
# 
#     * What's the lowest point for search traffic in the calendar year?
# 
# ## Step 5 (Optional): Forecast Revenue by Using Time Series Models
# 
# A few weeks after your initial analysis, the finance group follows up to find out if you can help them solve a different problem. Your fame as a growth analyst in the company continues to grow!
# 
# Specifically, the finance group wants a forecast of the total sales for the next quarter. This will dramatically increase their ability to plan budgets and to help guide expectations for the company investors.
# 
# To do so, complete the following steps:
# 
# 1. Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data.
# 
# 2. Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)
# 
# 3. Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.
# 

# ## Install and import the required libraries and dependencies

# In[ ]:


# Import the required libraries and dependencies
import pandas as pd
from pathlib import Path
import numpy as np
import holoviews as hv
from prophet import Prophet
import hvplot.pandas
import datetime as dt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Step 1: Find Unusual Patterns in Hourly Google Search Traffic
# 
# The data science manager asks if the Google search traffic for the company links to any financial events at the company. Or, does the search traffic data just present random noise? To answer this question, pick out any unusual patterns in the Google search data for the company, and connect them to the corporate financial events.
# 
# To do so, complete the following steps:
# 
# 1. Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?
# 
# 2. Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?
# 

# #### Step 1: Read the search data into a DataFrame, and then slice the data to just the month of May 2020. (During this month, MercadoLibre released its quarterly financial results.) Use hvPlot to visualize the results. Do any unusual patterns exist?

# In[ ]:


# Upload the "google_hourly_search_trends.csv" file into Colab, then store in a Pandas DataFrame
# Set the "Date" column as the Datetime Index.

mercado_trends_df = pd.read_csv(
    Path("./Resources/google_hourly_search_trends.csv"), 
    index_col="Date", 
    parse_dates=True, 
    infer_datetime_format=True
)

# Review the first and last five rows of the DataFrame
display(mercado_trends_df.head())
display(mercado_trends_df.tail())


# In[ ]:


# data checkpoint
display(mercado_trends_df.shape)


# In[ ]:


# Review the data types of the DataFrame using the info function
display(mercado_trends_df.info())


# In[ ]:


# Slice the DataFrame to just the month of May 2020
mercado_trends_may_2020_df =mercado_trends_df.loc['2020-05']


# In[ ]:


# data checkpoint
display(mercado_trends_may_2020_df.head())
display(mercado_trends_may_2020_df.tail())


# In[ ]:


# Use hvPlot to visualize the data for May 2020
mercado_trends_may_2020_df.hvplot(
    x="Date",
    y="Search Trends",
    title="Cercado Search Trends: May 2020"
)


# #### Step 2: Calculate the total search traffic for the month, and then compare the value to the monthly median across all months. Did the Google search traffic increase during the month that MercadoLibre released its financial results?

# In[ ]:


# Calculate the sum of the total search traffic for May 2020
mercado_trends_may_2020_sum_srs = mercado_trends_may_2020_df.sum()

# View the mercado_trends_may_2020_sum value
print(f"Total Cercado search trends for May 2020 = {mercado_trends_may_2020_sum_srs[0]}")


# In[ ]:


# Calcluate the monhtly median search traffic across all months 
# Group the DataFrame by index year and then index month, chain the sum and then the median functions


# In[ ]:


mercado_groupby_year_month_lst = [mercado_trends_df.index.year, mercado_trends_df.index.month]


# In[ ]:


# data checkpoint
display(type(mercado_groupby_year_month_lst))
display(mercado_groupby_year_month_lst)


# In[ ]:


mercado_trends_per_month_sum_df = mercado_trends_df.groupby(by=mercado_groupby_year_month_lst).sum()


# In[ ]:


# data checkpoint
display(type(mercado_trends_per_month_sum_df))
display(mercado_trends_per_month_sum_df.head(n=10))
display(mercado_trends_per_month_sum_df.tail(n=10))


# In[ ]:


mercado_trends_all_month_median_srs = mercado_trends_per_month_sum_df.median()

# View the median_monthly_traffic value
display(type(mercado_trends_all_month_median_srs))
display(mercado_trends_all_month_median_srs)


# In[ ]:


# Compare the seach traffic for the month of May 2020 to the overall monthly median value
display(mercado_trends_may_2020_sum_srs / mercado_trends_all_month_median_srs)


# ##### Answer the following question: 

# **Question:** Did the Google search traffic increase during the month that MercadoLibre released its financial results?
# 
# **Answer:** Yes, it increased (see next cell for details)

# In[ ]:


print(f"The total Cercado search trends for May 2020 = {mercado_trends_may_2020_sum_srs[0]}, "
      f"the median Cercaod search trends for all months = {mercado_trends_all_month_median_srs[0]}")
print(f"This represents an {(mercado_trends_may_2020_sum_srs[0] / mercado_trends_all_month_median_srs[0] - 1)*100:0.2f} percent change in search trends for May 2020")


# ## Step 2: Mine the Search Traffic Data for Seasonality
# 
# Marketing realizes that they can use the hourly search data, too. If they can track and predict interest in the company and its platform for any time of day, they can focus their marketing efforts around the times that have the most traffic. This will get a greater return on investment (ROI) from their marketing budget.
# 
# To that end, you want to mine the search traffic data for predictable seasonal patterns of interest in the company. To do so, complete the following steps:
# 
# 1. Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).
# 
# 2. Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?
# 
# 3. Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?
# 

# #### Step 1: Group the hourly search data to plot the average traffic by the day of the week (for example, Monday vs. Friday).

# In[ ]:


# Group the hourly search data to plot (use hvPlot) the average traffic by the day of week 
mercado_groupby_hour_dayofweek_lst = [mercado_trends_df.index.hour, mercado_trends_df.index.dayofweek]


# In[ ]:


display(mercado_groupby_hour_dayofweek_lst)


# #### Step 2: Using hvPlot, visualize this traffic as a heatmap, referencing the `index.hour` as the x-axis and the `index.dayofweek` as the y-axis. Does any day-of-week effect that you observe concentrate in just a few hours of that day?

# In[ ]:


# chain the mean of all the hours / dayofweek
mercado_trends_per_hour_dayofweek_avg_df = mercado_trends_df.groupby(by=mercado_groupby_hour_dayofweek_lst).mean()


# In[ ]:


# data checkpoint
display(type(mercado_trends_per_hour_dayofweek_avg_df))
display(mercado_trends_per_hour_dayofweek_avg_df.head(n=10))
display(mercado_trends_per_hour_dayofweek_avg_df.tail(n=10))


# In[ ]:


# Change the multi-index of (Date, Date) to (Hour, DayOfWeek)
mercado_trends_per_hour_dayofweek_avg_df.index.names = ["Hour", "DayOfWeek"]


# In[ ]:


# Data checkpoint
display(mercado_trends_per_hour_dayofweek_avg_df.index.names)
display(mercado_trends_per_hour_dayofweek_avg_df.head())


# In[ ]:


# Use hvPlot to visualize the hour of the day and day of week search traffic as a heatmap.
mercado_trends_per_hour_dayofweek_avg_df.hvplot.heatmap(
    x='Hour', 
    y='DayOfWeek',
    C='Search Trends', 
    cmap='reds', 
    title="Mercado Search Treands: DayOfWeek vs Hourly", 
    width=800, 
    height=500
)


# In[ ]:


# ALTERNATE approach for heatmap (use the heatmaps aggregate method)
mercado_trends_df.hvplot.heatmap(
    x='index.hour',
    xlabel = "Hour", 
    y='index.dayofweek',
    ylabel = "DayOfWeek", 
    C='Search Trends', 
    cmap='reds', 
    title="Mercado Search Treands: DayOfWeek vs Hourly", 
    width=800, 
    height=500
).aggregate(function=np.mean)


# ##### Answer the following question:

# **Question:** Does any day-of-week effect that you observe concentrate in just a few hours of that day?
# 
# **Answer:** It appears that the concentrations of search trends between the 22nd hour to the 1st hour of the next day (hour loops ...,21, 22, 23, 0, 1, 2, 3, ...).  I am not sure what timezone this data was collected from (given that this is a Latin America company)

# #### Step 3: Group the search data by the week of the year. Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?

# In[ ]:


# Group the hourly search data to plot (use hvPlot) the average traffic by the week of the year
mercado_groupby_weekofyear_lst = [mercado_trends_df.index.weekofyear]
mercado_groupby_weekofyear_avg_df = mercado_trends_df.groupby(by=mercado_groupby_weekofyear_lst).mean()
mercado_groupby_weekofyear_avg_df.hvplot(
    kind='bar', 
    xlabel = "Week of Year",
    ylabel = "Search Trends",
    title = "Mercado Search Treands: Avg on Week of Day",
    width=800, 
    height=400,
    rot=90)


# ##### Answer the following question:

# **Question:** Does the search traffic tend to increase during the winter holiday period (weeks 40 through 52)?
# 
# **Answer:** Zooming in on the above hvplot, it seems that between the 42nd week to the 51st week, the search trends do increase (and drop significantly on the 52 week).  Note, the absolute values of the search trends during these months are not significatnly different than the previous months from week 1 to week 40 (which is a bit surprising)

# ## Step 3: Relate the Search Traffic to Stock Price Patterns
# 
# You mention your work on the search traffic data during a meeting with people in the finance group at the company. They want to know if any relationship between the search data and the company stock price exists, and they ask if you can investigate.
# 
# To do so, complete the following steps:
# 
# 1. Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.
# 
# 2. Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (`2020-01` to `2020-06` in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?
# 
# 3. Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:
# 
#     * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility
# 
#     * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis
# 
# 4. Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?
# 

# #### Step 1: Read in and plot the stock price data. Concatenate the stock price data to the search data in a single DataFrame.

# In[ ]:


# Upload the "mercado_stock_price.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the Datetime Index.

mercado_stock_df =  pd.read_csv(
    Path("./Resources/mercado_stock_price.csv"), 
    index_col="date", 
    parse_dates=True, 
    infer_datetime_format=True
)


# View the first and last five rows of the DataFrame
display(mercado_stock_df.head())
display(mercado_stock_df.tail())


# In[ ]:


# Use hvPlot to visualize the closing price of the df_mercado_stock DataFrame
mercado_stock_df.hvplot(
    x="date",
    xlabel="Date",
    y="close",
    ylabel="Close Price",
    title="Mercado Closing Prices"
)


# In[ ]:


# data checkpoint

display(mercado_stock_df.head())
display(mercado_stock_df.tail())

display(mercado_trends_df.head())
display(mercado_trends_df.tail())


# In[ ]:


# Concatenate the mercado_stock_df DataFrame with the mercado_trends_df DataFrame
# Concatenate the DataFrame by columns (axis=1), and drop and rows with only one column of data
mercado_stock_trends_df = pd.concat([mercado_trends_df, mercado_stock_df], axis=1).dropna()

# Set the index to Date
mercado_stock_trends_df.index.rename('date', inplace=True)

# View the first and last five rows of the DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())


# #### Step 2: Market events emerged during the year of 2020 that many companies found difficult. But, after the initial shock to global financial markets, new customers and revenue increased for e-commerce platforms. Slice the data to just the first half of 2020 (`2020-01` to `2020-06` in the DataFrame), and then use hvPlot to plot the data. Do both time series indicate a common trend that’s consistent with this narrative?

# In[ ]:


# For the combined dataframe, slice to just the first half of 2020 (2020-01 through 2020-06) 
first_half_2020 = mercado_stock_trends_df['2020-01':'2020-06']

# View the first and last five rows of first_half_2020 DataFrame
display(first_half_2020.head())
display(first_half_2020.tail())


# In[ ]:


# Use hvPlot to visualize the close and Search Trends data
# Plot each column on a separate axes using the following syntax
# `hvplot(shared_axes=False, subplots=True).cols(1)`
first_half_2020.hvplot(shared_axes=False, subplots=True).cols(1)


# ##### Answer the following question:

# **Question:** Do both time series indicate a common trend that’s consistent with this narrative?
# 
# **Answer:** Both time series do not indicate a common trend.  The stock price does show that after the initial shock to global financial markets, the stock price rebounded and even surpassed the price at the beginning of the year.  However, the search trend time series remaned in the same cyclical pattern as the beginning of the year.

# #### Step 3: Create a new column in the DataFrame named “Lagged Search Trends” that offsets, or shifts, the search traffic by one hour. Create two additional columns:
# 
# * “Stock Volatility”, which holds an exponentially weighted four-hour rolling average of the company’s stock volatility
# 
# * “Hourly Stock Return”, which holds the percent change of the company's stock price on an hourly basis
# 

# In[ ]:


display(mercado_stock_trends_df.head())


# In[ ]:


# Create a new column in the mercado_stock_trends_df DataFrame called Lagged Search Trends
# This column should shift the Search Trends information by one hour
mercado_stock_trends_df['Lagged Search Trends'] = mercado_stock_trends_df['Search Trends'].shift(1)


# In[ ]:


# data checkpoint
display(mercado_stock_trends_df.head())


# In[ ]:


# Create a new column in the mercado_stock_trends_df DataFrame called Stock Volatility
# This column should calculate the standard deviation of the closing stock price return data over a 4 period rolling window
mercado_stock_trends_df['Stock Volatility'] = mercado_stock_trends_df['close'].pct_change().rolling(window=4).std()


# In[ ]:


# Use hvPlot to visualize the stock volatility
mercado_stock_trends_df['Stock Volatility'].hvplot(
    x="date",
    xlabel="Date",
    y="Stock Volatility",
    ylabel="Stock Volatility",
    title="Cercado Closing Prices"
)


# **Solution Note:** Note how volatility spiked, and tended to stay high, during the first half of 2020. This is a common characteristic of volatility in stock returns worldwide: high volatility days tend to be followed by yet more high volatility days. When it rains, it pours.

# In[ ]:


# Create a new column in the mercado_stock_trends_df DataFrame called Hourly Stock Return
# This column should calculate hourly return percentage of the closing price
mercado_stock_trends_df['Hourly Stock Return'] = mercado_stock_trends_df['close'].pct_change() 


# In[ ]:


# View the first and last five rows of the mercado_stock_trends_df DataFrame
display(mercado_stock_trends_df.head())
display(mercado_stock_trends_df.tail())


# #### Step 4: Review the time series correlation, and then answer the following question: Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?

# In[ ]:


# Construct correlation table of Stock Volatility, Lagged Search Trends, and Hourly Stock Return
mercado_stock_trends_corr_df = mercado_stock_trends_df[['Stock Volatility', 'Lagged Search Trends', 'Hourly Stock Return']].corr()

display(mercado_stock_trends_corr_df)


# ##### Answer the following question:
# 

# **Question:** Does a predictable relationship exist between the lagged search traffic and the stock volatility or between the lagged search traffic and the stock price returns?
# 
# **Answer:** There is an ~15.9% prediction percentage (negatively correlated) between Lagged Search Trends and Volatility.  The correlation is small, but it is not negligible.  There is a 1.8% prediction (postive correlation) between Lagged Search Trends and the Hourly Stock Return - so basically 0 correlation.  

# ## Step 4: Create a Time Series Model with Prophet
# 
# Now, you need to produce a time series model that analyzes and forecasts patterns in the hourly search data. To do so, complete the following steps:
# 
# 1. Set up the Google search data for a Prophet forecasting model.
# 
# 2. After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?
# 
# 3. Plot the individual time series components of the model to answer the following questions:
# 
#     * What time of day exhibits the greatest popularity?
# 
#     * Which day of the week gets the most search traffic?
# 
#     * What's the lowest point for search traffic in the calendar year?
# 

# #### Step 1: Set up the Google search data for a Prophet forecasting model.

# In[ ]:


# data checkpoint
display(mercado_trends_df.head())
mercado_trends_df.hvplot()


# In[ ]:


# Using the mercado_trends_df DataFrame, reset the index so the date information is no longer the index (moved to column)
mercado_prophet_df = mercado_trends_df.reset_index()


# In[ ]:


# data checkpoint
display(mercado_prophet_df.head())


# In[ ]:


# Label the columns ds and y so that the syntax is recognized by Prophet
mercado_prophet_df.columns = ['ds', 'y']


# In[ ]:


# data checkpoint
display(mercado_prophet_df.head())
display(mercado_prophet_df.info())


# In[ ]:


# Drop an NaN values from the prophet_df DataFrame
mercado_prophet_df = mercado_prophet_df.dropna()


# In[ ]:


# data checkpoint
display(mercado_prophet_df.info())


# In[ ]:


# View the first and last five rows of the mercado_prophet_df DataFrame
display(mercado_prophet_df.head())
display(mercado_prophet_df.tail())


# In[ ]:


# Call the Prophet function, store as an object
model_mercado_trends = Prophet()


# In[ ]:


# Fit the time-series model.
model_mercado_trends.fit(mercado_prophet_df)


# In[ ]:


# Create a future dataframe to hold predictions
# Make the prediction go out as far as 2000 hours (approx 80 days)
future_mercado_trends_df =  model_mercado_trends.make_future_dataframe(periods=2000, freq='H')

# View the last five rows of the future_mercado_trends DataFrame
display(future_mercado_trends_df.tail())


# In[ ]:


# Make the predictions for the trend data using the future_mercado_trends DataFrame
forecast_mercado_trends_df = model_mercado_trends.predict(future_mercado_trends_df)

# Display the first five rows of the forecast_mercado_trends DataFrame
display(forecast_mercado_trends_df.head())


# #### Step 2: After estimating the model, plot the forecast. How's the near-term forecast for the popularity of MercadoLibre?

# In[ ]:


# Plot the Prophet predictions for the Mercado trends data
model_mercado_trends.plot(forecast_mercado_trends_df)


# In[ ]:


figures = model_mercado_trends.plot_components(forecast_mercado_trends_df);


# ##### Answer the following question:

# **Question:**  How's the near-term forecast for the popularity of MercadoLibre?
# 
# **Answer:** The near-term forecat for the populatiry of MercadoLibre is down. This can be seen by the "trend" line in the plot above
# 

# #### Step 3: Plot the individual time series components of the model to answer the following questions:
# 
# * What time of day exhibits the greatest popularity?
# 
# * Which day of the week gets the most search traffic?
# 
# * What's the lowest point for search traffic in the calendar year?
# 

# In[ ]:


# Set the index in the forecast_mercado_trends_df DataFrame to the ds datetime column
forecast_mercado_trends_df = forecast_mercado_trends_df.set_index(['ds'])


# In[ ]:


# data checkpoint
display(forecast_mercado_trends_df.head())


# In[ ]:


# View the only the yhat,yhat_lower and yhat_upper columns from the DataFrame
display(forecast_mercado_trends_df[['yhat', 'yhat_lower', 'yhat_upper']])


# Solutions Note: `yhat` represents the most likely (average) forecast, whereas `yhat_lower` and `yhat_upper` represents the worst and best case prediction (based on what are known as 95% confidence intervals).

# In[ ]:


# From the forecast_mercado_trends_df DataFrame, use hvPlot to visualize
#  the yhat, yhat_lower, and yhat_upper columns over the last 2000 hours 
forecast_mercado_trends_df[['yhat', 'yhat_lower', 'yhat_upper']].iloc[-2000:, :].hvplot(title="Mercado Search Treands Forecast")


# In[ ]:


# data checkpoint
# plot above is too congested, zoom in to the last 200 hours
forecast_mercado_trends_df[['yhat', 'yhat_lower', 'yhat_upper']].iloc[-200:, :].hvplot()


# In[ ]:


# Reset the index in the forecast_mercado_trends_df DataFrame
forecast_mercado_trends_df = forecast_mercado_trends_df.reset_index()


# In[ ]:


# data checkpoint
display(forecast_mercado_trends_df.head())


# In[ ]:


# Use the plot_components function to visualize the forecast results 
# for the forecast_canada DataFrame 
# ???? Is this a typo in the question ???  What is forecast_canada ???
figures_mercado_trends = model_mercado_trends.plot_components(forecast_mercado_trends_df);


# ##### Answer the following questions:

# **Question:** What time of day exhibits the greatest popularity?
# 
# **Answer:** 00:00:00 (NOTE, this is relative to the given timezone the data points were provided in the csv file - the real timezone maybe different)

# **Question:** Which day of week gets the most search traffic? 
#    
# **Answer:** # Tuesday

# **Question:** What's the lowest point for search traffic in the calendar year?
# 
# **Answer:** Mid October
# 

# ## Step 5 (Optional): Forecast Revenue by Using Time Series Models
# 
# A few weeks after your initial analysis, the finance group follows up to find out if you can help them solve a different problem. Your fame as a growth analyst in the company continues to grow!
# 
# Specifically, the finance group wants a forecast of the total sales for the next quarter. This will dramatically increase their ability to plan budgets and to help guide expectations for the company investors.
# 
# To do so, complete the following steps:
# 
# 1. Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data. The daily sales figures are quoted in millions of USD dollars.
# 
# 2. Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)
# 
# 3. Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.
# 
# 
# 

# #### Step 1: Read in the daily historical sales (that is, revenue) figures, and then apply a Prophet model to the data.

# In[ ]:


# Upload the "mercado_daily_revenue.csv" file into Colab, then store in a Pandas DataFrame
# Set the "date" column as the DatetimeIndex
# Sales are quoted in millions of US dollars

mercado_sales_df = pd.read_csv(
    Path("./Resources/mercado_daily_revenue.csv"), 
    index_col="date", 
    parse_dates=True, 
    infer_datetime_format=True
)

# Review the DataFrame
display(mercado_sales_df.head())
display(mercado_sales_df.tail())


# In[ ]:


# data checkpoint
display(mercado_sales_df.shape)


# In[ ]:


# Review the data types of the DataFrame using the info function
display(mercado_sales_df.info())


# In[ ]:


# Use hvPlot to visualize the daily sales figures 
mercado_sales_df.hvplot(
    x="date",
    y="Daily Sales",
    title="Cercado Daily Revenue"
)


# In[ ]:


# Apply a Facebook Prophet model to the data.

# Set up the dataframe in the neccessary format:
# Reset the index so that date becomes a column in the DataFrame
mercado_sales_prophet_df = mercado_sales_df.reset_index()


# In[ ]:


# data checkpoint
display(mercado_sales_prophet_df.head())


# In[ ]:


# Adjust the columns names to the Prophet syntax
mercado_sales_prophet_df.columns = ['ds', 'y']


# In[ ]:


# data checkpoint
display(mercado_sales_prophet_df.head())
display(mercado_sales_prophet_df.tail())
display(mercado_sales_prophet_df.info())
# No NaN's to drop...


# In[ ]:


# Visualize the DataFrame
mercado_sales_prophet_df.hvplot(
    x="ds",
    xlabel="date (ds)",
    y="y",
    ylabel="Daily Salses (y)",
    title="Cercado Daily Revenue (Adjusted DataFrame for Prophet)"
)


# In[ ]:


# Create the model
mercado_sales_prophet_model = Prophet()

# Fit the model
mercado_sales_prophet_model.fit(mercado_sales_prophet_df)


# In[ ]:


# Predict sales for 90 days (1 quarter) out into the future.

# Start by making a future dataframe
mercado_sales_prophet_future_df = mercado_sales_prophet_model.make_future_dataframe(periods=90, freq='D')

# Display the last five rows of the future DataFrame
display(mercado_sales_prophet_future_df.tail())


# In[ ]:


# Make predictions for the sales each day over the next quarter
mercado_sales_prophet_forecast_df = mercado_sales_prophet_model.predict(mercado_sales_prophet_future_df)


# In[ ]:


# Display the first 5 rows of the resulting DataFrame
display(mercado_sales_prophet_forecast_df.head())
display(mercado_sales_prophet_forecast_df.tail())


# #### Step 2: Interpret the model output to identify any seasonal patterns in the company's revenue. For example, what are the peak revenue days? (Mondays? Fridays? Something else?)

# In[ ]:


# Use the plot_components function to analyze seasonal patterns in the company's revenue
figures = mercado_sales_prophet_model.plot_components(mercado_sales_prophet_forecast_df)


# ##### Answer the following question:

# **Question:** For example, what are the peak revenue days? (Mondays? Fridays? Something else?)
# 
# **Answer:** Highest revenue days are from Monday to Wednesday (with Wednesday being the peak day)

# #### Step 3: Produce a sales forecast for the finance group. Give them a number for the expected total sales in the next quarter. Include the best- and worst-case scenarios to help them make better plans.

# In[ ]:


# Plot the predictions for the Mercado sales
mercado_sales_prophet_model.plot(mercado_sales_prophet_forecast_df)


# In[ ]:


# For the mercado_sales_prophet_forecast DataFrame, set the ds column as the DataFrame Index
mercado_sales_prophet_forecast_df = mercado_sales_prophet_forecast_df.set_index(['ds'])

# Display the first and last five rows of the DataFrame
display(mercado_sales_prophet_forecast_df.head())
display(mercado_sales_prophet_forecast_df.tail())


# In[ ]:


# data checkpoint
display(mercado_sales_prophet_forecast_df.shape)


# In[ ]:


# Produce a sales forecast for the finance division
# giving them a number for expected total sales next quarter.
# Provide best case (yhat_upper), worst case (yhat_lower), and most likely (yhat) scenarios.

# Create a forecast_quarter Dataframe for the period 2020-07-01 to 2020-09-30
# The DataFrame should include the columns yhat_upper, yhat_lower, and yhat

#############################################################################################
# ??? I am assuming the dates provided above, 2020-07-01 to 2020-09-30, is a typo ???
# The csv dates provided (500 points) range from 1/1/2019 to 5/14/2020
# So the next quarter (90) days ends on 8/12/2020 (from 1/1/20109 to 8/12/2020 is 590 points)
# So I will change this quesiton to represent quarter dates: 2020-05-15 to 2020-08-12
#############################################################################################

mercado_sales_forecast_quarter_df = mercado_sales_prophet_forecast_df[['yhat', 'yhat_lower', 'yhat_upper']].iloc[-90:, :]


# In[ ]:


# data checkpoint
display(mercado_sales_forecast_quarter_df.head())
display(mercado_sales_forecast_quarter_df.tail())


# In[ ]:


# Update the column names for the forecast_quarter DataFrame
# to match what the finance division is looking for 
mercado_sales_forecast_quarter_df = mercado_sales_forecast_quarter_df.rename(
    columns={
        'yhat_upper': 'Best Case',
        'yhat_lower':'Worst Case', 
        'yhat':'Most Likely Case'
    }
)

# Review the last five rows of the DataFrame
display(mercado_sales_forecast_quarter_df.tail())


# In[ ]:


# Data checkpoint
mercado_sales_forecast_quarter_df.hvplot(
    xlabel="Daily Sales",
    ylabel="Date",
    title="Mercado Quarter Sales Forecast (2020-05-15 to 2020-08-12)"
)


# In[ ]:


# Displayed the summed values for all the rows in the forecast_quarter DataFrame
display(mercado_sales_forecast_quarter_df.sum())  # 3 sum values for each case (best, worsk, most likely)


# ### Based on the forecast information generated above, produce a sales forecast for the finance division, giving them a number for expected total sales next quarter. Include best and worst case scenarios, to better help the finance team plan.
# 
# **Answer:** The expected (most likely case) for Mercado's sales over the next quarter (2020-05-15 to 2020-08-12) shoule be approximately 1.945 billion.  The upper and lower bounds of the quarter sales will be between 1.773 billion (worst-case) and 2.116 billion (best-case).

# In[ ]:




