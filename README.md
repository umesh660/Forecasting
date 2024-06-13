# Forecasting Economic Indicators and FTSE 100: A Technical Report

## Introduction
This report forecasts key UK economic indicators and uses them to predict the FTSE 100 index until December 2024. The analysis employs exponential smoothing and ARIMA models on datasets from the UK Government Office for National Statistics (ONS).

## Data Preparation
### Data Sources
1. **Average Weekly Earnings (K54D)**
   - [ONS Link](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/datasets/averageweeklyearnings)
   - Column BX, from row 570 down.
2. **Retail Sales Index (EAFV)**
   - [ONS Link](https://www.ons.gov.uk/businessindustryandtrade/retailindustry/datasets/retailsales)
   - Column AI, from row 189 down.
3. **Index of Production (K226)**
   - [ONS Link](https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/indexofproduction)
   - Column EX, from row 975 down.
4. **Turnover and Orders (JQ2J)**
   - [ONS Link](https://www.ons.gov.uk/businessindustryandtrade/manufacturingandproductionindustry/timeseries/jq2j/ios1/previous)
   - Column B, from row 161 down.

### Processing
- Ensure all datasets are monthly and cover the same period.
- Handle missing values and normalize the data.

## Forecasting Methods
### Exponential Smoothing
- Forecast each series separately until December 2024.

### ARIMA for K54D
- Compare ARIMA and exponential smoothing for K54D.

## Multivariate Regression Model
- Use the four series to predict the FTSE 100 index.
- Align datasets and develop a regression model.

## Results
- Present forecasts for each series.
- Compare ARIMA and exponential smoothing for K54D.
- Evaluate the regression model for FTSE 100.

## Conclusion
- Summarize findings and provide recommendations.

## Appendix
### Python Code Descriptions
1. **Data Preparation**: Import and clean ONS datasets.
2. **Exponential Smoothing**: Apply to each time series.
3. **ARIMA for K54D**: Fit and forecast using ARIMA.
4. **FTSE 100 Regression**: Develop and validate regression model.

This report provides a robust analysis for predicting economic indicators and the FTSE 100, aiding Future Stocks in informed decision-making.
