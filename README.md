# traffic-flow-volume-prediction


# Problem Definition

**Problem:** I approach real-time traffic volume prediction as a multivariate time-series problem, where my model estimates future traffic flow at time steps Tt+1 to Tt+f based on t-hours of historical observations (from Tt-l to Tt), with f being the prediction horizon and l being the length of past observations.

**Objective:** To develop an efficient and accurate traffic flow volume prediction system, I will conduct a comprehensive analysis and comparison of Machine Learning models (Linear Regression, SVM, Random Forest Regression, XGBoost) and Neural Networks (GRU, LSTM, BI-LSTM, CNN-LSTM). This project will involve rigorous data preprocessing, feature engineering, hyperparameter tuning, and exploratory data analysis to optimize the performance of ML models. The performance of each model will be assessed using the following metrics: MAE, RMSE, MAPE, and R2 scores.

# Dataset Information

# The Metro Interstate Traffic Volume dataset consists of 48,204 records collected from sensors installed at the MN DoT ATR station 301 on the Hourly Interstate 94 Westbound, located between Minneapolis and St Paul, MN. The dataset includes information on hourly traffic volume, weather conditions, and holidays to analyze their impact on traffic flow. Attributes featured in the dataset are holiday, temperature (in Kelvin), rainfall (mm/hour), snowfall (mm/hour), cloud cover percentage, weather descriptions (short and long), date and time (in local CST), and the hourly traffic volume on I-94 ATR 301 westbound.

Link: [https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume\#](https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume)

# Data Analysis

1.  **Data cleaning**: Removed duplicates based on the 'date_time' attribute and outliers present in 'temp' and 'rain_1h' attributes using cutoff for outliers as 1.5 times the IQR. Resulted in the removal of 7629 records.

    Total records remaining = 40575.

2.  **Feature Engineering and Data augmentation**

    -   Extracted time-related features (year, month, day, hour, weekday, date) from the date_time column.

    -   Binarized holiday, rain, and snow attributes into holiday_bin, rain_bin, and snow_bin.

    -   Created categorical attributes for season, day_of_week, and hour to find patterns within several months, days in a week, and hours of a day.

    -   Generated 1 to 6-hour lagged features for rain, snow, temp, and cloud cover percentage (clouds_all) representing the effect of these environmental factors on traffic volume during the next few hours after occurrence.

    -   Encoded cyclical dependencies for hour, day_of_week, and months using sine and cosine transformations. These transformations encode the cyclical nature of the attribute by mapping the values onto a circle. Sin and Cos allows us to capture the periodic nature of time-related variables.

    -   Applied rolling mean statistic for temp, clouds_all, rain_1h, and snow_1h to smooth out short-term fluctuations or noise and highlight underlying trends or patterns in the traffic.

    -   Created features with mean target encoding for temp_avg, rain_avg, and cloud_avg.

3.  **EDA (Exploratory Data Analysis) Observations**

4.  **Rain impact**: rain_bin is not representative of traffic_volume, but rain_1h and its transformed features might be.

    ![](65827c3f5f4504e4871d6ff93b92360f.png)

5.  **Seasonal trends**: Traffic volume is higher during summer and lower in fall and winter.

    ![](a05801379f598ee939e919273b9e2a51.png)

6.  **Weekday vs. weekend**: On average, total daily and peak traffic is lower on weekends compared to weekdays.

    ![](27e384c95fe36334f4cf84c5fdd43b26.png)

7.  **Yearly variations**: Less data is available for 2012, and dips in traffic_volume occurred in 2014, 2015, and 2018 are due to changes in road infrastructure.

    ![](8d0665a6f5ff4b6f13818f085cd1ecc5.png)

8.  **Time of day**: Most traffic occurs during morning and evening, with the least traffic volume late at night.

    ![](289e1ac97474d4d1327b666219f6100c.png)

9.  **Feature Selection:** lagged features, rolling mean statistic features, mean target encoded features, cyclical features + features that are excluded based on EDA performed and from lasso regression [ 'temp', 'rain_1h', 'clouds_all', 'traffic_volume', 'year', 'hour', 'date', 'holiday_bin', 'rain_bin', 'day_of_week_cat', 'hour_cat', 'snow_bin', 'snow_1h' ].

# Experimental Setup

| **![A screenshot of a computer screen Description automatically generated with medium confidence](482c609431304bdbb28db1f0445fc0c7.png)** **Figure 1**: GRU Architecture             |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **![A screenshot of a computer Description automatically generated with medium confidence](b52797fb919a06fc157e1685685713ec.png)** **Figure 2**: LSTM Architecture with Single Layer |
| **![A screenshot of a computer Description automatically generated with medium confidence](bdcb49c7f8ed983ad41fe92c78a1eddd.png)** **Figure 3**: LSTM Architecture with Two Layers   |
| **![A screenshot of a computer Description automatically generated with medium confidence](419f57f12ec63195110cf5e421df5446.png)** **Figure 4**: BI-LSTM Architecture                |
| **![A screenshot of a computer Description automatically generated with medium confidence](2cf03ee387c19993253689d22b589879.png)** **Figure 5**: CNN-LSTM Architecture               |

|  Machine Learning Models:   **Linear Regression (Worst Model):** plain linear regression with no regularization. **Random Forest Regression:** KFold cross-validation with 10 splits is applied to random forest regression, shuffling the data, using a fixed random state of 42 for consistent results, and calculated average performance metrics across all folds. **XGBoost:** It uses an objective of squared error, 1000 estimators, a fixed random state of 42, and parallel computation with n_jobs=-1. Early stopping is applied after 10 rounds to prevent overfitting. KFold cross-validation is employed to evaluate model performance, calculating average performance metrics across all folds for a comprehensive assessment. **Support Vector Machine (SVM):** The SVR model uses a Radial Basis Function (RBF) kernel with a regularization parameter (C) of 1, an epsilon-tube width of 0.1, and a gamma value determined by the 'scale' setting. |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

**Model Training**: All neural network models are trained using an Adam optimizer with a learning rate of 0.0001 and decay rate of 1e-5. It compiles the model using MSE as the loss function. The model is fitted with the training data, using 300 epochs, a batch size of 30, and validation data for evaluation. Early stopping and learning rate reduction callbacks are employed to prevent overfitting and improve performance, while preserving the training data order with 'shuffle' set to False.

| **Initial Prediction Curve – Linear Regression**                                                                               | **Final Prediction Curve – LSTM_1Layer**                                                                                       |
|--------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![](da961d2591f00279f7e227e320f229ee.png)                                                                                      | ![A picture containing screenshot, text, plot Description automatically generated](85b5f3a83d300a8970f95c1820bf42c7.png)       |
| ![A picture containing text, screenshot, line, plot Description automatically generated](7c105460193afb40752f8218ca928cd6.png) | ![A picture containing text, plot, screenshot, line Description automatically generated](470f5f91e7b5546c34a675c4d91a1faa.png) |

**Evaluation Metric:** Evaluated the quality of predicted traffic flow using the MAE (Mean Absolute Error), RMSE (Root Mean Square Error), MAPE (Mean Absolute Percentage Error), and R\^2 scores. MAE measures average absolute errors, RMSE emphasizes larger errors, MAPE expresses average errors as percentages, and R2 score indicates the regression model's goodness of fit compared to a baseline. Based on the data gathered and prediction curves, I have chosen MAPE score to be more representative of the Model Accuracy and hence chosen it as a base to evaluate and compare model performance.

| **ML Model**                 | **MAE** | **RMSE** | **MAPE** | **R2 Score** |
|------------------------------|---------|----------|----------|--------------|
| Linear Regression            | 0.0562  | 0.0736   | 0.2661   | 0.9262       |
| Random Forest Regression     | 0.0234  | 0.0377   | 0.0763   | 0.9806       |
| XGBoost                      | 0.0227  | 0.0362   | 0.0748   | 0.9799       |
| Support Vector Machine (SVM) | 0.0558  | 0.0719   | 0.2303   | 0.9294       |

**Table 1**: Experiment Results from Machine Learning Models

| **NN Model** | **MAE** | **RMSE** | **MAPE** | **R2 Score** |
|--------------|---------|----------|----------|--------------|
| GRU          | 4.1370  | 6.3518   | 0.0027   | 1.0          |
| LSTM – 1L    | 1.7612  | 2.2150   | 0.0012   | 1.0          |
| LSTM – 2L    | 13.583  | 15.258   | 0.008    | 0.9999       |
| BI-LSTM      | 4.8124  | 5.6245   | 0.0025   | 1.0          |
| CNN-LSTM     | 34.61   | 38.52    | 0.17     | 0.0189       |

**Table 2**: Experiment Results from Neural Network Architectures

Focusing on MAPE values, XGBoost (0.0748) and Random Forest Regression (0.0763) are the top-performing ML models, while LSTM-1L (0.0012) and GRU (0.0027) show the best performance among NN models. The CNN-LSTM model has a significantly higher MAPE value (0.17), indicating poorer performance. Overall, the results suggest that XGBoost, Random Forest Regression, LSTM-1L, and GRU models are effective choices for traffic flow volume prediction, with LSTM-1L having the lowest MAPE value, making it the best model among the evaluated options.

## Error Analysis

1.  **MAE**: The LSTM-1L model has the lowest Mean Absolute Error (MAE) among all models, indicating that its predictions have the smallest average deviation from the true values. Traditional ML models, such as Linear Regression and SVM, have higher MAE values, suggesting that they are less accurate than the NN models.

2.  **RMSE**: The LSTM-1L model also has the lowest Root Mean Square Error (RMSE), which signifies that it handles larger errors more effectively compared to other models. RMSE values for the ML models are higher, showing that they are not as good at handling large errors as the NN models.

3.  **MAPE**: LSTM-1L has the lowest Mean Absolute Percentage Error (MAPE), implying that it produces the most accurate predictions in terms of percentage errors. The CNN-LSTM model has the highest MAPE, indicating that its predictions are the least accurate in terms of percentage errors. ML models have higher MAPE values than the best performing NN models, reflecting their weaker performance.

4.  **R2 Score**: The R2 scores for LSTM-1L, GRU, and BI-LSTM models are all perfect (1.0), indicating that they can explain 100% of the variance in the data. In contrast, traditional ML models have lower R2 scores, suggesting they cannot explain the data's variance as effectively as the NN models.

## Use Case

The traffic flow volume prediction system developed in this project has the potential to provide several valuable use cases that can significantly improve urban transportation management and planning.

1.  Optimized real-time traffic management: The system can enable dynamic adjustments to traffic signal timings, facilitating smoother traffic flow and reducing congestion across the city.

2.  Informed infrastructure planning: Transportation authorities can leverage accurate traffic volume predictions to identify the need for additional lanes, alternate routes, or public transit options, accommodating future demands more effectively.

3.  Enhanced emergency response: By anticipating traffic volume patterns, emergency responders can select the most efficient routes to reach their destinations, minimizing response times and potentially saving lives.

4.  Empowered traveler information: Providing predicted traffic volume data to drivers and commuters can help them make informed decisions about the best times to travel or choose alternate routes to circumvent congestion.

5.  Sustainable urban planning: City planners can utilize traffic volume predictions to design more efficient and eco-friendly urban environments, incorporating pedestrian and bike-friendly infrastructure, and promoting public transit usage.

6.  Improved traffic simulation and modeling: Integrating the prediction system into traffic simulation tools can lead to the development of more accurate and realistic traffic models, fostering advancements in transportation research and development.

# Future Scope

1.  **Enhanced Accuracy:** Continued advancements in deep learning techniques, such as transformer-based models like GPT-3, will likely lead to increased accuracy in forecasting traffic volume.

2.  **Multimodal Approaches:** Exploring multimodal approaches that combine tabular features with other modalities such as text, audio, or user interactions can lead to more comprehensive and context-aware predictions.

3.  **Attention Mechanisms:** Exploring advanced attention mechanisms like self-attention or transformer-based attention can improve the prediction by focusing on relevant time periods and features.

# Conclusion

In conclusion, the single-layer LSTM with a dense layer provided the best performance across all metrics, closely followed by the BI-LSTM model. When compared to traditional ML models, both LSTM and BI-LSTM demonstrated superior performance in traffic flow volume prediction. Although BI-LSTM captures both forward and backward dependencies within time series data, the inclusion of lagged, cyclic, and rolling mean statistics may have supplied sufficient information, rendering the simpler LSTM model adequate for achieving the desired performance. A thorough analysis of the data has proven to be instrumental in this case, enabling to select the most suitable neural network models, which outperformed all the tested ML models.

