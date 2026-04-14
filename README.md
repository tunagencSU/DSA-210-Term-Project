# A Decision Support Model for Spatial and Temporal Analysis of Camping Sites in the Black Sea Region

## 1. Motivation and Purpose
The aim of this project is to analyze the 18 camping sites in the black sea region. The main goal is to build a machine learning algorithm which can determine the condition of each of these camp site (Estimated Human density, weather condition, overall satisfaction expectation) in a specific date by analysing weather condition, transportation condition, human mobility and satisfaction.

## 2. Data Sources and Collection Methodology
Since there is no centralized data set which can contain all the specific parameters and information for analysing these 18 camp sites, the “Unified Data Set” will be achieved by collecting and cleaning the data set obtained by independent sources by the following pipeline:

* **Meteorological Data:** Data weather condition will be extracted from the meteostat Python library. The obtained data will be divided into weekly sections. Spatial interpolation will be used for high-altitude areas that lack direct measurement stations.
* **Digital footprint (Human Density):** for isolated camp sites there is no formal data about human density. Therefore to estimate the weekly digital footprint, the annual number of human visits data will be collected by web scraping and offical formal information sources (news, journal etc.). Afther the annual data collection, Google comments will be analyzed and the human density will be estimated in weekly periods accordingly by comparing the google comment data and the data collected by web scraping and offical formal information sources (news, journal etc.).
* **Accessibility to the healthcare and city site and road conditions:** to correctly mesure the transportation condition and how much the camp site is isolated from the city, the nearest city site and healthcare center's distance and accessibility is estimated with google maps and web scraping. Slope data will be estimated accordingly with google earth data. Also the road surface data will be estimated from OpenStreetMap for each camp site. These data will be used for the estimation of transportation risks score.

## 3. Dataset Characteristics

* **Collection Period & Temporal Resolution:** The collected data will cover a historical period of 5 years (2021-2025). The collected data will be divided into weekly intervals and be used for the estimation of weather conditions. For the human densitiy data all 2024 and 2025 Google comments, Wikiloc publishes and formal information sources about camp cites will be analyzed to estimate the monthly human density.
* **Sample Size:** The Weather condition data will be the 18 camp sites' weather condition over a 5 years period (260 week) which will be approximately 4680 primary records. For the human density Google comment will be analysed, therefore no exact number can be shown (approximately between 40000 – 50000 Google comment data).
* **Key Variables and Expected Units:**
  * *Weather:* Temperature in Celsius (°C), Precipitation in millimeters (mm).
  * *Distance/Access:* Driving time in minutes.
  * *Terrain:* Slope in degrees, Soil composition in percentages (clay/sand).
  * *Risk Scores:* Normalized indices (scaled from 0.0 to 1.0).


## 4. Analytical Approach

The analysis proceeds in three stages: firstly, using exploratory data analysis on the Black Sea region's campsites to determine weather conditions, terrain risks, and human density to map distributional patterns and bivariate relationships. Secondly, applying correlation analysis and hypothesis tests to determine how much accessibility, human density and weather condition affect the condition of each campsite. Thirdly, for the machine learning part, models like Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, and SVR will be used to predict the overall score between 0.0 and 1.0. I plan to apply SHAP values to the best model to understand which factors are more important. Also, the campsites will be classified as Low, Medium, or High risk. Finally, unsupervised clustering will help identify sites that get highly visited even though they are hard to reach.
