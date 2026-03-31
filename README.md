# A Decision Support Model for Spatial and Temporal Analysis of Camping Sites in the Black Sea Region

## 1. Motivation and Purpose
The aim of this project is to analyze the 30 camping sites which are equally distributed in the black sea region (10 for west, 10 for middle and 10 for east black sea region). The main goal is to build a machine learning algorithm which can determine the condition of each of these camp site (Estimated Human density and weather condition, overall risk expectation) in a specific date by analysing weather condition, terrain status, transportation, probability of wildlife encounter and human mobility.

## 2. Data Sources and Collection Methodology
Since there is no centralized data set which can contain all the specific parameters and information for analysing these 30 camp sites, the “Unified Data Set” will be achieved by collecting and cleaning the data set obtained by independent sources by the following pipeline:

* **Meteorological Data:** Data weather condition will be extracted from the meteostat Python library. The obtained data will be divided into weekly sections. Spatial interpolation will be used for high-altitude areas that lack direct measurement stations.
* **Digital footprint (Human Density):** some natural park have formal data collected by the government. However for isolated camp sites there is no formal data about human density. Therefore to estimate the monthly digital footprint, Google comments about the isolated camp sites and Wikiloc will be analyzed and the human density will be estimated accordingly by comparing the data collected by web scraping and the governmental formal data.
* **Accessibility to the healthcare and city site:** to correctly mesure how much the camp site is isolated from the city, the nearest city site and healthcare center's distance and accessibility is estimated with OSMnx library and OpenStreetMap data.
* **Terrain Risks:** Slope data will be pulled from NASA SRTM Digital Elevation Models and soil composition data will be pulled from the SoilGrids REST API. Also the road surface data will be pulled from OpenStreetMap api for each camp site. These data will be used for the estimation of transportation risks score.
* **Wild Life Encounter Risks:** a risk score will be estimated for human-wildlife encounters based on the Human Footprint Index, distance to nearby villages, and the biological seasons of the animals (such as hibernation periods).

## 3. Dataset Characteristics

* **Collection Period & Temporal Resolution:** The collected data will cover a historical period of 5 years. The collected data will be divided into weekly intervals and be used for the estimation of weather conditions. For the human densitiy data all of the available Google comments, Wikiloc publishes and formal governmental records about the nature parks will be analyzed to estimate the monthly human density.
* **Sample Size:** The Weather condition data will be the 30 camp sites' weather condition over a 5 years period (260 week) which will be approximately 7800 primary records. For the human density data Wikiloc and Google comment will be analysed, therefore no exact number can be shown (approximately between 50000 – 60000 Google and Wikiloc data).
* **Key Variables and Expected Units:**
  * *Weather:* Temperature in Celsius (°C), Precipitation in millimeters (mm).
  * *Distance/Access:* Driving time in minutes.
  * *Terrain:* Slope in degrees, Soil composition in percentages (clay/sand).
  * *Risk Scores:* Normalized indices (scaled from 0.0 to 1.0).


## 4. Analytical Approach

The analysis proceeds in three stages: firstly, using exploratory data analysis on the East, West, and Middle Black Sea region's campsites to determine weather conditions, terrain risks, and human density to map distributional patterns and bivariate relationships. Secondly, applying correlation analysis and hypothesis tests to determine how much accessibility, human density, and wildlife encounter risks affect the condition of each campsite. Thirdly, for the machine learning part, models like Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, and SVR will be used to predict the risk score between 0.0 and 1.0. I plan to apply SHAP values to the best model to understand which factors are more important. Also, the campsites will be classified as Low, Medium, or High risk. Finally, unsupervised clustering will help identify sites that get highly visited even though they are hard to reach.
