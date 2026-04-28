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

## 5. Hypothesis Tests
#### A. Relationship Between Temperature and Visitor Numbers 
> **Method:** Pearson Correlation Test <br>
> **Objective:** To determine whether air temperature significantly affects the number of visitors to the camping sites.

* **$H_0$ (Null Hypothesis):** There is no statistically significant relationship between temperature and the number of visitors.
* **$H_1$ (Alternative Hypothesis):** There is a statistically significant relationship between temperature and the number of visitors.

**Result and Interpretation:**
The $H_0$ hypothesis was rejected in **17** of the 18 camping sites analyzed ($p < 0.05$). The relationship was found to be insignificant in only 1 location.

**Key Finding:** Temperature is the strongest and most universal meteorological factor determining visitor density in Black Sea camping sites. As the weather warms up, the demand for camping sites increases in a statistically clear and measurable way. Temperature serves as the primary predictive variable for our modeling.

---

#### B. Relationship Between Precipitation and Visitor Numbers
> **Method:** Independent Samples Welch's T-Test <br>
> **Objective:** To determine if there is a significant difference in the number of visitors between rainy and non-rainy weeks.

* **$H_0$:** The visitor averages for rainy and non-rainy weeks are equal (Precipitation has no effect).
* **$H_1$:** There is a significant difference in the visitor averages between rainy and non-rainy weeks.

**Result and Interpretation:**
The $H_0$ hypothesis was rejected in **7** of the 18 camping sites analyzed (e.g., *Ayder Yaylası, Borçka Karagöl, Elevit Yaylası*), proving that precipitation has a significant effect on the number of visitors. In the remaining **11 camping sites**, no statistically significant difference was found.

**Key Finding:** Unlike temperature, precipitation does not affect every camping site equally. While some camping sites show high "sensitivity" to precipitation and lose visitors, others maintain their visitor base even in rainy weather. This variance likely depends on external factors such as physical infrastructure (availability of indoor areas), transportation difficulty, or visitor profile (adventurous vs. day-tripper).

---

#### C. Visitor Volume and Precipitation Sensitivity
> **Method:** Chi-Square Test of Independence <br>
> **Objective:** To examine whether we can make a general deduction such as *"camping sites that are visited more frequently (popular) are affected more/less by precipitation."*

* **$H_0$:** The visitor volume category (Above/Below Median) and precipitation sensitivity are independent of each other (No relationship).
* **$H_1$:** There is a significant relationship between the visitor category and precipitation sensitivity.

**Result and Interpretation:**
* **Test Statistic:** $\chi^2 = 0.0000$
* **P-Value:** $1.0000$

Since the calculated p-value is greater than $0.05$, **the $H_0$ hypothesis is accepted.**

**Key Finding:** Whether a camping site is popular (crowded) or quiet on an annual basis is not a factor that affects its visitor loss on rainy days (precipitation sensitivity). Both crowded and lesser-known quiet camping sites are affected by precipitation in entirely independent ways. 


## 6. Academic Integrity & AI Disclosure

In accordance with the academic integrity guidelines of the DSA 210 course, I declare that AI tools (e.g., LLMs) were utilized to assist with the code generation, data processing, and text refinement stages of this project. 

All specific prompts used and the corresponding generated outputs (chat histories) have been fully saved and documented. They are readily available upon request if needed.
