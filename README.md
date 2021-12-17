# Forecasting-intermittent-demand-a-comparative-approach

### Problem description
Sustainable use of drinking water is essential to society. Water resources are under increasing pressure from growing demand and climate change. Dutch water utilities are looking to raise awareness and find potential ways of saving drinking water. In doing so, [Seita](https://seita.nl/project/water-leaks-process-analytics/) is working on a pattern recognition and classification toolkit for monitoring systems that will allow real-time automated analysis. Distinctive in this work is the use of Intermittent Demand data. ”When a product has many periods of zero demand (also known as irregular demand), it is said to have intermittent demand (ID). In these cases, the demand is often small and highly variable in scale and the many zero values in ID time-series usually render usual forecasting methods difficult to apply. 

### Data
The original dataset used in this analysis describes the water consumption from a customer building. Data was collected from 4 July 2019 to 31 December 2020. Value is expressed in litres and it refers to the quantity of water measured up to that date.

![alt text](https://user-images.githubusercontent.com/57104110/146569112-8be75ce9-509f-4201-8d03-b7a301eb3b35.png)

### Experimental setup
In order to come up with to the final results, the following approach was adopted. A series of 8 forecasters were implemented, where both Machine Learning and Statistical models were used. Then models’ predictions were compared among themselves to evaluate their outcomes. Eventually models performances were investigated against different forecast horizons and different times of the day (including anomalies), by selecting some specific scenarios. These particular scenarios were selected to have the following characteristics: 
* a Forecast Horizon  equal to 48 hours, which means we want to predict each time two days in the future;
* 2 time moments of the day, which are noon and midnight, as we want to see how it differs if we start predicting values during the day or during the night;
* 3 different moments of the demand, which are a moment of low demand of a normal week, a moment of demand just before the beginning of the anomaly and a moment of demand in a week into the anomaly. 

![alt text](https://user-images.githubusercontent.com/57104110/146570045-b0d80441-7ff3-45ce-9362-205d1c886176.png) | ![alt text](https://user-images.githubusercontent.com/57104110/146570113-aecd1a33-3512-43ab-bfa1-63f97e2e8bf0.png)

