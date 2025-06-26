# Customer Segmentation and Profiling for an retail platform

## Overview

This repo is designed as a way of practicing unsupervised learning (clustering) with the goal of segmenting and profiling customers for a retail platform. In it you can find:

* A [Jupyter Notebook](customerProfiling.ipynb) containing the pipeline, from data cleaning and exploration through to unsupervised K-means clustering and customer profiling. The final output of this pipeline, alongside informative plots based on the formed clusters, is a dataset which feeds into...
* A [Streamlit App](https://customersegmentationandprofiling.streamlit.app/) with a reactive UI designed for non-technical stakeholders to explore the impact of the results. It allows a stakeholder to input a hypothetical customer's details and understand what type of customer they are likely to be (e.g. high spending, buys luxury etc...), as well as outputting a recommended marketing plan for that segment. I use K-nearest neighbours to predict what cluster an inputted customer would be in, and recommend a marketing strategy based on this.


## Dataset Description

The [customer personality analysis dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) is a synthetic marketing dataset. It is well-suited for unsupervised learning tasks such as clustering and RFM analysis. The dataset is specifically desgined to mimic that which you might find in a retail company which sells consumer goods (food, jewellery, household items).

### Feature Description

#### People Attributes

| Column Name      | Description                                                  |
|------------------|--------------------------------------------------------------|
| `ID`             | Customer's unique identifier                                 |
| `Year_Birth`     | Customer's birth year                                        |
| `Education`      | Customer's education level                                   |
| `Marital_Status` | Customer's marital status                                    |
| `Income`         | Customer's yearly household income                           |
| `Kidhome`        | Number of children in customer's household                   |
| `Teenhome`       | Number of teenagers in customer's household                  |
| `Dt_Customer`    | Date of customer's enrollment with the company               |
| `Recency`        | Number of days since customer's last purchase                |
| `Complain`       | 1 if the customer complained in the last 2 years, 0 otherwise|

#### Product Spending

| Column Name         | Description                                  |
|---------------------|----------------------------------------------|
| `MntWines`          | Amount spent on wine in last 2 years         |
| `MntFruits`         | Amount spent on fruits in last 2 years       |
| `MntMeatProducts`   | Amount spent on meat in last 2 years         |
| `MntFishProducts`   | Amount spent on fish in last 2 years         |
| `MntSweetProducts`  | Amount spent on sweets in last 2 years       |
| `MntGoldProds`      | Amount spent on gold in last 2 years         |

#### Promotion & Campaign Response

| Column Name      | Description                                               |
|------------------|-----------------------------------------------------------|
| `NumDealsPurchases` | Number of purchases made with a discount               |
| `AcceptedCmp1`   | 1 if customer accepted the offer in the 1st campaign      |
| `AcceptedCmp2`   | 1 if customer accepted the offer in the 2nd campaign      |
| `AcceptedCmp3`   | 1 if customer accepted the offer in the 3rd campaign      |
| `AcceptedCmp4`   | 1 if customer accepted the offer in the 4th campaign      |
| `AcceptedCmp5`   | 1 if customer accepted the offer in the 5th campaign      |
| `Response`       | 1 if customer accepted the offer in the last campaign     |

#### Purchase Channels

| Column Name           | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `NumWebPurchases`     | Number of purchases made through the company’s website           |
| `NumCatalogPurchases` | Number of purchases made using a catalogue                       |
| `NumStorePurchases`   | Number of purchases made directly in stores                      |
| `NumWebVisitsMonth`   | Number of visits to company’s website in the last month          |

