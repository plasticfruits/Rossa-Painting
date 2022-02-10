# README

There are 5 different python files:
* `model_predict.py` use to load and predict a model. It exports a `.csv` file with the results
* `utils.py` contains utility functions
* `generate_enricher_df` generates a .csv file with enrichment that can be loaded for modelling and prediction.  The following features are included:
    - Store: used for join
    - Assortment:
    - CompetitionDistance:
    - avg_wday_sales: average sales per week day
    - avg_wyear_sales: average sales per week of the year
    - avg_monthly_sales: average sales per month
    - avg_wday_customes: average customers per week day
    - avg_wyear_customes: average customers per week of the year
    - avg_monthly_customes: average customers per month
    - avg_sales_school_holiday: average sales for a given school holiday
    - avg_sales_state_holiday: average sales for a given state holiday
    - avg_sales_store_type: average sales by store type
* `models_training.py` used for testing different models and hyper-parameter tunning methods
* `model_export.py` used to export models as `.sav` files in the `models` directory. Currently exporting:
    - Linear Regression model
    - Gradient Boosting Tree


### Predict
Uses `model_predict.py` to predict the sales for a given dataset.

