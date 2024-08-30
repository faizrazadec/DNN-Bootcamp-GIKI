# Solar Power Generation and Load Prediction

## Project Overview

The goal of this project was to predict solar power generation (`generation_W`) and load consumption (`load_W`) using advanced machine learning techniques. We focused on improving prediction accuracy to optimize renewable energy management.

## Strategy

### 1. Data Collection and Preparation
- **Source:** The dataset consisted of time-series data related to solar power generation and load consumption, enriched with weather information.
- **Preprocessing:** The data was preprocessed to remove irrelevant columns and handle missing values. New features like time-based attributes (e.g., year, month, day, hour, minute) and trigonometric transformations of cyclical features (e.g., hour_sin, hour_cos) were created.

### 2. Feature Engineering
- **Time-Based Features:** Extracted features such as year, month, day, hour, minute, and day of the week from the `timestamp` column.
- **Cyclical Features:** Applied trigonometric transformations to encode cyclical time features, such as hours and months, using sine and cosine functions.
- **Lag Features:** Introduced lag features to capture the temporal dependencies in the data.

### 3. Model Selection
- **LightGBM:** Chosen for its efficiency and ability to handle large datasets. LightGBM uses a feedback mechanism to improve mean absolute error (MAE).
- **GRU:** Explored for capturing temporal patterns in the data, but LightGBM was preferred due to better performance in this context.

### 4. Model Implementation
- **Training Process:** 
  - Data was split into training and test sets using an 80-20 split.
  - The models were trained on One-Hot Encoded (OHE) features.
- **Hyperparameter Tuning:** LightGBM's hyperparameters were optimized for better accuracy.

### 5. Evaluation
- **Metrics:** The model was evaluated using Mean Absolute Error (MAE).
- **Validation:** Overfitting was checked by evaluating the model on the validation set. Early stopping was implemented to prevent overfitting.

## Data Format

### Training Data

| Column Name   | Description                               | Data Type |
| ------------- | ----------------------------------------- | --------- |
| `timestamp`   | Timestamp of the data point               | `datetime`|
| `system_id`   | Unique identifier for the PV system       | `int`     |
| `generation_W`| Power generation in watts                 | `float64` |
| `load_W`      | Load consumption in watts                 | `float64` |
| `panels_capacity` | Capacity of solar panels               | `float64` |
| `load_capacity`   | Load capacity                          | `float64` |
| `tavg`        | Average temperature                       | `float64` |
| `tmin`        | Minimum temperature                       | `float64` |
| `tmax`        | Maximum temperature                       | `float64` |
| `prcp`        | Precipitation                             | `float64` |
| `wdir`        | Wind direction                            | `float64` |
| `wspd`        | Wind speed                                | `float64` |
| `pres`        | Atmospheric pressure                      | `float64` |
| `year`        | Extracted year from timestamp             | `int`     |
| `month`       | Extracted month from timestamp            | `int`     |
| `day`         | Extracted day from timestamp              | `int`     |
| `hour`        | Extracted hour from timestamp             | `int`     |
| `minute`      | Extracted minute from timestamp           | `int`     |
| `day_of_week` | Extracted day of the week from timestamp  | `int`     |
| `hour_sin`    | Sine transformation of the hour feature   | `float64` |
| `hour_cos`    | Cosine transformation of the hour feature | `float64` |
| `month_sin`   | Sine transformation of the month feature  | `float64` |
| `month_cos`   | Cosine transformation of the month feature| `float64` |

### Test Data

The test data follows a similar format, excluding the target columns `generation_W` and `load_W`.

## Model Implementation

The final model used in this project was LightGBM, trained with optimized hyperparameters. The training process was conducted on the One-Hot Encoded data, with a focus on minimizing MAE through iterative improvements.

### Saving and Loading the Model
The trained LightGBM model was saved using `joblib`:
```python
import joblib

# Save the model
joblib.dump(lgb_model_load, 'lgb_model_load.pkl')

# Load the model
lgb_model_load = joblib.load('lgb_model_load.pkl')
