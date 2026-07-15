# NYC Taxi Trip Duration Prediction

A Keras deep learning model that predicts NYC taxi trip duration from ~1.46 million historical trips.

## Overview

Uses the [Kaggle "New York City Taxi Trip Duration"](https://www.kaggle.com/c/nyc-taxi-trip-duration) dataset (`train.csv` / `test.csv`, not included in this repo — download separately). Features include vendor ID, passenger count, pickup/dropoff latitude & longitude, and a store-and-forward flag.

The notebook iterates through three neural networks to get from underfitting to a good fit:

1. **Model 1** — 5 Dense(14) + Dropout layers, MSE loss. Underfits badly (loss barely moves across 50 epochs).
2. **Model 2** — smaller 4-layer network with early stopping. Still underfits.
3. **Model 3** — 4× Dense(14)+Dropout(0.5) layers, tuned loss function. No under- or over-fitting — validation loss tracks training loss closely.

Final model's mean predicted trip duration is ≈633 seconds, a large improvement over the earlier models' ≈800-990s (which had collapsed toward predicting close to the mean regardless of input).

## Tech stack

- `pandas` / `numpy` for data prep
- `scikit-learn` (`MinMaxScaler`, `train_test_split`)
- `TensorFlow` / `Keras` (`Sequential`, `Dense`, `Dropout`, `EarlyStopping`)

## Running it

```bash
pip install pandas numpy scikit-learn tensorflow
```

Download `train.csv` and `test.csv` from the Kaggle competition page above and place them alongside the notebook.
