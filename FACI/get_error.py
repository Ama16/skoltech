import pandas as pd
import numpy as np
from etna.pipeline import Pipeline
from etna.models import SeasonalMovingAverageModel
from etna.metrics import MAE, MAPE
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def get_errors(ts, start_time, model, transforms):
    end_time = max(ts.df.index)
    
    valid_dates = pd.date_range(start_time, end_time)
    base_model = SeasonalMovingAverageModel(window=1, seasonality=7)
    pipeline = Pipeline(base_model)
    metrics, _, _ = pipeline.backtest(ts, metrics=[MAE()], n_folds=len(valid_dates))
    metrics.rename(columns={"fold_number": "timestamp"}, inplace=True)
    metrics["timestamp"] = metrics["timestamp"].apply(lambda x: valid_dates[x])
    sma_err_df = pd.DataFrame([])
    for i in np.unique(metrics["segment"]):
        df_tmp = pd.DataFrame({"timestamp": metrics[metrics["segment"] == i]["timestamp"].values,
                              "mae_lag7": metrics[metrics["segment"] == i]["MAE"].values,
                              "segment": i})
        sma_err_df = pd.concat([sma_err_df, df_tmp])

    model_err = pd.DataFrame([])
    for k in tqdm(valid_dates):
        ts_train, ts_test = ts.train_test_split(test_start=k, test_end=k)
        
        ts_train.fit_transform(transforms)
        model.fit(ts_train)
        future_ts = ts_train.make_future(1)
        forecast_ts = model.forecast(future_ts)
        metric = MAPE()(ts_test, forecast_ts)
        metric_mae = MAE()(ts_test, forecast_ts)
        
        mape_ = []
        mae_ = []
        segments = []
        forecasts_ = []
        for i in metric.keys():
            mape_.append(metric[i])
            mae_.append(metric_mae[i])
            forecasts_.append(forecast_ts[:, i, "target"].values[0])
            segments.append(i)
        df_tmp = pd.DataFrame({"segment": segments, "timestamp": k, "mape": mape_, "mae": mae_, "forecast": forecasts_})
        model_err = pd.concat([model_err, df_tmp])
    return model_err.merge(sma_err_df, on=["segment", "timestamp"])