from sklearn.pipeline import Pipeline
from pipeline_classes import CreateCombinedDataFrame
from _config import config
import time
import pandas as pd

accel_data = pd.read_csv(config["accel_path"])
reports_data = pd.read_csv(config["reports_path"])

X = (reports_data, accel_data)

# This pipeline combines the self-reports and accelerometer dataframes with a given timewindow into a single dataframe as a csv file
combining_dataframes_pipeline = Pipeline([
    #('import_data', ImportData(use_accel=True, use_reports=True, use_combined=False, use_features=False)),  # input path to self-reports data),
    ('create_combined_dataframe', CreateCombinedDataFrame(time_window=config["time_window"], label_columns=config["label_columns"])),
])

# This will measure the time taken to run the pipeline
start_time = time.time()

# This will start the pipeline and return the combined dataframe
output_df = combining_dataframes_pipeline.fit_transform(X)


end_time = time.time()
print(f"Time taken: {int((end_time - start_time) // 60)} minutes and {(end_time - start_time) % 60:.2f} seconds")
