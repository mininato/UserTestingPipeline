import gradio as gr
from pipeline_classes import CreateCombinedDataFrame, ScaleXYZData, ExtractFeatures, TrainModel, ClassifyMovementData, LowPassFilter, PCAHandler
from sklearn.pipeline import Pipeline
from _config import config
import pandas as pd
import numpy as np
import joblib
import json


# Define pipelines
combining_dataframes_pipeline = Pipeline([
    #('import_data', ImportData(use_accel=True, use_reports=True, use_combined=False, use_features=False)),
    ('create_combined_dataframe', CreateCombinedDataFrame(time_window=None, label_columns=None)),
])

feature_extraction_pipeline = Pipeline([
    #('import_data', ImportData(use_accel=False, use_reports=False, use_combined=True, use_features=False)),
    ('low_pass_filter', LowPassFilter(cutoff_frequency=None, sampling_rate=None, order=None)),
    ('scale_xyz_data', ScaleXYZData(scaler_type=None)),
    ('extract_features', ExtractFeatures(window_length=None,
                                         window_step_size=None,
                                         data_frequency=None,
                                         selected_domains=None,
                                         include_magnitude=None,
                                         features_label_columns=None)),
])

training_model_pipeline = Pipeline([
    #('import_data', ImportData(use_accel=False, use_reports=False, use_combined=False, use_features=True)),
    ('pca_handler', PCAHandler(apply_pca=None, variance=None)),
    ('train_model', TrainModel(classifier=None, train_label= None, target=None)),
])

analyzing_data_pipeline = Pipeline([
    #('import_data', ImportData(use_accel=True, use_reports=False, use_combined=False, use_features=False)),
    ('low_pass_filter', LowPassFilter(cutoff_frequency=None, sampling_rate=None, order=None)),
    ('scale_xyz_data', ScaleXYZData(scaler_type=None)),
    ('extract_features', ExtractFeatures(window_length=None,
                                         window_step_size=None,
                                         data_frequency=None,
                                         selected_domains=None,
                                         include_magnitude=None,
                                         features_label_columns=None)),
    ('classify_movement_data', ClassifyMovementData(model_file=None)),
])

complete_training_model_pipeline = Pipeline([
    #('import_data', ImportData(use_accel=True, use_reports=True, use_combined=False, use_features=False)),
    ('create_combined_dataframe', CreateCombinedDataFrame(time_window=None, label_columns=None)),
    ('low_pass_filter', LowPassFilter(cutoff_frequency=None, sampling_rate=None, order=None)),
    ('scale_xyz_data', ScaleXYZData(scaler_type=None)),
    ('extract_features', ExtractFeatures(window_length=None,
                                         window_step_size=None,
                                         data_frequency=None,
                                         selected_domains=None,
                                         include_magnitude=None,
                                         features_label_columns=None)),
    ('pca_handler', PCAHandler(apply_pca=None, variance=None)),
    ('train_model', TrainModel(classifier=None, train_label= None, target=None)),
])

def execute_combine_pipeline(accel_file, report_file,
                     time_window=None, label_columns=None
                     ):
    try:      
        # Load data files only if paths are valid
        accel_data = pd.read_csv(accel_file) if accel_file else None
        report_data = pd.read_csv(report_file) if report_file else None

        # Validate inputs for the selected pipeline
        if accel_data is None or report_data is None:
            return "Error: Both accelerometer and self-report data files are required for this pipeline.", None
        combining_dataframes_pipeline.set_params(
            create_combined_dataframe__time_window=time_window,
            create_combined_dataframe__label_columns=label_columns.split(','))
        X = report_data, accel_data
        result = combining_dataframes_pipeline.fit_transform(X)
        output_file = "combine_dataframes_output.csv"
        result.to_csv(output_file, index=False)

        return output_file
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return str(e), None


def execute_feature_extraction_pipeline(combined_file, cutoff_frequency, order, scaler_type, window_length, window_step_size, data_frequency, include_magnitude, features_label_columns):
    try:
        combined_data = pd.read_csv(combined_file) if combined_file else None
        if combined_data is None:
            return "Error: Combined data file is required for this pipeline.", None

        feature_extraction_pipeline.set_params(
            low_pass_filter__cutoff_frequency=cutoff_frequency,
            low_pass_filter__order=order, 
            low_pass_filter__sampling_rate=data_frequency,
            scale_xyz_data__scaler_type=scaler_type, 
            extract_features__window_length=window_length,
            extract_features__window_step_size=window_step_size, 
            extract_features__data_frequency=data_frequency,
            #extract_features__selected_domains=None,
            extract_features__include_magnitude=include_magnitude,
            extract_features__features_label_columns=features_label_columns.split(','))
        result = feature_extraction_pipeline.fit_transform(combined_data)
        output_file = "extract_features_output.csv"
        result.to_csv(output_file, index=False)
        return output_file

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return str(e)
    
def execute_training_pipeline(features_file, apply_pca, pca_variance, classifier, train_label, target):
    try:
        print(f"features_file: {features_file}")
        features_data = pd.read_csv(features_file) if features_file else None
        if features_data is None:
            return "Error: Features data file is required for this pipeline.", None
        
        training_model_pipeline.set_params(
            pca_handler__apply_pca=apply_pca,
            pca_handler__variance=pca_variance,
            train_model__classifier=classifier,
            train_model__train_label=train_label,
            train_model__target=target)
        
        X = features_data
        training_model_pipeline.fit(X)
        output_file, secondary_output_file = training_model_pipeline.named_steps['train_model'].get_output_files()
        return output_file, secondary_output_file

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return str(e), None
    
def execute_analyze_pipeline(accel_file, model_file, cutoff_frequency, order, scaler_type, window_length, data_frequency, include_magnitude, features_label_columns):
    try:
        print("hallo")
        accel_data = pd.read_csv(accel_file) if accel_file else None
        print("hallo2")
        if accel_data is None:
            return "Error: Accelerometer data file is required for this pipeline.", None
        
        analyzing_data_pipeline.set_params(
            low_pass_filter__cutoff_frequency=cutoff_frequency,
            low_pass_filter__order=order, 
            low_pass_filter__sampling_rate=data_frequency,
            scale_xyz_data__scaler_type=scaler_type, 
            extract_features__window_length=window_length,
            extract_features__window_step_size=window_length, 
            extract_features__data_frequency=data_frequency,
            #extract_features__selected_domains=None,
            extract_features__include_magnitude=include_magnitude,
            extract_features__features_label_columns=features_label_columns.split(','),
            classify_movement_data__model_file=model_file.name
            )
        print("hallo3")

        result = analyzing_data_pipeline.fit_transform(accel_data)
        output_file = "analyze_data_output.csv"
        result.to_csv(output_file, index=False)
        return output_file

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return str(e), None
    
def execute_complete_training_pipeline(accel_file, report_file, time_window, label_columns,
                                       cutoff_frequency, order, scaler_type, window_length, window_step_size, data_frequency, include_magnitude, features_label_columns,
                                        apply_pca, pca_variance, classifier, train_label, target):
    try:
        accel_data = pd.read_csv(accel_file) if accel_file else None
        report_data = pd.read_csv(report_file) if report_file else None
        if accel_data is None or report_data is None:
            return "Error: Both accelerometer and self-report data files are required for this pipeline.", None
        
        complete_training_model_pipeline.set_params(
            create_combined_dataframe__time_window=time_window,
            create_combined_dataframe__label_columns=label_columns.split(','),
            low_pass_filter__cutoff_frequency=cutoff_frequency,
            low_pass_filter__order=order, 
            low_pass_filter__sampling_rate=data_frequency,
            scale_xyz_data__scaler_type=scaler_type, 
            extract_features__window_length=window_length,
            extract_features__window_step_size=window_step_size, 
            extract_features__data_frequency=data_frequency,
            #extract_features__selected_domains=None,
            extract_features__include_magnitude=include_magnitude,
            extract_features__features_label_columns=label_columns.split(','),
            pca_handler__apply_pca=apply_pca,
            pca_handler__variance=pca_variance,
            train_model__classifier=classifier,
            train_model__train_label=label_columns,
            train_model__target=target
        )
        X = report_data, accel_data
        complete_training_model_pipeline.fit(X)
        output_file, secondary_output_file = complete_training_model_pipeline.named_steps['train_model'].get_output_files()
        return output_file, secondary_output_file

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return str(e), None

# Gradio Blocks Interface
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Combine DataFrames"):
            accel_file = gr.File(label="Upload Accelerometer Data")
            report_file = gr.File(label="Upload Self-Report Data")
            time_window = gr.Number(label="Time Window (minutes)", value=2)
            label_columns = gr.Textbox(label="Label Columns (comma-separated)", value="valence,arousal")
            combine_button = gr.Button("Combine DataFrames")
            combine_output = gr.File(label="Download Combined DataFrame")

            def combine_dataframes(accel_file, report_file, time_window, label_columns):
                output_file = execute_combine_pipeline(accel_file, report_file, time_window, label_columns)
                return output_file

            combine_button.click(combine_dataframes, inputs=[accel_file, report_file, time_window, label_columns], outputs=combine_output)

        with gr.TabItem("Extract Features"):
            combined_file = gr.File(label="Upload Combined Data")

            cutoff_frequency = gr.Number(label="Cutoff Frequency (Hz)", value=10)
            order = gr.Number(label="Order", value=4)

            scaler_type = gr.Radio(label="Scaler Type", choices=["standard", "minmax"])

            window_length = gr.Number(label="Window Length (seconds)", value=60)
            window_step_size = gr.Number(label="Window Step Size (seconds)", value=20)
            data_frequency = gr.Number(label="Data Frequency (Hz)", value=25)

            #selected_domains= gr.Textbox(label="Only these domains (Comma-Seperated) / If you want all then leave out", value=None)
            include_magnitude= gr.Checkbox(label="Include Magnitude", value=True)
            features_label_columns= gr.Textbox(label="Label Columns (comma-separated)", value="valence,arousal")
        
            extract_button = gr.Button("Extract Features")
            extract_output = gr.File(label="Download Extracted Features")

            def extract_features(combined_file, cutoff_frequency, order, scaler_type, window_length, window_step_size, data_frequency, include_magnitude, features_label_columns):
                    output_file = execute_feature_extraction_pipeline(combined_file,
                                                    cutoff_frequency, order, scaler_type, window_length, window_step_size, data_frequency,
                                                     include_magnitude, features_label_columns
                                                    )
                    return output_file

            extract_button.click(extract_features, inputs=[combined_file, cutoff_frequency, order, scaler_type, window_length, window_step_size,
                                                           data_frequency, include_magnitude, features_label_columns],  outputs=extract_output)

        with gr.TabItem("Train Model"):
            features_file = gr.File(label="Upload Features Data")

            apply_pca = gr.Checkbox(label="Apply PCA", value=False)
            pca_variance = gr.Number(label="PCA Variance", value=0.95)
            classifier = gr.Dropdown(label="Classifier", choices=["xgboost", "svm", "randomforest"], value="xgboost")
            train_label = gr.Textbox(label="Label Columns (comma-separated)", value="valence,arousal")
            target = gr.Textbox(label="Target Label", value="arousal")

            train_button = gr.Button("Train Model")
            train_output_json = gr.File(label="Download Model JSON")
            train_output_pkl = gr.File(label="Download Model PKL")

            def train_model(features_file, apply_pca, pca_variance, classifier, train_label,  target):
                output_file, secondary_output_file = execute_training_pipeline(features_file, apply_pca, pca_variance, classifier, train_label, target)
                return output_file, secondary_output_file

            train_button.click(train_model, inputs=[features_file, apply_pca, pca_variance, classifier, train_label, target], outputs=[train_output_json, train_output_pkl])

        with gr.TabItem("Analyze Data"):
            accel_file = gr.File(label="Upload Accelerometer Data")
            model_file = gr.File(label="Upload Model")

            cutoff_frequency = gr.Number(label="Cutoff Frequency (Hz)", value=10)
            order = gr.Number(label="Order", value=4)

            scaler_type = gr.Radio(label="Scaler Type", choices=["standard", "minmax"])

            window_length = gr.Number(label="Window Length (seconds)", value=60)
            data_frequency = gr.Number(label="Data Frequency (Hz)", value=25)

            #selected_domains= gr.Textbox(label="Only these domains (Comma-Seperated) / If you want all then leave out", value=None)
            include_magnitude= gr.Checkbox(label="Include Magnitude", value=True)
            features_label_columns= gr.Textbox(label="Label Columns (comma-separated)", value="valence,arousal")
        
            analyze_button = gr.Button("Analyze Data")
            analyze_output = gr.File(label="Download Analyzed Data")

            def analyze_data(accel_file, model_file, cutoff_frequency, order, scaler_type, window_length, data_frequency, include_magnitude, features_label_columns):
                output_file = execute_analyze_pipeline(accel_file, model_file, cutoff_frequency, order, scaler_type, window_length,
                                                           data_frequency, include_magnitude, features_label_columns)
                return output_file

            analyze_button.click(analyze_data, inputs=[accel_file, model_file, cutoff_frequency, order, scaler_type, window_length,
                                                           data_frequency, include_magnitude, features_label_columns ], outputs=analyze_output)

        with gr.TabItem("Complete Train Model"):
            accel_file = gr.File(label="Upload Accelerometer Data")
            report_file = gr.File(label="Upload Self-Report Data")

            time_window = gr.Number(label="Time Window (minutes)", value=2)
            label_columns = gr.Textbox(label="Label Columns (comma-separated)", value="valence,arousal")

            cutoff_frequency = gr.Number(label="Cutoff Frequency (Hz)", value=10)
            order = gr.Number(label="Order", value=4)

            scaler_type = gr.Radio(label="Scaler Type", choices=["standard", "minmax"])
            
            window_length = gr.Number(label="Window Length (seconds)", value=60)
            window_step_size = gr.Number(label="Window Step Size (seconds)", value=20)
            data_frequency = gr.Number(label="Data Frequency (Hz)", value=25)

            include_magnitude= gr.Checkbox(label="Include Magnitude", value=True)
            #features_label_columns= gr.Textbox(label="Label Columns (comma-separated)", value="valence,arousal")

            apply_pca = gr.Checkbox(label="Apply PCA", value=False)
            pca_variance = gr.Number(label="PCA Variance", value=0.95)
            classifier = gr.Dropdown(label="Classifier", choices=["xgboost", "svm", "randomforest"], value="xgboost")
            #train_label = gr.Textbox(label="Label Columns (comma-separated)", value="valence,arousal")
            target = gr.Textbox(label="Target Label", value="arousal")

            complete_train_button = gr.Button("Complete Train Model")

            complete_train_output_pkl = gr.File(label="Download Model PKL")
            complete_train_output_json = gr.File(label="Download Model JSON")

            def complete_train_model(accel_file, report_file, time_window, label_columns, 
                                       cutoff_frequency, order, scaler_type, window_length, window_step_size, data_frequency, include_magnitude, features_label_columns,
                                        apply_pca, pca_variance, classifier, train_label, target):
                output_file, secondary_output_file = execute_complete_training_pipeline(accel_file, report_file, time_window, label_columns, 
                                       cutoff_frequency, order, scaler_type, window_length, window_step_size, data_frequency, include_magnitude, features_label_columns,
                                        apply_pca, pca_variance, classifier, train_label, target)
                return output_file, secondary_output_file

            complete_train_button.click(complete_train_model, inputs=[accel_file, report_file, time_window, label_columns, 
                                       cutoff_frequency, order, scaler_type, window_length, window_step_size, data_frequency, include_magnitude, features_label_columns,
                                        apply_pca, pca_variance, classifier, train_label, target], outputs=[complete_train_output_pkl, complete_train_output_json])


demo.launch()