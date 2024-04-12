import os
import mlflow
import pandas as pd

from tasks.model import build_model, save_model, train_model, upload_model
from tasks.dataset import prepare_dataset, validate_data
# from tasks.deploy import (build_ref_data, save_and_upload_ref_data,
#                           build_drift_detectors, save_and_upload_drift_detectors)
from tasks.utils.tf_data_utils import AUGMENTER
from flows.utils import log_mlflow_info, build_and_log_mlflow_url
from prefect import flow, get_run_logger
from prefect.artifacts import create_link_artifact
from typing import Dict, Any

CENTRAL_STORAGE_PATH = os.getenv("CENTRAL_STORAGE_PATH", "/home/Jang/central_storage")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@flow(name='train_flow')
def train_flow(cfg: Dict[str, Any]):
    logger = get_run_logger()
    central_models_dir = os.path.join(CENTRAL_STORAGE_PATH, 'models')
    central_ref_data_dir = os.path.join(CENTRAL_STORAGE_PATH, 'ref_data')
    model_cfg = cfg['model']
    drift_cfg = model_cfg['drift_detection']
    mlflow_train_cfg = cfg['train']['mlflow']
    hparams = cfg['train']['hparams']
    ds_cfg = cfg['dataset']

    input_shape = (model_cfg['input_size']['h'], model_cfg['input_size']['w'])
    model = build_model(input_size=input_shape, n_classes=len(model_cfg['classes']),  
                        classifier_activation=model_cfg['classifier_activation'],
                        classification_layer=model_cfg['classification_layer'])   

    ds_repo_path, annotation_df = prepare_dataset(ds_root=ds_cfg['ds_root'], 
                                                  ds_name=ds_cfg['ds_name'], 
                                                  dvc_tag=ds_cfg['dvc_tag'], 
                                                  dvc_checkout=ds_cfg['dvc_checkout'])


    # data validation before using
    report_path = f"files/{ds_cfg['ds_name']}_{ds_cfg['dvc_tag']}_validation.html" # this must be .html
    validate_data(ds_repo_path, img_ext = 'jpeg', save_path=report_path)
    
    mlflow.set_experiment(mlflow_train_cfg["exp_name"])
    with mlflow.start_run(description=mlflow_train_cfg['exp_desc']) as train_run:
        log_mlflow_info(logger,train_run)
        mlflow_run_url = build_and_log_mlflow_url(logger,train_run)
        
        mlflow.set_tags(tags=mlflow_train_cfg['exp_tags'])
        mlflow.log_params(hparams)
        # for simplicity, I gonna save data validation report along with training task
        mlflow.log_artifact(report_path)
        trained_model = train_model(model, model_cfg['classes'], ds_repo_path, annotation_df,
                           img_size=input_shape, epochs=hparams['epochs'],
                           batch_size=hparams['batch_size'], init_lr=hparams['init_lr'],
                           augmenter=AUGMENTER)
        model_dir, metadata_file_path = save_model(trained_model, model_cfg)
        model_save_dir, metadata_file_name = upload_model(model_dir=model_dir, 
                                                      metadata_file_path=metadata_file_path,
                                                      remote_dir=central_models_dir)        
            

def start(cfg):
    train_flow(cfg)
        
        
    







