from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from train_trafficflow  import load_args_from_json, create_save_dir, main, setup_logger
from typing import List
from fastapi import Query
from fastapi.encoders import jsonable_encoder
import copy
import numpy as np
import os
import datetime

# app = FastAPI()
app = FastAPI(default_response_class=ORJSONResponse)

# 앱 시작 시 초기화
# @app.on_event("startup")
# def startup_event():
#     config_path = "/TESTAM/config_train_trafficflow.json"
#     args = load_args_from_json(config_path)

#     # 서버 전역에 저장
#     app.state.args = args

@app.get("/train-trafficflow1", response_class=ORJSONResponse)
def train_with_config():
    
    args = load_args_from_json("/TESTAM/config_train_trafficflow.json")
    create_save_dir(args)
    result = main(args)
    
    return jsonable_encoder(result, custom_encoder={np.ndarray: lambda x: x.tolist()})

