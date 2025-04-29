from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from train_demand  import load_args_from_json, create_save_dir, main, setup_logger
from typing import List
from fastapi import Query
from fastapi.encoders import jsonable_encoder
import numpy as np
import os
import datetime

# app = FastAPI()
app = FastAPI(default_response_class=ORJSONResponse)

# 앱 시작 시 초기화
# @app.on_event("startup")
# def startup_event():
#     config_path = "/TESTAM/config_train_demand.json"
#     args = load_args_from_json(config_path)

#     # 로그 디렉토리 및 파일 설정
#     log_dir = "/TESTAM/logs"
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(
#         log_dir, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
#     )
#     setup_logger(log_file)

#     app.state.args = args
#     app.state.log_file = log_file

@app.get("/train-demand1", response_class=ORJSONResponse)
def train_with_config():
    
    args = load_args_from_json("/TESTAM/config_train_demand.json")
    create_save_dir(args)
        
    result = main(args)
    return jsonable_encoder(result, custom_encoder={np.ndarray: lambda x: x.tolist()})
