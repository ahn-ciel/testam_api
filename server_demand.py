from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from inference_trafficflow  import prediction, load_args_from_json, setup_model, load_data, setup_logger
from typing import List
from fastapi import Query
from fastapi.encoders import jsonable_encoder
import copy
import numpy as np
import datetime
import os

# app = FastAPI()
app = FastAPI(default_response_class=ORJSONResponse)

# 앱 시작 시 초기화
@app.on_event("startup")
def startup_event():
    # 로그 디렉토리 생성
    log_dir = "/TESTAM/logs"
    os.makedirs(log_dir, exist_ok=True)

    # 로그 파일 경로 생성
    log_file = os.path.join(
        log_dir, f"inference_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logger(log_file)  # 로깅 초기화
        
    config_path = "/TESTAM/config_demand.json"
    args = load_args_from_json(config_path)

    # 모델 로드
    engine, supports, device = setup_model(args)

    # 서버 전역에 저장
    app.state.args = args
    app.state.engine = engine
    app.state.supports = supports
    app.state.device = device

# 개발 할때 편의 위해
@app.get("/predict-demand1", response_class=ORJSONResponse)
def predict_demand_base_config():
    # 매번 로드
    args = load_args_from_json("/TESTAM/config_demand.json")  
    dataloader, scaler_obj = load_data(args)
    result = prediction(args, app.state.engine, dataloader, scaler_obj, app.state.device)
    return jsonable_encoder(result, custom_encoder={np.ndarray: lambda x: x.tolist()})

# 운영 할때 편의 위해
@app.get("/predict-demand2", response_class=ORJSONResponse)
def predict_demand_base_parameters(data_path: str, scaler: List[float] = Query(...)):
    # 1. 기존 상태 가져오기
    args_base = app.state.args
    engine = app.state.engine
    device = app.state.device

    # 2. args 복사 후 값만 변경
    args = copy.deepcopy(args_base)
    args.data = data_path
    args.scaler = scaler
    
    dataloader, scaler_obj =load_data(args)
    # engine 재사용해서 추론 실행
    result = prediction(args, app.state.engine, dataloader, scaler_obj, app.state.device)
    return jsonable_encoder(result, custom_encoder={np.ndarray: lambda x: x.tolist()})    
# 실행명령어: 
# http://localhost:8000/predict-trafficflow?data_path=data/dj_23T_v2_3_1h_cValue/test_zero.npz&scaler=23.5462007&scaler=13.0605837