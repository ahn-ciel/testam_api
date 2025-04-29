import torch
import numpy as np
import os
from util import *
import json
from types import SimpleNamespace
# from engine import trainer
from engine_for_adj import trainer
import datetime
import logging

def load_args_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    value_name = data.get("VALUE", "")
    # "VALUE" 문자열을 모두 value_name으로 치환
    def replace_value(obj):
        if isinstance(obj, str):
            return obj.replace("VALUE", value_name)
        elif isinstance(obj, dict):
            return {k: replace_value(v) for k, v in obj.items() if k != "VALUE"}
        elif isinstance(obj, list):
            return [replace_value(i) for i in obj]
        else:
            return obj

    replaced_data = replace_value(data)
    return SimpleNamespace(**replaced_data)

def setup_logger(log_file):
    # Setup logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s]: %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()         # Log to console
        ]
    )
    
def compare_indata_predictdata(prediction):
    out = prediction
    
    actual_data = out['ground_truth'].transpose(1, 0, 3, 2) # 실제 값 (10419, 12, 325, 2)
    predicted_data = out['prediction'].transpose(1, 0, 3, 2) # 예측 값 (10419, 12, 325)
    
    for i in range(predicted_data.shape[-1]):
        if args.values is not None and i < args.values:
            predic = predicted_data[:,:,:,i]
            realy = actual_data[:,:,:,i]
        else:
            predic = predicted_data[:,:,:,i]
            realy = actual_data[:,:,:,i]
        metrics = metric(torch.tensor(predic), torch.tensor(realy))
        print(f"Evaluate best model {i} columns - mae: {metrics[0]:.4f}, mape: {metrics[1]:.4}, rmse: {metrics[2]:.4}")
    
        absolute_error = np.abs(realy - predic)
        mae = np.mean(absolute_error)
        mse = np.mean((realy - predic) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((realy - predic) / realy)) * 100
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")

# 0. 엔진 준비
def initialize_engine(args, device, supports):
    
    # Initialize engine for inference
    engine = trainer(args.scaler, args.in_dim, args.out_dim, args.num_nodes, args.nhid, dropout=0., device=device, supports = supports)
    # Load the trained model
    if not args.load_path or not os.path.exists(args.load_path):
        # raise ValueError(f"Model path '{args.load_path}' is invalid or does not exist.")
        logging.warning(f"[Warning] Model path '{args.load_path}' is invalid or does not exist. Proceeding without loading weights.")
    
    engine.model.load_state_dict(torch.load(args.load_path, map_location=device, weights_only=True))
    engine.model.eval()    
    return engine  

# 1. 모델만 로드
def setup_model(args):
    device = torch.device(args.device)

    supports = None
    if args.adjdata and os.path.exists(args.adjdata):
        sensor_ids, _, adj_mx = load_adj(args.adjdata, args.adjtype)
        args.num_nodes = len(sensor_ids)
        supports = [torch.tensor(sup).to(device) for sup in adj_mx]

    engine = initialize_engine(args, device=device, supports=supports)
    print("engine.scaler:",engine.scaler)
    return engine, supports, device

# 2. 데이터만 로드하는 함수
def load_data(args):
    dataloader = load_single_dataset(args.data, args.batch_size, args.scaler)
    scaler = dataloader["scaler"]
    return dataloader, scaler     
       
def prediction(args, engine, dataloader, scaler, device):

    # Inference
    outputs = []
    y_tensor = torch.Tensor(dataloader['y']).to(device)
    # dataloader['y']가 3 dim일 경우 -> 4dim 
    if y_tensor.dim() == 3:
        y_tensor = y_tensor.unsqueeze(0)  # (1, offset, N, V)
    realy = y_tensor.transpose(1, 3)[:, :args.out_dim, :, :]

    for iter, (x, y) in enumerate(dataloader['data_loader'].get_iterator()):
        print("----0. x",x.shape)
        x_tensor = torch.Tensor(x).to(device)
        # dataloader['x']가 3 dim일 경우 -> 4dim 
        if x_tensor.dim() == 3:
            x_tensor = x_tensor.unsqueeze(0)  # (1, offset, N, V)
        testx = x_tensor.transpose(1, 3)
        print("----1. testx",testx.shape)
        with torch.no_grad():
            preds, gate, ind_out = engine.model(testx, gate_out=True)
        outputs.append(preds)

    # Processing outputs
    yhat = torch.cat(outputs, dim=0)[:realy.size(0), ...]
    
    # 역변환 (한 번에)
    yhat = scaler.inverse_transform(yhat) 
    # 먼저 전체 예측과 정답을 numpy로 옮기고 벡터 연산
    yhat_np = yhat.cpu().numpy()       # shape: (N, out_dim, node, offset)
    realy_np = realy.cpu().numpy()
    print("----2. yhat_np", yhat_np.shape)
    # # dataloader['realtime']가 3 dim일 경우 -> 4dim 
    if dataloader['realtime'].ndim ==3:
        dataloader['realtime'] = np.expand_dims(dataloader['realtime'], axis=0)  # (1, offset, node, value)
    
    # post-process 
    pred = np.clip(yhat_np, 0, None)
    pred = np.rint(pred).astype(int)
    real = np.rint(realy_np).astype(int)
    
    results = {
        "prediction": pred.transpose(0, 3, 2, 1),    # (N, offset, node, values)
        "ground_truth": real.transpose(0, 3, 2, 1)
    }
        
    print(np.asarray(results['prediction']).shape, np.asarray(results['ground_truth']).shape)
    print(dataloader["realtime"].shape)
    # Final results saving
    date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    api_response = {
        "run_time" : date_now,
        "timestamps" : dataloader["realtime"][0,:2,0,:],
        "prediction": results['prediction'][0,:2,0,:],  
        "ground_truth": results['ground_truth'][0,:2,0,:],
    }        

    return api_response


if __name__ == "__main__":
    import time
    t1 = time.time()
    
    # craate logfile
    log_file = f"/TESTAM/logs/inferTrafficflow_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(log_file)
        
    config_path = '/TESTAM/config_trafficflow.json' 
    args = load_args_from_json(config_path)
    # 모델 셋업
    engine, supports, device = setup_model(args)
    # 데이터 로드 
    dataloader, scaler_obj =load_data(args)
    # 추론
    api_response = prediction(args, engine, dataloader, scaler_obj, device)
    t2 = time.time()
    print(f"test2: Total inference time: {t2 - t1:.4f} seconds")
    print(api_response)
    