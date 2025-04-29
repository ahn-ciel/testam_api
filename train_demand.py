import torch
import numpy as np
import time, os
import util
import json
from types import SimpleNamespace
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

def evaluate_dimension(yhat_np, realy_np, axis, label, check=None, zero_threshold=1e-5):
    if check is not None and check != realy_np.shape[axis]:
        logging.warning(
            f"[Warning] {label} ({check}) != realy_np.shape[{axis}] ({realy_np.shape[axis]}), "
            f"so {label} evaluation was skipped.")
        return 
    
    amae, amape, armse = [], [], []
    dim = yhat_np.shape[axis]
    
    for i in range(dim):
        if axis == 1:  # outdim
            pred = yhat_np[:, i, :, :]
            real = realy_np[:, i, :, :]
        elif axis == 2:  # node
            pred = yhat_np[:, :, i, :]
            real = realy_np[:, :, i, :]
        elif axis == 3 or axis == -1:  # offset
            pred = yhat_np[..., i]
            real = realy_np[..., i]
        else:
            # raise ValueError("Unsupported axis")
            logging.warning("Unsupported axis encountered, skipping...")

        # 0값 마스킹 처리 
        mask = np.abs(real) > zero_threshold
        if not np.any(mask):
            logging.warning(f"{label} {i + 1}: Skipped due to all real values near zero.")
            continue

        pred_masked = pred[mask]
        real_masked = real[mask]     
            
        metrics = util.metric(torch.tensor(pred_masked), torch.tensor(real_masked))
        logging.info(f"{label} {i + 1}: MAE={metrics[0]:.4f}, MAPE={metrics[1]:.4f}, RMSE={metrics[2]:.4f}")
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
    
    logging.info(f"Avg over {dim} {label}s - MAE: {np.mean(amae):.4f}, MAPE: {np.mean(amape):.4f}, RMSE: {np.mean(armse):.4f}")

def create_save_dir(args):
    if os.path.exists(os.path.dirname(args.save)):
        reply = str(input(f'{os.path.dirname(args.save)} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit()
    else:
        os.makedirs(os.path.dirname(args.save))
                
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    # Setup logger to log into a file and the console
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    log_file = args.save+ f'_train_{today}_{args.num_nodes}_log.txt'
    setup_logger(log_file)

    #load data
    device = torch.device(args.device)
    supports = None
            
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']

    logging.info(args)
    logging.info(f"train data scaler mean, std : {scaler.mean, scaler.std}")


    engine = trainer(scaler, args.in_dim, args.out_dim, args.num_nodes, args.nhid, args.dropout,
                         device, quantile=args.quantile, is_quantile=args.is_quantile, supports = supports)

    logging.info(f"Train the model with {count_parameters(engine.model)} parameters")


    if args.load_path:
        engine.model.load_state_dict(torch.load(args.load_path, map_location=device))
        engine.model.to(device)
        logging.info(f"model loaded sucessfully! {args.load_path}")

    logging.info("start training...")
    
    his_loss =[]
    val_time = []
    train_time = []
    wait = 0
    best = 1e9
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,:args.out_dim,:,:], i)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log_msg = f"Iter: {iter:03d}, Train Loss: {train_loss[-1]:.4f}, Train MAPE: {train_mape[-1]:.4f}, Train RMSE: {train_rmse[-1]:.4f}"
                logging.info(log_msg)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.eval(testx, testy[:,:args.out_dim,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])

        s2 = time.time()
        logging.info(f"Epoch: {i:03d}, Inference Time: {s2 - s1:.4f} secs")
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        logging.info(f"Epoch: {i:03d}, Train Loss: {mtrain_loss:.4f}, Train MAPE: {mtrain_mape:.4f}, Train RMSE: {mtrain_rmse:.4f}, "
                     f"Valid Loss: {mvalid_loss:.4f}, Valid MAPE: {mvalid_mape:.4f}, Valid RMSE: {mvalid_rmse:.4f}, "
                     f"Training Time: {t2 - t1:.4f}/epoch")
        if best > his_loss[-1]:
            best = his_loss[-1]
            wait = 0
            torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
        else:
            wait = wait + 1
        if wait > args.patience:
            logging.info("Early Termination!")
            break
    logging.info(f"Average Training Time: {np.mean(train_time):.4f} secs/epoch")
    logging.info(f"Average Inference Time: {np.mean(val_time):.4f} secs")

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict( torch.load(
        args.save + f"_epoch_{bestid+1}_{round(his_loss[bestid],2)}.pth", weights_only=True))


    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,:args.out_dim,:,:]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        with torch.no_grad():
            preds = engine.model(testx)
        outputs.append(preds)

    yhat = torch.cat(outputs,dim=0)[:realy.size(0),...]

    logging.info(f"Training finished. The valid loss on the best model is {round(his_loss[bestid], 4)}.")

    # 역변환 (한 번에)
    yhat = scaler.inverse_transform(yhat)
    realy = realy
    # NumPy로 변환 (GPU → CPU)
    yhat_np = yhat.cpu().numpy()
    realy_np = realy.cpu().numpy() 
    # 결과 저장용
    results = {
        'prediction': yhat_np.transpose(0, 3, 2, 1), # (N, offset, node, values)
        'ground_truth': realy_np.transpose(0, 3, 2, 1),
        'scaler': np.array([scaler.mean, scaler.std])
    }  
    # 평가 조건에 따라 수행
    if args.eval_outdim:       
        evaluate_dimension(yhat_np, realy_np, axis=1, label="outdim", check=args.out_dim)
    if args.eval_nodes:
        evaluate_dimension(yhat_np, realy_np, axis=2, label="node", check=args.num_nodes)
    if args.eval_offset:
        evaluate_dimension(yhat_np, realy_np, axis=3, label="horizon")
        
    # 공통
    save_eval_path = args.save+f"_exp{args.expid}_prediction.npz"
    np.savez_compressed(args.save+f"_exp{args.expid}_prediction.npz", **results)
    save_model_path = args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth"
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    api_response={
        "run_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message1": f"Trained model is stored : {save_model_path}",
        "message2": f"Results of given test data have been saved in .npz format. File path: {save_eval_path}"
    }
    return api_response

if __name__ == "__main__":
    t1 = time.time()
    config_path = '/TESTAM/config_train_demand.json' 
    args = load_args_from_json(config_path)
    
    create_save_dir(args)
    
    api_response = main(args)
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
