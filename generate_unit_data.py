from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import sys
from datetime import timedelta
#tafficflow
# datetime_hms_list=['00:05:00', '00:15:00', '00:25:00', '00:35:00', '00:45:00', '00:55:00', '01:05:00', '01:15:00', '01:25:00', '01:35:00', '01:45:00', '01:55:00', '02:05:00', '02:15:00', '02:25:00', '02:35:00', '02:45:00', '02:55:00', '03:05:00', '03:15:00', '03:25:00', '03:35:00', '03:45:00', '03:55:00', '04:05:00', '04:15:00', '04:25:00', '04:35:00', '04:45:00', '04:55:00', '05:05:00', '05:15:00', '05:25:00', '05:35:00', '05:45:00', '05:55:00', '06:05:00', '06:15:00', '06:25:00', '06:35:00', '06:45:00', '06:55:00', '07:05:00', '07:15:00', '07:25:00', '07:35:00', '07:45:00', '07:55:00', '08:05:00', '08:15:00', '08:25:00', '08:35:00', '08:45:00', '08:55:00', '09:05:00', '09:15:00', '09:25:00', '09:35:00', '09:45:00', '09:55:00', '10:05:00', '10:15:00', '10:25:00', '10:35:00', '10:45:00', '10:55:00', '11:05:00', '11:15:00', '11:25:00', '11:35:00', '11:45:00', '11:55:00', '12:05:00', '12:15:00', '12:25:00', '12:35:00', '12:45:00', '12:55:00', '13:05:00', '13:15:00', '13:25:00', '13:35:00', '13:45:00', '13:55:00', '14:05:00', '14:15:00', '14:25:00', '14:35:00', '14:45:00', '14:55:00', '15:05:00', '15:15:00', '15:25:00', '15:35:00', '15:45:00', '15:55:00', '16:05:00', '16:15:00', '16:25:00', '16:35:00', '16:45:00', '16:55:00', '17:05:00', '17:15:00', '17:25:00', '17:35:00', '17:45:00', '17:55:00', '18:05:00', '18:15:00', '18:25:00', '18:35:00', '18:45:00', '18:55:00', '19:05:00', '19:15:00', '19:25:00', '19:35:00', '19:45:00', '19:55:00', '20:05:00', '20:15:00', '20:25:00', '20:35:00', '20:45:00', '20:55:00', '21:05:00', '21:15:00', '21:25:00', '21:35:00', '21:45:00', '21:55:00', '22:05:00', '22:15:00', '22:25:00', '22:35:00', '22:45:00', '22:55:00', '23:05:00', '23:15:00', '23:25:00', '23:35:00', '23:45:00', '23:55:00']
# time_ind_list=[0.00347222, 0.01041667, 0.01736111, 0.02430556, 0.03125, 0.03819444, 0.04513889, 0.05208333, 0.05902778, 0.06597222, 0.07291667, 0.07986111, 0.08680556, 0.09375, 0.10069444, 0.10763889, 0.11458333, 0.12152778, 0.12847222, 0.13541667, 0.14236111, 0.14930556, 0.15625, 0.16319444, 0.17013889, 0.17708333, 0.18402778, 0.19097222, 0.19791667, 0.20486111, 0.21180556, 0.21875, 0.22569444, 0.23263889, 0.23958333, 0.24652778, 0.25347222, 0.26041667, 0.26736111, 0.27430556, 0.28125, 0.28819444, 0.29513889, 0.30208333, 0.30902778, 0.31597222, 0.32291667, 0.32986111, 0.33680556, 0.34375, 0.35069444, 0.35763889, 0.36458333, 0.37152778, 0.37847222, 0.38541667, 0.39236111, 0.39930556, 0.40625, 0.41319444, 0.42013889, 0.42708333, 0.43402778, 0.44097222, 0.44791667, 0.45486111, 0.46180556, 0.46875, 0.47569444, 0.48263889, 0.48958333, 0.49652778, 0.50347222, 0.51041667, 0.51736111, 0.52430556, 0.53125, 0.53819444, 0.54513889, 0.55208333, 0.55902778, 0.56597222, 0.57291667, 0.57986111, 0.58680556, 0.59375, 0.60069444, 0.60763889, 0.61458333, 0.62152778, 0.62847222, 0.63541667, 0.64236111, 0.64930556, 0.65625, 0.66319444, 0.67013889, 0.67708333, 0.68402778, 0.69097222, 0.69791667, 0.70486111, 0.71180556, 0.71875, 0.72569444, 0.73263889, 0.73958333, 0.74652778, 0.75347222, 0.76041667, 0.76736111, 0.77430556, 0.78125, 0.78819444, 0.79513889, 0.80208333, 0.80902778, 0.81597222, 0.82291667, 0.82986111, 0.83680556, 0.84375, 0.85069444, 0.85763889, 0.86458333, 0.87152778, 0.87847222, 0.88541667, 0.89236111, 0.89930556, 0.90625, 0.91319444, 0.92013889, 0.92708333, 0.93402778, 0.94097222, 0.94791667, 0.95486111, 0.96180556, 0.96875, 0.97569444, 0.98263889, 0.98958333, 0.99652778]

#demand
datetime_hms_list= ['00:00:00', '00:10:00', '00:20:00', '00:30:00', '00:40:00', '00:50:00', '01:00:00', '01:10:00', '01:20:00', '01:30:00', '01:40:00', '01:50:00', '02:00:00', '02:10:00', '02:20:00', '02:30:00', '02:40:00', '02:50:00', '03:00:00', '03:10:00', '03:20:00', '03:30:00', '03:40:00', '03:50:00', '04:00:00', '04:10:00', '04:20:00', '04:30:00', '04:40:00', '04:50:00', '05:00:00', '05:10:00', '05:20:00', '05:30:00', '05:40:00', '05:50:00', '06:00:00', '06:10:00', '06:20:00', '06:30:00', '06:40:00', '06:50:00', '07:00:00', '07:10:00', '07:20:00', '07:30:00', '07:40:00', '07:50:00', '08:00:00', '08:10:00', '08:20:00', '08:30:00', '08:40:00', '08:50:00', '09:00:00', '09:10:00', '09:20:00', '09:30:00', '09:40:00', '09:50:00', '10:00:00', '10:10:00', '10:20:00', '10:30:00', '10:40:00', '10:50:00', '11:00:00', '11:10:00', '11:20:00', '11:30:00', '11:40:00', '11:50:00', '12:00:00', '12:10:00', '12:20:00', '12:30:00', '12:40:00', '12:50:00', '13:00:00', '13:10:00', '13:20:00', '13:30:00', '13:40:00', '13:50:00', '14:00:00', '14:10:00', '14:20:00', '14:30:00', '14:40:00', '14:50:00', '15:00:00', '15:10:00', '15:20:00', '15:30:00', '15:40:00', '15:50:00', '16:00:00', '16:10:00', '16:20:00', '16:30:00', '16:40:00', '16:50:00', '17:00:00', '17:10:00', '17:20:00', '17:30:00', '17:40:00', '17:50:00', '18:00:00', '18:10:00', '18:20:00', '18:30:00', '18:40:00', '18:50:00', '19:00:00', '19:10:00', '19:20:00', '19:30:00', '19:40:00', '19:50:00', '20:00:00', '20:10:00', '20:20:00', '20:30:00', '20:40:00', '20:50:00', '21:00:00', '21:10:00', '21:20:00', '21:30:00', '21:40:00', '21:50:00', '22:00:00', '22:10:00', '22:20:00', '22:30:00', '22:40:00', '22:50:00', '23:00:00', '23:10:00', '23:20:00', '23:30:00', '23:40:00', '23:50:00']
time_ind_list= [0.0, 0.00694444, 0.01388889, 0.02083333, 0.02777778, 0.03472222, 0.04166667, 0.04861111, 0.05555556, 0.0625, 0.06944444, 0.07638889, 0.08333333, 0.09027778, 0.09722222, 0.10416667, 0.11111111, 0.11805556, 0.125, 0.13194444, 0.13888889, 0.14583333, 0.15277778, 0.15972222, 0.16666667, 0.17361111, 0.18055556, 0.1875, 0.19444444, 0.20138889, 0.20833333, 0.21527778, 0.22222222, 0.22916667, 0.23611111, 0.24305556, 0.25, 0.25694444, 0.26388889, 0.27083333, 0.27777778, 0.28472222, 0.29166667, 0.29861111, 0.30555556, 0.3125, 0.31944444, 0.32638889, 0.33333333, 0.34027778, 0.34722222, 0.35416667, 0.36111111, 0.36805556, 0.375, 0.38194444, 0.38888889, 0.39583333, 0.40277778, 0.40972222, 0.41666667, 0.42361111, 0.43055556, 0.4375, 0.44444444, 0.45138889, 0.45833333, 0.46527778, 0.47222222, 0.47916667, 0.48611111, 0.49305556, 0.5, 0.50694444, 0.51388889, 0.52083333, 0.52777778, 0.53472222, 0.54166667, 0.54861111, 0.55555556, 0.5625, 0.56944444, 0.57638889, 0.58333333, 0.59027778, 0.59722222, 0.60416667, 0.61111111, 0.61805556, 0.625, 0.63194444, 0.63888889, 0.64583333, 0.65277778, 0.65972222, 0.66666667, 0.67361111, 0.68055556, 0.6875, 0.69444444, 0.70138889, 0.70833333, 0.71527778, 0.72222222, 0.72916667, 0.73611111, 0.74305556, 0.75, 0.75694444, 0.76388889, 0.77083333, 0.77777778, 0.78472222, 0.79166667, 0.79861111, 0.80555556, 0.8125, 0.81944444, 0.82638889, 0.83333333, 0.84027778, 0.84722222, 0.85416667, 0.86111111, 0.86805556, 0.875, 0.88194444, 0.88888889, 0.89583333, 0.90277778, 0.90972222, 0.91666667, 0.92361111, 0.93055556, 0.9375, 0.94444444, 0.95138889, 0.95833333, 0.96527778, 0.97222222, 0.97916667, 0.98611111, 0.99305556]

def generate_next_datetime_indices(df, num_steps=5):
    """
    df의 마지막 index 이후로 datetime_hms 기준으로 num_steps만큼의 datetime 인덱스를 생성합니다.
    다음 날로 넘어가야 할 경우 자동으로 날짜를 하루 증가시켜 생성합니다.

    Parameters:
        df (pd.DataFrame): datetime index를 가진 DataFrame
        datetime_hms (list of str): 'HH:MM:SS' 형식의 시간 리스트 (24시간 기준, 정렬된 상태)
        num_steps (int): 생성할 인덱스 수 (기본값 5)

    Returns:
        list of str: 새로 생성된 full datetime 인덱스 리스트
    """
    # 마지막 인덱스 가져오기
    last_index = pd.to_datetime(df.index[-1])
    last_time_str = last_index.strftime("%H:%M:%S")
    last_date = last_index.date()
    last_date_str = last_index.strftime("%Y-%m-%d")

    # 기준 시간 존재 여부 확인
    if last_time_str not in datetime_hms_list:
        raise ValueError(f"{last_time_str} not found in datetime_hms")

    # 인덱스 위치 찾기
    start_idx = datetime_hms_list.index(last_time_str)
    remaining_times = datetime_hms_list[start_idx + 1:]
    additional_count = num_steps - len(remaining_times)

    # 현재 날짜 기준 추가
    new_datetime_index = [f"{last_date_str} {t}" for t in remaining_times[:num_steps]]

    # 다음날로 넘어가야 하는 경우
    if additional_count > 0:
        next_day = last_date + timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")
        next_day_times = datetime_hms_list[:additional_count]
        new_datetime_index += [f"{next_day_str} {t}" for t in next_day_times]

    return new_datetime_index


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_period_in_day=True, 
        add_day_in_week=True, add_date=True, min_interval=10,scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    
    예) METR-LA 
    24시간 x (60분 /5분) = 288
    x = (epoch_size, 12, 207, 2)
    y = (epoch_size, 12, 207, 2)
    1) traffic speed : 도로에서 차량 흐름과 관련된 실제 값. 평균 차량 속도
    2) time of day을 비율로 나타낸것 : 0~1 까지 값으로 표현. 하루를 288으로 나눈 값. 시간정보를 정규화.
    """
    """
    np.tile(time_ind, [1, num_nodes, 1]) : 두 번째 축에서 num_nodes번 반복하여 배열을 확장
    """
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    if add_time_in_day:
        try:
            df.index = pd.to_datetime(df.index) # for EXPY-TKY
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        except:
            time_ind = (df.index.values%(60*24/min_interval))/(60*24/min_interval) # for EXPY-TKY
            # time_ind = (df.index.values%288)/288
        
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
    if add_period_in_day:       
        """time_ind 대신에 sin_ind, cos_ind 넣기
        """
        try:
            df.index = pd.to_datetime(df.index) # for EXPY-TKY
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        except:
            time_ind = (df.index.values%(60*24/min_interval))/(60*24/min_interval) # for EXPY-TKY
            # time_ind = (df.index.values%288)/288

        print("time_index:", len(time_ind), time_ind[0], time_ind[-1])
        # sin, cos 값 계산
        sin_ind = np.around(np.sin(2 * np.pi * time_ind), 4)
        cos_ind = np.around(np.cos(2 * np.pi * time_ind), 4)

        for ind in [sin_ind, cos_ind]:
            period_in_day = np.tile(ind, [1, num_nodes, 1]).transpose((2, 1, 0))
            # feature_list.insert(0, period_in_day)        
            feature_list.append(period_in_day)
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)
    if add_date:
        realtime = df.index.values # for EXPY-TKY
        formatted_realtime = pd.to_datetime(realtime).strftime("%Y-%m-%d %H:%M:%S")
        date = np.tile(formatted_realtime, [1, num_nodes, 1]).transpose((2, 1, 0))
        # feature_list.append(date)
    
    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    realday =[]
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
        realday.append(date[t + y_offsets])
    # x = np.stack(x, axis=0)
    # y = np.stack(y, axis=0)
    # realday = np.stack(realday, axis=0)
    # ahn
    # if x[0].ndim ==3:
    #     x[0] = np.expand_dims(x[0], axis=0)  
    # if y[0].ndim ==3:
    #     y[0] = np.expand_dims(y[0], axis=0)  
    # if realday[0].ndim ==3:
    #     realday[0] = np.expand_dims(realday[0], axis=0)  
    return x[0], y[0], realday[0]
    # return x, y, realday


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y

    # df = pd.read_csv(args.traffic_df_filename, index_col = 0, parse_dates=True)
    # df = pd.read_csv(args.traffic_df_filename, index_col = 0)
    df_stack = pd.read_csv(args.traffic_df_stack)
    df_cur = pd.read_csv(args.traffic_df_cur)

    # 컬럼이름이 순서와 이름이 같을때만 y
    if list(df_stack.columns) == list(df_cur.columns):
        idx_name=df_stack.columns[0]
        df = pd.concat([df_stack, df_cur], axis=0, ignore_index=True)
        df.set_index(idx_name, inplace=True)
    else:
        print("Column names do not match.")
        print("df_stack columns:", list(df_stack.columns))
        print("df_cur columns:", list(df_cur.columns))   
        sys.exit(1)
    #ahn:
    new_times = generate_next_datetime_indices(df, num_steps=10)
    print("===> 생성된 인덱스:", new_times)
    new_df = pd.DataFrame(data=0, index=new_times, columns=df.columns)
    df_ = pd.concat([df, new_df])
    
    
    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y, realtime = generate_graph_seq2seq_io_data(
        df_,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=args.add_time_in_day,
        add_period_in_day=args.add_period_in_day,
        add_day_in_week=args.add_day_in_week,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape, "realday shape:", realtime.shape)

    x_test, y_test = x, y
    realtime_test = realtime

    for cat in ["test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _realtime = locals()["realtime_"+ cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape, "realtime:", _realtime.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}_unit.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
            realtime=_realtime
        )


if __name__ == "__main__":
    # sys.argv = ['generate_training_data.py', '--output_dir','data/EXPY-TKY', '--traffic_df_filename','data/EXPYTKY/expy-tky_202110.csv']
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/ciel-dj_P_1_1h", help="Output directory.")
    parser.add_argument("--traffic_df_stack", type=str, default="data/ciel/ciel-dj_P_1_1h_test_stack1.csv", help="Raw traffic readings.",)
    parser.add_argument("--traffic_df_cur", type=str, default="data/ciel/ciel-dj_P_1_1h_test_cur1.csv", help="Raw traffic readings.",)
    parser.add_argument("--add_time_in_day", type=bool, default=False)
    parser.add_argument("--add_period_in_day", type=bool, default=True)
    parser.add_argument("--add_day_in_week", type=bool, default=False)
    parser.add_argument("--seq_length_x", type=int, default=6, help="Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=6, help="Sequence Length.",)
    parser.add_argument("--y_start", type=int, default=1, help="Y pred start", )

    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
        if reply[0] != 'y': exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args)
