import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cdist
import json
import pickle

# 1479 ea - type: int 
NAME='v2'
# dongjakITS_config.json : linkID_all, linkID_v1, linkID_v2 기록
with open("cielDB/dongjakITS/dongjakITS_config.json", "r", encoding="utf-8") as f:
    loaded_json = json.load(f)
link_ids=loaded_json[f"linkID_{NAME}"]
  
df_link = pd.read_csv("publicDB/kor_linkID_info_20250124_utf.csv", encoding="utf-8")

# LINESTRING에서 X, Y 좌표 추출 함수
def extract_coordinates(linestring):
    coords = re.findall(r"([\d\.]+) ([\d\.]+)", linestring)
    return [[float(x), float(y)] for x, y in coords]

results = []
for link in link_ids:
    # 링크 정보에서 해당 LINK_ID의 row 찾기
    link_row = df_link[df_link["LINK_ID"] == int(link)]  # link_id가 문자열일 경우 int 변환
    
    if not link_row.empty:
        # F_NODE, T_NODE 값 가져오기
        f_node= link_row.iloc[0]['F_NODE']
        t_node= link_row.iloc[0]['T_NODE']
        road_name= link_row.iloc[0]['ROAD_NAME']
        road_rank= link_row.iloc[0]['ROAD_RANK']
        link_speed= link_row.iloc[0]['MAX_SPD']
        road_len= link_row.iloc[0]['LENGTH']
        geometry= link_row.iloc[0]['geometry']
        
        extracted_coords = extract_coordinates(geometry)
        coords = np.array(extracted_coords)
        x_max, y_max =round(np.max(coords[:,0]),3), round(np.max(coords[:,1]),3)
        x_min, y_min =round(np.min(coords[:,0]),3), round(np.min(coords[:,1]),3)
        x_mid, y_mid =round((x_max+x_min)/2,3), round((y_max+y_min)/2,3)
        all=[link, f_node, t_node, link_speed, road_rank, road_name, road_len, x_mid, y_mid, x_min, y_min, x_max, y_max]
    results.append(all)
        
columns=["LINK_ID","F_NODE","T_NODE","LINK_SPD","ROAD_RANK","ROAD_NAME","ROAD_LEN","LINK_X","LINK_Y","F_X","F_Y","T_X","T_Y"] # 13ea
df_arg = pd.DataFrame(results, columns=columns) 

df_arg.to_csv(f"cielDB/dongjakITS/dongjakITS_{NAME}_linkID_geometry.csv", index=False)

# 유클리드 거리 행렬 계산
sensor_locations = df_arg[["LINK_X", "LINK_Y"]].values
distance_matrix = cdist(sensor_locations, sensor_locations, metric='euclidean')

# Gaussian Kernel을 적용한 인접 행렬 계산 (특정 거리 이상일 때 0 처리)
sigma = 100  # 적절한 거리 스케일 파라미터 설정
threshold_distance = 500  # 특정 거리 이상이면 연결 없음 (0 처리)

# 가우시안 커널 계산, 특정 거리 이상이면 0으로 설정
adjacency_matrix = np.exp(- (distance_matrix ** 2) / sigma ** 2)
adjacency_matrix = np.round(adjacency_matrix, 3)
adjacency_matrix[distance_matrix > threshold_distance] = 0

# 인접 행렬을 DataFrame으로 변환
adj_matrix_df = pd.DataFrame(adjacency_matrix, index=df_arg["LINK_ID"], columns=df_arg["LINK_ID"])
# adj_matrix = adj_matrix_df.to_numpy()

# 결과 출력 및 저장
print("Adjacency Matrix Shape:", adjacency_matrix.shape)
print("Adjacency Matrix 2nd_max, 2nd_min:", np.unique(adjacency_matrix)[-2], np.unique(adjacency_matrix)[1])
# adj_matrix_df.to_csv(f"cielDB/dongjakITS/dongjakITS_{NAME}_linkID_adjMatrix.csv")

print(adjacency_matrix.shape, type(adjacency_matrix))
print(adjacency_matrix[:10,:10])

### npz 파일로 저장
# 1. 센서 ID 리스트
# 2. 센서 ID를 인덱스로 매핑하는 딕셔너리
# 3. 인접 행렬 (Adjacency Matrix)을 npz 파일로 저장
sensor_ids = df_arg["LINK_ID"]
sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
# np.savez_compressed(f"cielDB/dongjakITS/dongjakITS_{NAME}_adjdata.npz", sensor_ids=sensor_ids, sensor_id_to_ind=sensor_id_to_ind, adj_mx=adjacency_matrix)
# print(f"dongjakITS_{NAME}_adjdata.npz 파일이 생성되었습니다!")
# 3. 인접 행렬 (Adjacency Matrix)을 pkl 파일로 저장
pkl_filename = f"cielDB/dongjakITS/dongjakITS_{NAME}_adjdata.pkl"
with open(pkl_filename, "wb") as f:
    pickle.dump((sensor_ids, sensor_id_to_ind, adjacency_matrix), f)
print(f"dongjakITS_{NAME}_adjdata.pkl 파일이 생성되었습니다!")