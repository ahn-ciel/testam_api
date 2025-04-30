### testam_api
수요예측과 교통예측을 위한 api 

### testam: trafficflow-adj data 
- 특정 지역의 노드를 한번 만들어 두면 끝<br>
    - 이건 특별히 api(코드 모듈화) 까지 만들어둘 필요 없음.<br>
    - 생성하는 방법 담긴 .py <br>
- [주의점] adjdata생성할 시, “.pkl” 포맷으로 만들어야 함.<br>
<br>
1. dongjakITS_linkid_search_lon_lat.py<br>
동작구 its 기반 데이터로 링크 정보 (도로 구간) ="publicDB/kor_nodelink_info_2023_utf8.csv”, 노드 정보 (위도, 경도 포함)="publicDB/seoul_nodeID_info.csv” 필요함.<br>
nodelink와 nodeID을 매칭 시켜 .csv(linkID_nodeID_coordinates.csv)으로 저장하는 역할함<br>
→ 실행결과: <br>
생성됨 ”dongjakITS_linkID_nodeID_coordinates.csv”<br>
LINK_ID,F_NODE,F_NAME,F_X,F_Y,T_NODE,T_NAME,T_X,T_Y,ROAD_RANK,ROAD_NAME,ROAD_len(m)<br>
1020000101,1020008600,장미맨션,126.9674759,37.5192653,1020006700,동작대교북측,126.9800257,37.516052,102,강변북로,1177.352483<br>
<br>
2. dongjakITS_korLinkID_createAdjMetrix.py<br>
dongjakITS_config.json : 동작구에 대한 linkID 리스트 들이 있음. <br>
“kor_linkID_info_20250124_utf.csv” : <br>
<br>
<kor_linkID_info_20250124_utf.csv에서 사용되는 컬럼들><br>
LINK_ID	도로 링크의 고유 ID (도로 구간 식별자)  <br>
F_NODE	출발 노드 ID (해당 도로 구간의 시작점)<br>
T_NODE	도착 노드 ID (해당 도로 구간의 끝점)<br>
LANES	차선 수<br>
ROAD_RANK	도로 등급 (107=일반도로 등급)<br>
ROAD_TYPE	도로 유형 코드<br>
ROAD_NO	도로 번호 (고속도로 등)<br>
ROAD_NAME	도로명 (도로의 공식 이름)<br>
ROAD_USE	도로 사용 유형<br>
MULTI_LINK	다중 링크 여부 (0=아님, 1=다중 링크)<br>
CONNECT	연결 여부<br>
MAX_SPD	해당 도로의 최고 제한 속도 (단위: km/h)<br>
REST_VEH	차량 제한 여부 (0=제한 없음)<br>
REST_W	도로 제한 폭<br>
REST_H	도로 제한 높이<br>
C-ITS	차세대 지능형 교통 시스템 관련 정보<br>
LENGTH	해당 도로 구간의 길이 (단위: m)<br>
UPDATEDATE	데이터 업데이트 날짜<br>
REMARK	추가적인 설명<br>
HIST_TYPE	이력 유형<br>
HISTREMARK	이력 설명<br>
geometry	해당 도로의 위치 좌표(선형 정보, LINESTRING)<br>
<br>
df_link 에서 linkID별로 정보를 가져옴.  linkID은 도로 구간에 대한 정보이므로 linkID의 중앙값으로 adjdata을 생성하기 위함. df_link의 F_NODE, T_NODE 의 중앙값으로 adj_matrix을 생성함.<br>
→ 실행결과:<br>
linkID 개수만큼 .shape 생성<br>
[.pkl 생성] “dongjakITS_{NAME}_adjdata.pkl”<br>
“dongjakITS_v2_linkID_adjMatrix.csv”예시:<br>
LINK_ID,1020000101,1020000104,1020000201,1020000202,1020000301, ...<br>
1020000101,1.0,0.0,0.0,0.0,0.0, ...<br>
1020000104,0.0,1.0,0.0,0.342,0.077, ...<br>
1020000201,0.0,0.0,1.0,0.0,0.0, ...<br>
1020000202,0.0,0.342,0.0,1.0,0.022, ...<br>
1020000301,0.0,0.077,0.0,0.022,1.0, ...<br>
...<br>
→ 활용:<br>
“adjdata.pkl”을 특정 위치에 놓고 train/ inference_config.json 파일 경로에 작성해서 사용하면 됌<br>
<br>
### testam: trafficflow data

- 예시파일:<br>
     data/ciel/dongjakITS_2023_v2_0107_1231_cValue.csv    <br>
- “dongjakITS_2023_v2_0107_1231_cValue.csv” sample :<br>
| 헤더1 | 1020000101 | 1020000104 | 1020000201 |
|-------|-------|-------|-------|
| 2023-01-07 00:05:00 | 20 | 30 | 37 |
| 2023-01-07 00:15:00 | 49 | 30 | 37 |
| 2023-01-07 00:25:00 | 54 | 30 | 37 |
- df.index = Index(['2023-01-07 00:05:00', '2023-01-07 00:15:00', '2023-01-07 00:25:00', …], dtype='object', name='생성시분', length=51696)<br>
- tess.shape = [1, offset, num_node, values]<br>
  - values = trafficflow, sin시간, cos시간<br>
- 필요한 최소단위:<br>
  10분씩 6개 1시간으로 구성한다면, 5개 과거 시간+ 1개 현재 시간 필요함.<br>
- 1개 유닛 데이터 셋을 넣으려면 batch_size =offset 크기만 큼 넣어야 함.<br>
  예: 10분 단위 1시간 데이터 셋이면 batch_size = 6<br>
- trafficflow은 its 추출해서 온 시간대에 기반으로 0~1 값 안에서 분위수로 작성됨<br>
datetime_hms_list=['00:05:00', '00:15:00', '00:25:00', ...] <br>
ime_ind_list=[0.00347222, 0.01041667, 0.01736111, ...]<br>
<br>
→ generate_unit_data.py <br>
생성한 데이터: "data/dj_23T_v2_3_1h_cValue/test_unit.npz"<br>
<br>
### testam: OD data<br>
- 예시 : data/ciel/ciel-dj_P_3_1h<br>
tetime_hms_list= ['00:00:00', '00:10:00', '00:20:00', ...]<br>
time_ind_list= [0.0, 0.00694444, 0.01388889, 0.02083333, ...]<br>
