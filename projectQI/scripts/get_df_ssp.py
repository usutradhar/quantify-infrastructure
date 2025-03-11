# Prepare df_ssp for plots and stats
'''
this file import per capita RBUV and RL for all three scenarios, mergeee them, add populaion attributes, 
create percent change columns and output a dataframe

'''
# import all packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



def label_burden(row, col1, col2):
    if np.round(row[col1], 4) < np.round(row[col2],4):
        return 'increasingBurden'
    elif np.round(row[col1],4) > np.round(row[col2],4):
        return 'decreasingBurden'
    elif np.round(row[col1],4) == np.round(row[col2],4):
        return 'noChange'
    else:
        return 'error'
    


def get_df_ssp(filepath =r'E:\Scripts\project_QI\outputfiles\csvs\\'):
    df_roads_ssp1 = pd.read_csv(filepath + 'roads_perCap_ssp1.csv', index_col = 0)
    df_roads_ssp1['GEOID'] = df_roads_ssp1['GEOID'].astype('str').str.rjust(7, '0')

    df_buildings_ssp1 = pd.read_csv(filepath + 'buildings_perCap_ssp1.csv', index_col = 0)
    df_buildings_ssp1['GEOID'] = df_buildings_ssp1['GEOID'].astype('str').str.rjust(7, '0')

    df_roads_ssp2 = pd.read_csv(r'E:\Scripts\project_QI\outputfiles\csvs\roads_perCap_ssp2_x_local.csv', index_col = 0)
    df_roads_ssp2['GEOID'] = df_roads_ssp2['GEOID'].astype('str').str.rjust(7, '0')

    df_buildings_ssp2 = pd.read_csv(r'E:\Scripts\project_QI\outputfiles\csvs\buildings_perCap_ssp2.csv', index_col = 0)
    df_buildings_ssp2['GEOID'] = df_buildings_ssp2['GEOID'].astype('str').str.rjust(7, '0')

    df_roads_ssp4 = pd.read_csv(filepath + 'roads_perCap_ssp4.csv', index_col = 0)
    df_roads_ssp4['GEOID'] = df_roads_ssp4['GEOID'].astype('str').str.rjust(7, '0')

    df_buildings_ssp4 = pd.read_csv(filepath + 'buildings_perCap_ssp4.csv', index_col = 0)
    df_buildings_ssp4['GEOID'] = df_buildings_ssp4['GEOID'].astype('str').str.rjust(7, '0')

    print(df_buildings_ssp1.shape, df_buildings_ssp2.shape, df_buildings_ssp4.shape, df_roads_ssp1.shape, df_roads_ssp2.shape, df_roads_ssp4.shape)

    # rename columns
    length_cols = ['GEOID', 'length_m_perCap_2020', 'length_m_perCap_2030','length_m_perCap_2040', 'length_m_perCap_2050', 'length_m_perCap_2060',
               'length_m_perCap_2070', 'length_m_perCap_2080', 'length_m_perCap_2090', 'length_m_perCap_2100']
    built_cols =['GEOID', 'volume_m3_perCap_2020', 'volume_m3_perCap_2030', 'volume_m3_perCap_2040','volume_m3_perCap_2050', 'volume_m3_perCap_2060',
                'volume_m3_perCap_2070', 'volume_m3_perCap_2080', 'volume_m3_perCap_2090', 'volume_m3_perCap_2100']

    df_roads = df_roads_ssp1.merge(df_roads_ssp2[length_cols].merge(df_roads_ssp4[length_cols], on = 'GEOID', suffixes= [None, '_ssp4']), on = 'GEOID', suffixes= ['_ssp1', '_ssp2'])
    df_buildings = df_buildings_ssp1.merge(df_buildings_ssp2[built_cols].merge(df_buildings_ssp4[built_cols], on = 'GEOID', suffixes= [None, '_ssp4']), on = 'GEOID', suffixes= ['_ssp1', '_ssp2'])
    print(df_roads.shape, df_buildings.shape)

    # merge buildings and roads
    df_ssp = df_buildings.merge(df_roads, on =['GEOID', 'NAMELSAD'])
    print(f"Total number of cities {df_ssp.shape[0]}")

    # Percent Change

    df_ssp['%_C_ssp1_RBUV_2020_2050'] = (df_ssp['volume_m3_perCap_2050_ssp1'] - df_ssp['volume_m3_perCap_2020_ssp1'])/ df_ssp['volume_m3_perCap_2020_ssp1']
    df_ssp['%_C_ssp1_RL_2020_2050'] = (df_ssp['length_m_perCap_2050_ssp1'] - df_ssp['length_m_perCap_2020_ssp1'])/ df_ssp['length_m_perCap_2020_ssp1']

    df_ssp['%_C_ssp2_RBUV_2020_2050'] = (df_ssp['volume_m3_perCap_2050_ssp2'] - df_ssp['volume_m3_perCap_2020_ssp2'])/ df_ssp['volume_m3_perCap_2020_ssp2']
    df_ssp['%_C_ssp2_RL_2020_2050'] = (df_ssp['length_m_perCap_2050_ssp2'] - df_ssp['length_m_perCap_2020_ssp2'])/ df_ssp['length_m_perCap_2020_ssp2']

    df_ssp['%_C_ssp4_RBUV_2020_2050'] = (df_ssp['volume_m3_perCap_2050_ssp4'] - df_ssp['volume_m3_perCap_2020_ssp4'])/ df_ssp['volume_m3_perCap_2020_ssp4']
    df_ssp['%_C_ssp4_RL_2020_2050'] = (df_ssp['length_m_perCap_2050_ssp4'] - df_ssp['length_m_perCap_2020_ssp4'])/ df_ssp['length_m_perCap_2020_ssp4']

    df_ssp['%_C_ssp1_RBUV_2020_2100'] = (df_ssp['volume_m3_perCap_2100_ssp1'] - df_ssp['volume_m3_perCap_2020_ssp1'])/ df_ssp['volume_m3_perCap_2020_ssp1']
    df_ssp['%_C_ssp1_RL_2020_2100'] = (df_ssp['length_m_perCap_2100_ssp1'] - df_ssp['length_m_perCap_2020_ssp1'])/ df_ssp['length_m_perCap_2020_ssp1']

    df_ssp['%_C_ssp2_RBUV_2020_2100'] = (df_ssp['volume_m3_perCap_2100_ssp2'] - df_ssp['volume_m3_perCap_2020_ssp2'])/ df_ssp['volume_m3_perCap_2020_ssp2']
    df_ssp['%_C_ssp2_RL_2020_2100'] = (df_ssp['length_m_perCap_2100_ssp2'] - df_ssp['length_m_perCap_2020_ssp2'])/ df_ssp['length_m_perCap_2020_ssp2']

    df_ssp['%_C_ssp4_RBUV_2020_2100'] = (df_ssp['volume_m3_perCap_2100_ssp4'] - df_ssp['volume_m3_perCap_2020_ssp4'])/ df_ssp['volume_m3_perCap_2020_ssp4']
    df_ssp['%_C_ssp4_RL_2020_2100'] = (df_ssp['length_m_perCap_2100_ssp4'] - df_ssp['length_m_perCap_2020_ssp4'])/ df_ssp['length_m_perCap_2020_ssp4']

    df_ssp['%_C_ssp2_RBUV_2050_2100'] = (df_ssp['volume_m3_perCap_2100_ssp2'] - df_ssp['volume_m3_perCap_2050_ssp2'])/ df_ssp['volume_m3_perCap_2050_ssp2']
    df_ssp['%_C_ssp2_RL_2050_2100'] = (df_ssp['length_m_perCap_2100_ssp2'] - df_ssp['length_m_perCap_2050_ssp2'])/ df_ssp['length_m_perCap_2050_ssp2']

    df_ssp['ssp2_percent_change_RBUV_2020_2040'] = (df_ssp['volume_m3_perCap_2040_ssp2'] - df_ssp['volume_m3_perCap_2020_ssp2'])/df_ssp['volume_m3_perCap_2020_ssp2']
    df_ssp['ssp2_percent_change_RBUV_2040_2060'] = (df_ssp['volume_m3_perCap_2060_ssp2'] - df_ssp['volume_m3_perCap_2040_ssp2'])/df_ssp['volume_m3_perCap_2040_ssp2'] 
    df_ssp['ssp2_percent_change_RBUV_2060_2080'] = (df_ssp['volume_m3_perCap_2080_ssp2'] - df_ssp['volume_m3_perCap_2060_ssp2'])/df_ssp['volume_m3_perCap_2060_ssp2'] 
    df_ssp['ssp2_percent_change_RBUV_2080_2100'] = (df_ssp['volume_m3_perCap_2100_ssp2'] - df_ssp['volume_m3_perCap_2080_ssp2'])/df_ssp['volume_m3_perCap_2080_ssp2'] 

    df_ssp['RBUV_2020_2030'] = (df_ssp['volume_m3_perCap_2030_ssp2'] - df_ssp['volume_m3_perCap_2020_ssp2'])/df_ssp['volume_m3_perCap_2020_ssp2']
    df_ssp['RBUV_2030_2040'] = (df_ssp['volume_m3_perCap_2040_ssp2'] - df_ssp['volume_m3_perCap_2030_ssp2'])/df_ssp['volume_m3_perCap_2030_ssp2']
    df_ssp['RBUV_2040_2050'] = (df_ssp['volume_m3_perCap_2050_ssp2'] - df_ssp['volume_m3_perCap_2040_ssp2'])/df_ssp['volume_m3_perCap_2040_ssp2']
    df_ssp['RBUV_2050_2060'] = (df_ssp['volume_m3_perCap_2060_ssp2'] - df_ssp['volume_m3_perCap_2050_ssp2'])/df_ssp['volume_m3_perCap_2050_ssp2']
    df_ssp['RBUV_2060_2070'] = (df_ssp['volume_m3_perCap_2070_ssp2'] - df_ssp['volume_m3_perCap_2060_ssp2'])/df_ssp['volume_m3_perCap_2060_ssp2']
    df_ssp['RBUV_2070_2080'] = (df_ssp['volume_m3_perCap_2080_ssp2'] - df_ssp['volume_m3_perCap_2070_ssp2'])/df_ssp['volume_m3_perCap_2070_ssp2']
    df_ssp['RBUV_2080_2090'] = (df_ssp['volume_m3_perCap_2090_ssp2'] - df_ssp['volume_m3_perCap_2080_ssp2'])/df_ssp['volume_m3_perCap_2080_ssp2']
    df_ssp['RBUV_2090_2100'] = (df_ssp['volume_m3_perCap_2100_ssp2'] - df_ssp['volume_m3_perCap_2090_ssp2'])/df_ssp['volume_m3_perCap_2090_ssp2']

    df_ssp['added_RBUV_2020_2050'] = (df_ssp['volume_m3_perCap_2050_ssp2'] - df_ssp['volume_m3_perCap_2020_ssp2'])
    df_ssp['added_RBUV_2050_2100'] = (df_ssp['volume_m3_perCap_2100_ssp2'] - df_ssp['volume_m3_perCap_2050_ssp2'])
    df_ssp['added_RL_2020_2050'] = (df_ssp['length_m_perCap_2050_ssp2'] - df_ssp['length_m_perCap_2020_ssp2'])
    df_ssp['added_RL_2050_2100'] = (df_ssp['length_m_perCap_2100_ssp2'] - df_ssp['length_m_perCap_2050_ssp2'])

    df_ssp['added_RBUV_2020_2030'] = (df_ssp['volume_m3_perCap_2030_ssp2'] - df_ssp['volume_m3_perCap_2020_ssp2'])
    df_ssp['added_RBUV_2030_2040'] = (df_ssp['volume_m3_perCap_2040_ssp2'] - df_ssp['volume_m3_perCap_2030_ssp2'])
    df_ssp['added_RBUV_2040_2050'] = (df_ssp['volume_m3_perCap_2050_ssp2'] - df_ssp['volume_m3_perCap_2040_ssp2'])
    df_ssp['added_RBUV_2050_2060'] = (df_ssp['volume_m3_perCap_2060_ssp2'] - df_ssp['volume_m3_perCap_2050_ssp2'])
    df_ssp['added_RBUV_2060_2070'] = (df_ssp['volume_m3_perCap_2070_ssp2'] - df_ssp['volume_m3_perCap_2060_ssp2'])
    df_ssp['added_RBUV_2070_2080'] = (df_ssp['volume_m3_perCap_2080_ssp2'] - df_ssp['volume_m3_perCap_2070_ssp2'])
    df_ssp['added_RBUV_2080_2090'] = (df_ssp['volume_m3_perCap_2090_ssp2'] - df_ssp['volume_m3_perCap_2080_ssp2'])
    df_ssp['added_RBUV_2090_2100'] = (df_ssp['volume_m3_perCap_2100_ssp2'] - df_ssp['volume_m3_perCap_2090_ssp2'])

    df_ssp['added_RL_2020_2030'] = (df_ssp['length_m_perCap_2030_ssp2'] - df_ssp['length_m_perCap_2020_ssp2'])
    df_ssp['added_RL_2030_2040'] = (df_ssp['length_m_perCap_2040_ssp2'] - df_ssp['length_m_perCap_2030_ssp2'])
    df_ssp['added_RL_2040_2050'] = (df_ssp['length_m_perCap_2050_ssp2'] - df_ssp['length_m_perCap_2040_ssp2'])
    df_ssp['added_RL_2050_2060'] = (df_ssp['length_m_perCap_2060_ssp2'] - df_ssp['length_m_perCap_2050_ssp2'])
    df_ssp['added_RL_2060_2070'] = (df_ssp['length_m_perCap_2070_ssp2'] - df_ssp['length_m_perCap_2060_ssp2'])
    df_ssp['added_RL_2070_2080'] = (df_ssp['length_m_perCap_2080_ssp2'] - df_ssp['length_m_perCap_2070_ssp2'])
    df_ssp['added_RL_2080_2090'] = (df_ssp['length_m_perCap_2090_ssp2'] - df_ssp['length_m_perCap_2080_ssp2'])
    df_ssp['added_RL_2090_2100'] = (df_ssp['length_m_perCap_2100_ssp2'] - df_ssp['length_m_perCap_2090_ssp2'])

    df_ssp['ssp2_percent_change_RL_2020_2040'] = (df_ssp['length_m_perCap_2040_ssp2'] - df_ssp['length_m_perCap_2020_ssp2'])/df_ssp['length_m_perCap_2020_ssp2']
    df_ssp['ssp2_percent_change_RL_2040_2060'] = (df_ssp['length_m_perCap_2060_ssp2'] - df_ssp['length_m_perCap_2040_ssp2'])/df_ssp['length_m_perCap_2040_ssp2'] 
    df_ssp['ssp2_percent_change_RL_2060_2080'] = (df_ssp['length_m_perCap_2080_ssp2'] - df_ssp['length_m_perCap_2060_ssp2'])/df_ssp['length_m_perCap_2060_ssp2'] 
    df_ssp['ssp2_percent_change_RL_2080_2100'] = (df_ssp['length_m_perCap_2100_ssp2'] - df_ssp['length_m_perCap_2080_ssp2'])/df_ssp['length_m_perCap_2080_ssp2'] 

    df_ssp['RL_2020_2030'] = (df_ssp['length_m_perCap_2030_ssp2'] - df_ssp['length_m_perCap_2020_ssp2'])/df_ssp['length_m_perCap_2020_ssp2']
    df_ssp['RL_2030_2040'] = (df_ssp['length_m_perCap_2040_ssp2'] - df_ssp['length_m_perCap_2030_ssp2'])/df_ssp['length_m_perCap_2030_ssp2']
    df_ssp['RL_2040_2050'] = (df_ssp['length_m_perCap_2050_ssp2'] - df_ssp['length_m_perCap_2040_ssp2'])/df_ssp['length_m_perCap_2040_ssp2']
    df_ssp['RL_2050_2060'] = (df_ssp['length_m_perCap_2060_ssp2'] - df_ssp['length_m_perCap_2050_ssp2'])/df_ssp['length_m_perCap_2050_ssp2']
    df_ssp['RL_2060_2070'] = (df_ssp['length_m_perCap_2070_ssp2'] - df_ssp['length_m_perCap_2060_ssp2'])/df_ssp['length_m_perCap_2060_ssp2']
    df_ssp['RL_2070_2080'] = (df_ssp['length_m_perCap_2080_ssp2'] - df_ssp['length_m_perCap_2070_ssp2'])/df_ssp['length_m_perCap_2070_ssp2']
    df_ssp['RL_2080_2090'] = (df_ssp['length_m_perCap_2090_ssp2'] - df_ssp['length_m_perCap_2080_ssp2'])/df_ssp['length_m_perCap_2080_ssp2']
    df_ssp['RL_2090_2100'] = (df_ssp['length_m_perCap_2100_ssp2'] - df_ssp['length_m_perCap_2090_ssp2'])/df_ssp['length_m_perCap_2090_ssp2']

    # merge wih population attributes
    df_population = pd.read_csv(r'E:\Scripts\project_QI\data\population\forecasted_trend.csv', index_col =0)
    df_population['GEOID'] = df_population['GEOID'].astype(str).str.rjust(7, '0')

    df_attr = pd.read_csv(r'E:\Scripts\project_QI\data\population\df_attributes.csv', index_col =0)
    df_attr['GEOID'] = df_attr['GEOID'].astype(str).str.rjust(7, '0')

    df_pop = df_population.merge(df_attr[['GEOID', 'tt_2_work_place', 'no_veh', 'veh_1', 'veh_2_or+', 'veh 1+','veh_<=_1']], on ='GEOID')

    df_ssp =  df_ssp.merge(df_pop[['GEOID', 'ssp12020', 'ssp22020', 'ssp32020', 'ssp42020', 'ssp52020','ssp12030', 'ssp22030', 'ssp32030', 'ssp42030', 'ssp52030', 'ssp12040',
        'ssp22040', 'ssp32040', 'ssp42040', 'ssp52040', 'ssp22050', 'ssp32050', 'ssp42050', 'ssp52050', 'ssp22060', 'ssp32060',
        'ssp42060', 'ssp52060', 'ssp12070', 'ssp22070', 'ssp32070', 'ssp42070','ssp52070', 'ssp12080', 'ssp22080', 'ssp32080', 'ssp42080', 'ssp52080',
        'ssp12090', 'ssp22090', 'ssp32090', 'ssp42090', 'ssp52090', 'ssp22100', 'ssp32100', 'ssp42100', 'ssp52100',
        'future trend from SSP 1', 'future trend from SSP 3', 'future trend from SSP 4','future trend from SSP 5',
        'tt_2_work_place', 'no_veh', 'veh_1', 'veh_2_or+', 'veh 1+','veh_<=_1']], on ='GEOID')
    # df_ssp.columns

    df_ssp['STATEFP'] = df_ssp['GEOID'].str[:2]

    df_ssp['RL_perCap_2020'] = pd.cut(df_ssp['length_m_perCap_2020_ssp2'], [-2, 5, 20, 50, 100, 10000], labels=["0-5", "5-20", "20-50", "50-100","100+"])
    df_ssp['RL_perCap_2050'] = pd.cut(df_ssp['length_m_perCap_2050_ssp2'], [-2, 5, 20, 50, 100, 10000], labels=["0-5", "5-20", "20-50", "50-100","100+"])
    df_ssp['RL_perCap_2100'] = pd.cut(df_ssp['length_m_perCap_2100_ssp2'], [-2, 5, 20, 50, 100, 10000], labels=["0-5", "5-20", "20-50", "50-100","100+"])

    df_ssp['RBUV_perCap_2020'] = pd.cut(df_ssp['volume_m3_perCap_2020_ssp2'], [-2, 250, 500, 1000, 5000, 1000000], labels=["0-250", "250-500", "500-1000", "1000-5000","5000+"])
    df_ssp['RBUV_perCap_2050'] = pd.cut(df_ssp['volume_m3_perCap_2050_ssp2'], [-2, 250, 500, 1000, 5000, 1000000], labels=["0-250", "250-500", "500-1000", "1000-5000","5000+"])
    df_ssp['RBUV_perCap_2100'] = pd.cut(df_ssp['volume_m3_perCap_2100_ssp2'], [-2, 250, 500, 1000, 5000, 1000000], labels=["0-250", "250-500", "500-1000", "1000-5000","5000+"])

    df_ssp['RL_Burden_2050'] = df_ssp.apply(label_burden, col1 ='length_m_perCap_2020_ssp2', col2 ='length_m_perCap_2050_ssp2', axis=1)
    df_ssp['RL_Burden_2100'] = df_ssp.apply(label_burden, col1 ='length_m_perCap_2050_ssp2', col2 ='length_m_perCap_2100_ssp2', axis=1)

    df_ssp['RBUV_Burden_2050'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2020_ssp2', col2 ='volume_m3_perCap_2050_ssp2', axis=1)
    df_ssp['RBUV_Burden_2100'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2050_ssp2', col2 ='volume_m3_perCap_2100_ssp2', axis=1)


    df_ssp['RBUV_Burden_2030_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2020_ssp2', col2 ='volume_m3_perCap_2030_ssp2', axis=1)
    df_ssp['RBUV_Burden_2040_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2030_ssp2', col2 ='volume_m3_perCap_2040_ssp2', axis=1)
    df_ssp['RBUV_Burden_2050_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2040_ssp2', col2 ='volume_m3_perCap_2050_ssp2', axis=1)
    df_ssp['RBUV_Burden_2060_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2050_ssp2', col2 ='volume_m3_perCap_2060_ssp2', axis=1)
    df_ssp['RBUV_Burden_2070_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2060_ssp2', col2 ='volume_m3_perCap_2070_ssp2', axis=1)
    df_ssp['RBUV_Burden_2080_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2070_ssp2', col2 ='volume_m3_perCap_2080_ssp2', axis=1)
    df_ssp['RBUV_Burden_2090_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2080_ssp2', col2 ='volume_m3_perCap_2090_ssp2', axis=1)
    df_ssp['RBUV_Burden_2100_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2090_ssp2', col2 ='volume_m3_perCap_2100_ssp2', axis=1)


    return df_ssp





