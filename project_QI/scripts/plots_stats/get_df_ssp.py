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
    if np.round(row[col1], 0) < np.round(row[col2],0):
        return 'increasingBurden'
    elif np.round(row[col1],0) > np.round(row[col2],0):
        return 'decreasingBurden'
    elif np.round(row[col1],0) == np.round(row[col2],0):
        return 'noChange'
    else:
        return 'error'
    


def get_df_ssp(filepath =r'outputfiles\\'):
    df_roads_min = pd.read_csv(r'outputfiles\outputs_beta_min\roads_perCap_ssp2_local_org.csv', index_col = 0)
    df_roads_min['GEOID'] = df_roads_min['GEOID'].astype('str').str.rjust(7, '0')

    df_buildings_min = pd.read_csv(r'outputfiles\outputs_beta_min\buildings_perCap_ssp2_org.csv', index_col = 0)
    df_buildings_min['GEOID'] = df_buildings_min['GEOID'].astype('str').str.rjust(7, '0')

    df_roads_mean = pd.read_csv(r'outputfiles\outputs_beta_mean\roads_perCap_ssp2_local_org.csv', index_col = 0)
    df_roads_mean['GEOID'] = df_roads_mean['GEOID'].astype('str').str.rjust(7, '0')

    df_buildings_mean = pd.read_csv(r'outputfiles\outputs_beta_mean\buildings_perCap_ssp2_org.csv', index_col = 0)
    df_buildings_mean['GEOID'] = df_buildings_mean['GEOID'].astype('str').str.rjust(7, '0')

    df_roads_max = pd.read_csv(r'outputfiles\outputs_beta_max\roads_perCap_ssp2_local_org.csv', index_col = 0)
    df_roads_max['GEOID'] = df_roads_max['GEOID'].astype('str').str.rjust(7, '0')

    df_buildings_max = pd.read_csv(r'outputfiles\outputs_beta_max\\buildings_perCap_ssp2_org.csv', index_col = 0)
    df_buildings_max['GEOID'] = df_buildings_max['GEOID'].astype('str').str.rjust(7, '0')

    print(df_buildings_min.shape, df_buildings_mean.shape, df_buildings_max.shape, df_roads_min.shape, df_roads_mean.shape, df_roads_max.shape)

    # rename columns
    length_cols = ['GEOID', 'length_m_perCap_2020', 'length_m_perCap_2030','length_m_perCap_2040', 'length_m_perCap_2050', 'length_m_perCap_2060',
               'length_m_perCap_2070', 'length_m_perCap_2080', 'length_m_perCap_2090', 'length_m_perCap_2100']

    built_cols =['GEOID', 'volume_m3_perCap_2020', 'volume_m3_perCap_2030', 'volume_m3_perCap_2040','volume_m3_perCap_2050', 'volume_m3_perCap_2060',
                'volume_m3_perCap_2070', 'volume_m3_perCap_2080', 'volume_m3_perCap_2090', 'volume_m3_perCap_2100']


    df_buildings = df_buildings_min.merge(df_buildings_mean[built_cols].merge(df_buildings_max[built_cols], on = 'GEOID', suffixes= [None, '_max']), on = 'GEOID', suffixes= ['_min', '_mean'])
    df_roads = df_roads_min.merge(df_roads_mean[length_cols].merge(df_roads_max[length_cols], on = 'GEOID', suffixes= [None, '_max']), on = 'GEOID', suffixes= ['_min', '_mean'])

    print(df_roads.shape, df_buildings.shape)

    # merge buildings and roads
    df_ssp = df_buildings.merge(df_roads, on =['GEOID', 'NAMELSAD'])
    print(f"Total number of cities {df_ssp.shape[0]}")

    # Percent Change
    df_ssp['%_C_mean_RBUV_2020_2050'] = (df_ssp['volume_m3_perCap_2050_mean'] - df_ssp['volume_m3_perCap_2020_mean'])/ df_ssp['volume_m3_perCap_2020_mean']
    df_ssp['%_C_mean_RL_2020_2050'] = (df_ssp['length_m_perCap_2050_mean'] - df_ssp['length_m_perCap_2020_mean'])/ df_ssp['length_m_perCap_2020_mean']

    df_ssp['%_C_mean_RBUV_2020_2100'] = (df_ssp['volume_m3_perCap_2100_mean'] - df_ssp['volume_m3_perCap_2020_mean'])/ df_ssp['volume_m3_perCap_2020_mean']
    df_ssp['%_C_mean_RL_2020_2100'] = (df_ssp['length_m_perCap_2100_mean'] - df_ssp['length_m_perCap_2020_mean'])/ df_ssp['length_m_perCap_2020_mean']

    df_ssp['%_C_mean_RBUV_2050_2100'] = (df_ssp['volume_m3_perCap_2100_mean'] - df_ssp['volume_m3_perCap_2050_mean'])/ df_ssp['volume_m3_perCap_2050_mean']
    df_ssp['%_C_mean_RL_2050_2100'] = (df_ssp['length_m_perCap_2100_mean'] - df_ssp['length_m_perCap_2050_mean'])/ df_ssp['length_m_perCap_2050_mean']

    df_ssp['mean_percent_change_RBUV_2020_2040'] = (df_ssp['volume_m3_perCap_2040_mean'] - df_ssp['volume_m3_perCap_2020_mean'])/df_ssp['volume_m3_perCap_2020_mean']
    df_ssp['mean_percent_change_RBUV_2040_2060'] = (df_ssp['volume_m3_perCap_2060_mean'] - df_ssp['volume_m3_perCap_2040_mean'])/df_ssp['volume_m3_perCap_2040_mean'] 
    df_ssp['mean_percent_change_RBUV_2060_2080'] = (df_ssp['volume_m3_perCap_2080_mean'] - df_ssp['volume_m3_perCap_2060_mean'])/df_ssp['volume_m3_perCap_2060_mean'] 
    df_ssp['mean_percent_change_RBUV_2080_2100'] = (df_ssp['volume_m3_perCap_2100_mean'] - df_ssp['volume_m3_perCap_2080_mean'])/df_ssp['volume_m3_perCap_2080_mean'] 

    df_ssp['RBUV_2020_2030'] = (df_ssp['volume_m3_perCap_2030_mean'] - df_ssp['volume_m3_perCap_2020_mean'])/df_ssp['volume_m3_perCap_2020_mean']
    df_ssp['RBUV_2030_2040'] = (df_ssp['volume_m3_perCap_2040_mean'] - df_ssp['volume_m3_perCap_2030_mean'])/df_ssp['volume_m3_perCap_2030_mean']
    df_ssp['RBUV_2040_2050'] = (df_ssp['volume_m3_perCap_2050_mean'] - df_ssp['volume_m3_perCap_2040_mean'])/df_ssp['volume_m3_perCap_2040_mean']
    df_ssp['RBUV_2050_2060'] = (df_ssp['volume_m3_perCap_2060_mean'] - df_ssp['volume_m3_perCap_2050_mean'])/df_ssp['volume_m3_perCap_2050_mean']
    df_ssp['RBUV_2060_2070'] = (df_ssp['volume_m3_perCap_2070_mean'] - df_ssp['volume_m3_perCap_2060_mean'])/df_ssp['volume_m3_perCap_2060_mean']
    df_ssp['RBUV_2070_2080'] = (df_ssp['volume_m3_perCap_2080_mean'] - df_ssp['volume_m3_perCap_2070_mean'])/df_ssp['volume_m3_perCap_2070_mean']
    df_ssp['RBUV_2080_2090'] = (df_ssp['volume_m3_perCap_2090_mean'] - df_ssp['volume_m3_perCap_2080_mean'])/df_ssp['volume_m3_perCap_2080_mean']
    df_ssp['RBUV_2090_2100'] = (df_ssp['volume_m3_perCap_2100_mean'] - df_ssp['volume_m3_perCap_2090_mean'])/df_ssp['volume_m3_perCap_2090_mean']

    df_ssp['added_RBUV_2020_2050'] = (df_ssp['volume_m3_perCap_2050_mean'] - df_ssp['volume_m3_perCap_2020_mean'])
    df_ssp['added_RBUV_2050_2100'] = (df_ssp['volume_m3_perCap_2100_mean'] - df_ssp['volume_m3_perCap_2050_mean'])
    df_ssp['added_RL_2020_2050'] = (df_ssp['length_m_perCap_2050_mean'] - df_ssp['length_m_perCap_2020_mean'])
    df_ssp['added_RL_2050_2100'] = (df_ssp['length_m_perCap_2100_mean'] - df_ssp['length_m_perCap_2050_mean'])

    df_ssp['added_RBUV_2020_2030'] = (df_ssp['volume_m3_perCap_2030_mean'] - df_ssp['volume_m3_perCap_2020_mean'])
    df_ssp['added_RBUV_2030_2040'] = (df_ssp['volume_m3_perCap_2040_mean'] - df_ssp['volume_m3_perCap_2030_mean'])
    df_ssp['added_RBUV_2040_2050'] = (df_ssp['volume_m3_perCap_2050_mean'] - df_ssp['volume_m3_perCap_2040_mean'])
    df_ssp['added_RBUV_2050_2060'] = (df_ssp['volume_m3_perCap_2060_mean'] - df_ssp['volume_m3_perCap_2050_mean'])
    df_ssp['added_RBUV_2060_2070'] = (df_ssp['volume_m3_perCap_2070_mean'] - df_ssp['volume_m3_perCap_2060_mean'])
    df_ssp['added_RBUV_2070_2080'] = (df_ssp['volume_m3_perCap_2080_mean'] - df_ssp['volume_m3_perCap_2070_mean'])
    df_ssp['added_RBUV_2080_2090'] = (df_ssp['volume_m3_perCap_2090_mean'] - df_ssp['volume_m3_perCap_2080_mean'])
    df_ssp['added_RBUV_2090_2100'] = (df_ssp['volume_m3_perCap_2100_mean'] - df_ssp['volume_m3_perCap_2090_mean'])

    df_ssp['added_RL_2020_2030'] = (df_ssp['length_m_perCap_2030_mean'] - df_ssp['length_m_perCap_2020_mean'])
    df_ssp['added_RL_2030_2040'] = (df_ssp['length_m_perCap_2040_mean'] - df_ssp['length_m_perCap_2030_mean'])
    df_ssp['added_RL_2040_2050'] = (df_ssp['length_m_perCap_2050_mean'] - df_ssp['length_m_perCap_2040_mean'])
    df_ssp['added_RL_2050_2060'] = (df_ssp['length_m_perCap_2060_mean'] - df_ssp['length_m_perCap_2050_mean'])
    df_ssp['added_RL_2060_2070'] = (df_ssp['length_m_perCap_2070_mean'] - df_ssp['length_m_perCap_2060_mean'])
    df_ssp['added_RL_2070_2080'] = (df_ssp['length_m_perCap_2080_mean'] - df_ssp['length_m_perCap_2070_mean'])
    df_ssp['added_RL_2080_2090'] = (df_ssp['length_m_perCap_2090_mean'] - df_ssp['length_m_perCap_2080_mean'])
    df_ssp['added_RL_2090_2100'] = (df_ssp['length_m_perCap_2100_mean'] - df_ssp['length_m_perCap_2090_mean'])

    df_ssp['mean_percent_change_RL_2020_2040'] = (df_ssp['length_m_perCap_2040_mean'] - df_ssp['length_m_perCap_2020_mean'])/df_ssp['length_m_perCap_2020_mean']
    df_ssp['mean_percent_change_RL_2040_2060'] = (df_ssp['length_m_perCap_2060_mean'] - df_ssp['length_m_perCap_2040_mean'])/df_ssp['length_m_perCap_2040_mean'] 
    df_ssp['mean_percent_change_RL_2060_2080'] = (df_ssp['length_m_perCap_2080_mean'] - df_ssp['length_m_perCap_2060_mean'])/df_ssp['length_m_perCap_2060_mean'] 
    df_ssp['mean_percent_change_RL_2080_2100'] = (df_ssp['length_m_perCap_2100_mean'] - df_ssp['length_m_perCap_2080_mean'])/df_ssp['length_m_perCap_2080_mean'] 

    df_ssp['RL_2020_2030'] = (df_ssp['length_m_perCap_2030_mean'] - df_ssp['length_m_perCap_2020_mean'])/df_ssp['length_m_perCap_2020_mean']
    df_ssp['RL_2030_2040'] = (df_ssp['length_m_perCap_2040_mean'] - df_ssp['length_m_perCap_2030_mean'])/df_ssp['length_m_perCap_2030_mean']
    df_ssp['RL_2040_2050'] = (df_ssp['length_m_perCap_2050_mean'] - df_ssp['length_m_perCap_2040_mean'])/df_ssp['length_m_perCap_2040_mean']
    df_ssp['RL_2050_2060'] = (df_ssp['length_m_perCap_2060_mean'] - df_ssp['length_m_perCap_2050_mean'])/df_ssp['length_m_perCap_2050_mean']
    df_ssp['RL_2060_2070'] = (df_ssp['length_m_perCap_2070_mean'] - df_ssp['length_m_perCap_2060_mean'])/df_ssp['length_m_perCap_2060_mean']
    df_ssp['RL_2070_2080'] = (df_ssp['length_m_perCap_2080_mean'] - df_ssp['length_m_perCap_2070_mean'])/df_ssp['length_m_perCap_2070_mean']
    df_ssp['RL_2080_2090'] = (df_ssp['length_m_perCap_2090_mean'] - df_ssp['length_m_perCap_2080_mean'])/df_ssp['length_m_perCap_2080_mean']
    df_ssp['RL_2090_2100'] = (df_ssp['length_m_perCap_2100_mean'] - df_ssp['length_m_perCap_2090_mean'])/df_ssp['length_m_perCap_2090_mean']

    # merge wih population attributes
    df_population = pd.read_csv(r'data\population\forecasted_trend.csv', index_col =0)
    df_population['GEOID'] = df_population['GEOID'].astype(str).str.rjust(7, '0')

    df_attr = pd.read_csv(r'data\population\df_attributes.csv', index_col =0)
    df_attr['GEOID'] = df_attr['GEOID'].astype(str).str.rjust(7, '0')

    df_pop = df_population.merge(df_attr[['GEOID', 'tt_2_work_place', 'no_veh', 'veh_1', 'veh_2_or+', 'veh 1+','veh_<=_1']], on ='GEOID')

    df_ssp = df_ssp.merge(df_pop[['GEOID', 'tt_2_work_place', 'no_veh', 'veh_1', 'veh_2_or+', 'veh 1+','veh_<=_1']], on ='GEOID')
    # df_ssp.columns

    df_ssp['STATEFP'] = df_ssp['GEOID'].str[:2]

    df_ssp['RL_perCap_2020'] = pd.cut(df_ssp['length_m_perCap_2020_mean'], [-2, 5, 20, 50, 100, 10000], labels=["0-5", "5-20", "20-50", "50-100","100+"])
    df_ssp['RL_perCap_2050'] = pd.cut(df_ssp['length_m_perCap_2050_mean'], [-2, 5, 20, 50, 100, 10000], labels=["0-5", "5-20", "20-50", "50-100","100+"])
    df_ssp['RL_perCap_2100'] = pd.cut(df_ssp['length_m_perCap_2100_mean'], [-2, 5, 20, 50, 100, 10000], labels=["0-5", "5-20", "20-50", "50-100","100+"])

    df_ssp['RBUV_perCap_2020'] = pd.cut(df_ssp['volume_m3_perCap_2020_mean'], [-2, 250, 500, 1000, 5000, 1000000], labels=["0-250", "250-500", "500-1000", "1000-5000","5000+"])
    df_ssp['RBUV_perCap_2050'] = pd.cut(df_ssp['volume_m3_perCap_2050_mean'], [-2, 250, 500, 1000, 5000, 1000000], labels=["0-250", "250-500", "500-1000", "1000-5000","5000+"])
    df_ssp['RBUV_perCap_2100'] = pd.cut(df_ssp['volume_m3_perCap_2100_mean'], [-2, 250, 500, 1000, 5000, 1000000], labels=["0-250", "250-500", "500-1000", "1000-5000","5000+"])

    df_ssp['RL_Burden_2050'] = df_ssp.apply(label_burden, col1 ='length_m_perCap_2020_mean', col2 ='length_m_perCap_2050_mean', axis=1)
    df_ssp['RL_Burden_2100'] = df_ssp.apply(label_burden, col1 ='length_m_perCap_2050_mean', col2 ='length_m_perCap_2100_mean', axis=1)

    df_ssp['RBUV_Burden_2050'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2020_mean', col2 ='volume_m3_perCap_2050_mean', axis=1)
    df_ssp['RBUV_Burden_2100'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2050_mean', col2 ='volume_m3_perCap_2100_mean', axis=1)


    df_ssp['RL_Burden_2050_min'] = df_ssp.apply(label_burden, col1 ='length_m_perCap_2020_min', col2 ='length_m_perCap_2050_min', axis=1)
    df_ssp['RL_Burden_2100_min'] = df_ssp.apply(label_burden, col1 ='length_m_perCap_2050_min', col2 ='length_m_perCap_2100_min', axis=1)

    df_ssp['RBUV_Burden_2050_min'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2020_min', col2 ='volume_m3_perCap_2050_min', axis=1)
    df_ssp['RBUV_Burden_2100_min'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2050_min', col2 ='volume_m3_perCap_2100_min', axis=1)

    df_ssp['RL_Burden_2050_max'] = df_ssp.apply(label_burden, col1 ='length_m_perCap_2020_max', col2 ='length_m_perCap_2050_max', axis=1)
    df_ssp['RL_Burden_2100_max'] = df_ssp.apply(label_burden, col1 ='length_m_perCap_2050_max', col2 ='length_m_perCap_2100_max', axis=1)

    df_ssp['RBUV_Burden_2050_max'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2020_max', col2 ='volume_m3_perCap_2050_max', axis=1)
    df_ssp['RBUV_Burden_2100_max'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2050_max', col2 ='volume_m3_perCap_2100_max', axis=1)


    df_ssp['RBUV_Burden_2030_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2020_mean', col2 ='volume_m3_perCap_2030_mean', axis=1)
    df_ssp['RBUV_Burden_2040_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2030_mean', col2 ='volume_m3_perCap_2040_mean', axis=1)
    df_ssp['RBUV_Burden_2050_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2040_mean', col2 ='volume_m3_perCap_2050_mean', axis=1)
    df_ssp['RBUV_Burden_2060_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2050_mean', col2 ='volume_m3_perCap_2060_mean', axis=1)
    df_ssp['RBUV_Burden_2070_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2060_mean', col2 ='volume_m3_perCap_2070_mean', axis=1)
    df_ssp['RBUV_Burden_2080_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2070_mean', col2 ='volume_m3_perCap_2080_mean', axis=1)
    df_ssp['RBUV_Burden_2090_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2080_mean', col2 ='volume_m3_perCap_2090_mean', axis=1)
    df_ssp['RBUV_Burden_2100_in'] = df_ssp.apply(label_burden, col1 ='volume_m3_perCap_2090_mean', col2 ='volume_m3_perCap_2100_mean', axis=1)


    return df_ssp





