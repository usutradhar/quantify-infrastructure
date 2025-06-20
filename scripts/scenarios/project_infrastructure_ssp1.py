# import all packages
import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import math
import statsmodels.formula.api as sm
from statsmodels.stats.anova import anova_lm

import sys
import time
from scripts.functions.functions_scaling_vect import find_scale_parameters, find_next_stock, process_stock_at_t


def project_infrastructure_ssp1(input_df,
                      check_na_columns, # = ['CensusPop_20', 'ssp12040', 'volume_Res_2020'],
                      current_stock_column, # = 'volume_Res_2020',
                      current_pop_column, # = 'CensusPop_20',
                      project_pop_columns, # = ['ssp12030', 'ssp12040', 'ssp12050', 'ssp12060', 'ssp12070', 'ssp12080', 'ssp12090', 'ssp12100'],
                      case, # = 'mean', 
                      infrastructure, # = 'RBUV',
                      output_path,
                      random_state=False): # r'outputfiles\outputs_beta_mean\\')
        
    
        if infrastructure == 'RBUV':
            df_clean = input_df.dropna(subset=check_na_columns).reset_index(drop=True)
            df_clean['avg_HU_size_Res_sqm'] = df_clean[current_stock_column] / df_clean[' !!Total:']

            # converting building footprint to gross area
            df_clean['surface_Res_gross_2020'] = df_clean['surface_Res_2020'] * df_clean['floors'] 
            print("Shape of the clean dataset with nonzero values:==", df_clean.shape)

            general_columns = ['GEOID', 'State', 'NAMELSAD', 'ALAND','REGION', 'STATEFP','ua-to-place allocation factor_max', 'population_ua_max', 'weighted_HU_density_sqmi', 
                                 'city type', 'median_income', 'avg_HU_size_Res_sqm','surface_Res_2020','surface_Res_gross_2020']
    
        elif infrastructure == 'RL':
            ### Read redefined urban rural classes
            df_urban_rural_conn = pd.read_csv(output_path + 'output_city_type_ssp1.csv')
            df_urban_rural_conn = df_urban_rural_conn[['GEOID','citytype_at_2030', 'citytype_at_2040', 'citytype_at_2050', 'citytype_at_2060',
                                                    'citytype_at_2070',  'citytype_at_2080', 'citytype_at_2090', 'citytype_at_2100']]

            df_urban_rural_conn['GEOID'] = df_urban_rural_conn['GEOID'].astype(str).str.rjust(7,'0')
            # This step excludes 36 CDPs that were newly added and therefore do not have sufficient data for urban rural classification
            df_clean = input_df.merge(df_urban_rural_conn, on = 'GEOID',)

            general_columns = ['GEOID', 'State', 'NAMELSAD','city type','median_income', 'label','future trend from SSP 2','weighted_HU_density_sqmi',
                               'road_density_m-sqm', 'total_length', 'citytype_at_2030', 'citytype_at_2040','citytype_at_2050', 'citytype_at_2060',
                               'citytype_at_2070',  'citytype_at_2080', 'citytype_at_2090', 'citytype_at_2100']
    

        df_clean['stock_at_t0'] = df_clean[current_stock_column]
        df_clean['per_cap_mass_at_2020'] =  df_clean['stock_at_t0'] / df_clean['CensusPop_20']

        
        columns_for_analysis = general_columns + [current_pop_column] + project_pop_columns + [current_stock_column] + ['stock_at_t0'] + ['per_cap_mass_at_2020'] 

        df_for_analysis = df_clean[columns_for_analysis]
        # print(df_for_analysis.columns)

        # projecting the first time step  
        current_stock_col = current_stock_column
        current_pop_col = current_pop_column
        next_pop_col = project_pop_columns[0]
        t = 2030

        df_for_2030 = process_stock_at_t(df_for_analysis, current_stock_col, current_pop_col, next_pop_col, t, 
                                         case = case, 
                                         infrastructure = infrastructure,
                                         random_state = random_state)
        df_for_analysis = df_for_analysis.merge(df_for_2030, on ='GEOID')

        current_stock_col = 'surface_Res_at_2030'
        time_steps = [2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
        steps = min(len(time_steps)-1, len(project_pop_columns))
      
        for i in range(steps):        
            current_pop = project_pop_columns[i]
            next_pop = project_pop_columns[i+1]
            t = time_steps[i+1]
            print("============================================")
            print(f"--------Running for year {t} with current stock {current_stock_col} , current population {current_pop}, for future population {next_pop}----------")
            print("============================================")
            start_time = time.time()
            df_for_t = process_stock_at_t(df_for_analysis, current_stock_col, current_pop, next_pop, t, 
                                          case = case, 
                                          infrastructure = infrastructure, 
                                          random_state = random_state)
            end_time = time.time()
            run_time = end_time - start_time
            print(f"Runtime:==================================== {run_time:.2f} seconds")


            current_stock_col = current_stock_col[:-4] + str(time_steps[i+1])
            df_for_analysis = df_for_analysis.merge(df_for_t, on ='GEOID')
            print(df_for_analysis.shape)

        df_for_analysis['percent change from 2020-2060'] = (df_for_analysis['per_cap_mass_at_2060'] - df_for_analysis['per_cap_mass_at_2020']) / df_for_analysis['per_cap_mass_at_2020']
        df_for_analysis['percent change from 2060-2100'] = (df_for_analysis['per_cap_mass_at_2100'] - df_for_analysis['per_cap_mass_at_2060']) / df_for_analysis['per_cap_mass_at_2060']
        df_for_analysis['percent change from 2020-2100'] = (df_for_analysis['per_cap_mass_at_2100'] - df_for_analysis['per_cap_mass_at_2020']) / df_for_analysis['per_cap_mass_at_2020']

        df_for_analysis['per cap added from 2020-2030'] = (df_for_analysis['per_cap_mass_at_2030'] - df_for_analysis['per_cap_mass_at_2020'])
        df_for_analysis['per cap added from 2030-2040'] = (df_for_analysis['per_cap_mass_at_2040'] - df_for_analysis['per_cap_mass_at_2030'])
        df_for_analysis['per cap added from 2040-2050'] = (df_for_analysis['per_cap_mass_at_2050'] - df_for_analysis['per_cap_mass_at_2040'])
        df_for_analysis['per cap added from 2050-2060'] = (df_for_analysis['per_cap_mass_at_2060'] - df_for_analysis['per_cap_mass_at_2050'])
        df_for_analysis['per cap added from 2060-2070'] = (df_for_analysis['per_cap_mass_at_2070'] - df_for_analysis['per_cap_mass_at_2060'])
        df_for_analysis['per cap added from 2070-2080'] = (df_for_analysis['per_cap_mass_at_2080'] - df_for_analysis['per_cap_mass_at_2070'])
        df_for_analysis['per cap added from 2080-2090'] = (df_for_analysis['per_cap_mass_at_2090'] - df_for_analysis['per_cap_mass_at_2080'])
        df_for_analysis['per cap added from 2090-2100'] = (df_for_analysis['per_cap_mass_at_2100'] - df_for_analysis['per_cap_mass_at_2090'])

        df_for_analysis_sub = df_for_analysis.assign(**df_for_analysis[['per_cap_mass_at_2020', 'per_cap_mass_at_2030', 'per_cap_mass_at_2040', 'per_cap_mass_at_2050', 
        'per_cap_mass_at_2060', 'per_cap_mass_at_2070', 'per_cap_mass_at_2080', 'per_cap_mass_at_2090', 'per_cap_mass_at_2100']].sub(df_for_analysis['per_cap_mass_at_2020'], axis=0).add_prefix('sub_'))


        if infrastructure == 'RBUV':
            projected_perCap = df_for_analysis_sub.rename(columns={'per_cap_mass_at_2020': 'volume_m3_perCap_2020','per_cap_mass_at_2030': 'volume_m3_perCap_2030', 'per_cap_mass_at_2040': 'volume_m3_perCap_2040', 
                                'per_cap_mass_at_2050': 'volume_m3_perCap_2050', 'per_cap_mass_at_2060': 'volume_m3_perCap_2060', 'per_cap_mass_at_2070': 'volume_m3_perCap_2070', 
                                'per_cap_mass_at_2080': 'volume_m3_perCap_2080', 'per_cap_mass_at_2090': 'volume_m3_perCap_2090', 'per_cap_mass_at_2100': 'volume_m3_perCap_2100'})
            # save output files
            projected_perCap[['GEOID', 'NAMELSAD', 'city type','REGION', 'ALAND', 'CensusPop_20','ssp12030', 'ssp12040','ssp12050', 'ssp12060','ssp12070', 'ssp12080', 'ssp12090', 'ssp12100', 
                            'volume_m3_perCap_2020', 'volume_m3_perCap_2030', 'volume_m3_perCap_2040', 'volume_m3_perCap_2050', 'volume_m3_perCap_2060', 'volume_m3_perCap_2070', 
                            'volume_m3_perCap_2080', 'volume_m3_perCap_2090', 'volume_m3_perCap_2100']].to_csv(output_path + 'buildings_perCap_ssp1.csv')

            projected_perCap[['STATEFP','GEOID', 'NAMELSAD', 'city type', 'citytype_at_2030', 'citytype_at_2040', 'citytype_at_2050','citytype_at_2060', 'citytype_at_2070', 'citytype_at_2080',
                                'citytype_at_2090', 'citytype_at_2100']].to_csv(output_path + 'output_city_type_ssp1.csv')
        
        elif infrastructure == 'RL':
            projected_perCap = df_for_analysis_sub.rename(columns={'per_cap_mass_at_2020': 'length_m_perCap_2020','per_cap_mass_at_2030': 'length_m_perCap_2030', 'per_cap_mass_at_2040': 'length_m_perCap_2040', 
                                'per_cap_mass_at_2050': 'length_m_perCap_2050', 'per_cap_mass_at_2060': 'length_m_perCap_2060', 'per_cap_mass_at_2070': 'length_m_perCap_2070', 
                                'per_cap_mass_at_2080': 'length_m_perCap_2080', 'per_cap_mass_at_2090': 'length_m_perCap_2090', 'per_cap_mass_at_2100': 'length_m_perCap_2100'})
            # save output files
            projected_perCap[['GEOID', 'NAMELSAD','length_m_perCap_2020','length_m_perCap_2030', 'length_m_perCap_2040', 'length_m_perCap_2050', 'length_m_perCap_2060', 
                        'length_m_perCap_2070', 'length_m_perCap_2080', 'length_m_perCap_2090', 'length_m_perCap_2100','label','future trend from SSP 2','weighted_HU_density_sqmi', 'median_income',
                        'citytype_at_2030', 'citytype_at_2040', 'citytype_at_2050', 'citytype_at_2060', 'citytype_at_2070', 'citytype_at_2080', 'citytype_at_2090','citytype_at_2100'
                        ]].to_csv(output_path + 'roads_perCap_ssp1_local.csv')

        return df_for_analysis