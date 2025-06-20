# Fit scaling law to infrastructure data

# import all packages
import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import math
import statsmodels.formula.api as sm
from statsmodels.stats.anova import anova_lm
import geopandas as gpd
import time 
pd.set_option('mode.chained_assignment', None) # To stop SettingWithCopy warning
##################################################################################################

def find_scale_parameters(df, infra_col, pop_col):
    '''
    Fit data to liner regression
    Columns are transformed to log scale before plotting
    - infra_col: total infrastructure length/ area/volume for a city
    - pop_col: population data for a city
    - returns model parameters as intercepts, beta, model_params
    '''
    df.loc[:,'log_pop_col'] = np.log(df[pop_col].astype(float))
    df.loc[:, 'log_infra_col'] = np.log(df[infra_col].astype(float))

    # fig = px.scatter(df, x='log_pop_col', y='log_infra_col', hover_data=['GEOID', 'NAMELSAD'],
    #                 width=800, height=800) 
    # fig.show()
    # Check plot how the log-log plot looks?
    sns.regplot(x='log_pop_col', y='log_infra_col', data=df, 
                y_jitter=.03, ci =None, scatter_kws={"s": 1})
    x = df['log_pop_col']
    y = df['log_infra_col']

    #run anova model to find confidence intervals
    # Fit the regression model
    model = sm.ols("y ~ x", data={"y": y, "x": x}).fit()

    # Display regression summary
    print(model.summary())
    print(model.params)
    print("R-squared value" , model.rsquared)
    print('====================================')
    # Calculate confidence intervals using ANOVA
    anova_result = anova_lm(model)
    conf_int = model.conf_int(alpha=0.05, cols=None)

    # Display ANOVA results and confidence intervals
    print("\nConfidence Intervals: 'log(a)' and 'b' values---")
    print(conf_int)
    intercepts = [conf_int.iloc[0][0], conf_int.iloc[0][1]]
    beta = [conf_int.iloc[1][0], conf_int.iloc[1][1]]
    model_params = model.params
    return intercepts, beta, model_params

##################################################################################################
list_of_city_types = ['urban', 'suburban', 'periurban', 'rural']


def find_next_stock(df, current_stock_col, current_pop_col, next_pop_col, citytype, t, random_state):
    '''
    takes as inputs:
    - current total infrastructure stock per city, 
    - current population and 
    - prjected population for the next time step
    -- returns a dataframe containing infrastructure_stock_next_time_step and per_capita_stock_at_next_time_step
    
    '''
    print(f"Initial dataframe:====")
    print(df.shape)
    
    df_stocks_iter = pd.DataFrame()
    citytype_col = 'city type'


    if t == 2030:
        citytype_col = 'city type'
    else:
        citytype_col = 'citytype_at_' + str(t-10)
    
    # df['per_cap_mass_t1'] = df[current_stock_col] / df[current_pop_col].round(0)
    df_nonzero = df[df[current_pop_col] !=0]
    df_nonzero_city = df_nonzero[df_nonzero[citytype_col] == citytype]
    print(f"Shape of the dataframe for {citytype}:====")
    print(df_nonzero_city.shape)
    

    print("Prune upper and lower 10 percent data before fitting to the scaling law")
    print("***DEFINITELY NEED TO VALIDATE THIS PRUNING***")
    ### ============== PRUNING ============== ### top and bottom 10 %
    # Set the threshold to the 90th percentile
    threshold_upper = df_nonzero_city[current_stock_col].quantile(0.9)
    threshold_lower = df_nonzero_city[current_stock_col].quantile(0.1)
    # Filter the DataFrame to include only points in the to middle 80%
    df_for_fitting = df_nonzero_city[df_nonzero_city[current_stock_col] < threshold_upper]
    df_for_fitting = df_nonzero_city[df_nonzero_city[current_stock_col] > threshold_lower]
    print("Shape of dataframe used in scaling fit:===")
    print(df_for_fitting.shape)

    x = find_scale_parameters(df_for_fitting, infra_col= 'stock_at_t0', pop_col = 'CensusPop_20')

    # Get the model parameters
    # Set the range for the uniform distribution
    beta_lower = x[1][0]
    beta_upper= x[1][1]
 
    print("Using the coefficient found from scaling law for the complete dataframe:===")
    # Create an empty column of stocks at next time step
    df_nonzero_city.loc[:,'stock_at_t2'] = np.nan
    df_nonzero_city.loc[:,'stocks_at_reg'] = np.nan

    df_stocks_iter = df_nonzero_city[['GEOID', 'NAMELSAD']]

#   # Randomly sample beta values    
    # TO MAINTAIN REPRODUCEABILITY 
    from numpy.random import default_rng
    if random_state == True:
        rng = default_rng(seed=42)
        for i in range(100):
            # Sampling coefficient beta from and uniform distribution 
            beta_uniform = rng.uniform(low=beta_lower, high=beta_upper, size=10000) 
            beta = rng.choice(beta_uniform)  
            # print(f"beta value================================== {beta}")
            # --- Vectorized with np.where ---
            log_a_series = np.log(df_nonzero_city[current_stock_col] / df_nonzero_city[current_pop_col]) 
            df_nonzero_city['stocks_at_reg'] = np.exp(log_a_series) * (df_nonzero_city[current_pop_col] ** beta)
            df_nonzero_city['stocks_at_t2'] = np.where(
                df_nonzero_city[next_pop_col] >= df_nonzero_city[current_pop_col],
                np.exp(log_a_series) * (df_nonzero_city[next_pop_col] ** beta),
                0
            )

            df_nonzero_city.loc[:,'added_stock'] = np.where(df_nonzero_city['stocks_at_t2'] == 0, 0, (df_nonzero_city['stocks_at_t2'] - df_nonzero_city['stocks_at_reg']))
            df_nonzero_city.loc[:,'stock_next_time_step'] = (df_nonzero_city[current_stock_col] + df_nonzero_city['added_stock'])

            # population cannot be a fraction. When Population drops below 1, it generates huge per capita infrastructure
            # However, using 0 as population will result in infinite values for per capita infrastructure
            # So, we replace population values less 1 by 1 to get the infrastructure value
            # Function to apply conditions and assign values to a new column
            def replace_frac_population_by_one(row):
                if (row[next_pop_col] < 1):
                    return (row[current_stock_col] + row['added_stock']) / 1 
                else:
                    return (row[current_stock_col] + row['added_stock']) / np.round(row[next_pop_col],0)

            # Apply the function to create a new column based on conditions
            df_nonzero_city['per_cap_mass_next_time_step'] = df_nonzero_city.apply(replace_frac_population_by_one, axis=1)

            df_stocks_iter = pd.concat([df_stocks_iter, df_nonzero_city[['per_cap_mass_next_time_step', 'stock_next_time_step']]], axis=1)

    else:
        for i in range(100):
            # Sampling coefficient beta from and uniform distribution
            beta_uniform = np.random.uniform(low=beta_lower, high=beta_upper, size=10000)  
            beta = np.random.choice(beta_uniform, size=1)
            # print(f"beta value================================== {beta}")

            # --- Vectorized with np.where ---
            log_a_series = np.log(df_nonzero_city[current_stock_col] / df_nonzero_city[current_pop_col]) 
            df_nonzero_city['stocks_at_reg'] = np.exp(log_a_series) * (df_nonzero_city[current_pop_col] ** beta)
            df_nonzero_city['stocks_at_t2'] = np.where(
                df_nonzero_city[next_pop_col] > df_nonzero_city[current_pop_col],
                np.exp(log_a_series) * (df_nonzero_city[next_pop_col] ** beta),
                0
            )
            
            df_nonzero_city.loc[:,'added_stock'] = np.where(df_nonzero_city['stocks_at_t2'] == 0, 0, (df_nonzero_city['stocks_at_t2'] - df_nonzero_city['stocks_at_reg']))
            df_nonzero_city.loc[:,'stock_next_time_step'] = (df_nonzero_city[current_stock_col] + df_nonzero_city['added_stock'])

            # population cannot be a fraction. When Population drops below 1, it generates huge per capita infrastructure
            # However, using 0 as population will result in infinite values for per capita infrastructure
            # So, we replace population values less 1 by 1 to get the infrastructure value
            # Function to apply conditions and assign values to a new column
            def replace_frac_population_by_one(row):
                if (row[next_pop_col] < 1):
                    return (row[current_stock_col] + row['added_stock']) / 1 
                else:
                    return (row[current_stock_col] + row['added_stock']) / np.round(row[next_pop_col],0)

            # Apply the function to create a new column based on conditions
            df_nonzero_city['per_cap_mass_next_time_step'] = df_nonzero_city.apply(replace_frac_population_by_one, axis=1)

            df_stocks_iter = pd.concat([df_stocks_iter, df_nonzero_city[['per_cap_mass_next_time_step', 'stock_next_time_step']]], axis=1)

    return df_stocks_iter

###############################################################################################################################################

def process_stock_at_t(df_for_analysis, current_stock_col, current_pop_col, next_pop_col, t, case, infrastructure, random_state):
    '''
    input:
    - current total infrastructure stock per city
    - current population 
    - prjected population for the next time step
    - next time step as year 
    
    '''
    print(df_for_analysis.shape)
    # run the fitting model for each city type
    df_t2_urban = find_next_stock(df_for_analysis, current_stock_col, current_pop_col, next_pop_col, list_of_city_types[0],t, random_state)
    df_t2_suburban = find_next_stock(df_for_analysis, current_stock_col, current_pop_col, next_pop_col, list_of_city_types[1],t, random_state)
    df_t2_periurban = find_next_stock(df_for_analysis, current_stock_col, current_pop_col, next_pop_col, list_of_city_types[2],t, random_state)
    df_t2_rural = find_next_stock(df_for_analysis, current_stock_col, current_pop_col, next_pop_col, list_of_city_types[3],t, random_state)

    df_all_types = pd.concat([df_t2_urban, df_t2_suburban, df_t2_periurban, df_t2_rural], axis =0)

    per_cap_mass_at_t = 'per_cap_mass_at_' + str(t)
    surface_Res_at_t = 'surface_Res_at_' + str(t)

    # NOT USING MIN OR MAX VALUES
    if case == 'min':
        df_all_types[per_cap_mass_at_t] = df_all_types['per_cap_mass_next_time_step'].min(axis=1)
        df_all_types[surface_Res_at_t] = df_all_types['stock_next_time_step'].min(axis=1)
    elif case == 'mean':
        df_all_types[per_cap_mass_at_t] = df_all_types['per_cap_mass_next_time_step'].mean(axis=1)
        df_all_types[surface_Res_at_t] = df_all_types['stock_next_time_step'].mean(axis=1)
    elif case == 'max':
        df_all_types[per_cap_mass_at_t] = df_all_types['per_cap_mass_next_time_step'].max(axis=1)
        df_all_types[surface_Res_at_t] = df_all_types['stock_next_time_step'].max(axis=1)
    else:
        print("ERROR!!")

    
    df_for_analysis = df_for_analysis.merge(df_all_types[['GEOID', per_cap_mass_at_t, surface_Res_at_t]], on = 'GEOID')

    sqm_to_sqmile = 3.86102e-7 # conversion factor
    
    df = df_for_analysis
 
    if infrastructure == 'RBUV':

        HU_density_sqmi_at_t = 'HU_density_sqmi_at_' + str(t)
        HU_density_sqmi_at_t_1 = 'HU_density_sqmi_at_' + str(t-10)
        ua_population_at_t = 'ua_population_at_' + str(t)
        ua_population_at_t_1 = 'ua_population_at_' + str(t-10)
        citytype_at_t = 'citytype_at_' + str(t)
        citytype_at_t_1 = 'citytype_at_' + str(t-10)
    
        df['change_in_Res_vol'] = df[surface_Res_at_t] - df[current_stock_col]
        df['added_HUs_at_t'] = df['change_in_Res_vol'] / (df['avg_HU_size_Res_sqm']) 

        if current_stock_col == 'volume_Res_2020':
            df[HU_density_sqmi_at_t] = df['weighted_HU_density_sqmi'] + (df['added_HUs_at_t']/ (df['ALAND'] * sqm_to_sqmile))
            df[ua_population_at_t] = df['population_ua_max'] * (1+(df[next_pop_col].round(0) - df[current_pop_col].round(0))/df[current_pop_col].round(0))
            citytype_at_t_1 = 'city type'
        else:
            df[HU_density_sqmi_at_t] = df[HU_density_sqmi_at_t_1] + (df['added_HUs_at_t']/ (df['ALAND'] * sqm_to_sqmile))
            df[ua_population_at_t] = df[ua_population_at_t_1] * (1+(df[next_pop_col].round(0) - df[current_pop_col].round(0))/df[current_pop_col].round(0))
            citytype_at_t_1 = citytype_at_t_1

        print(df.shape)
        # ======== REDEFINE URBAN ==========================            
        # --------- Vectorized version using np.select ---------
        redefine_cond1 = (
            (df[ua_population_at_t] >= 500000) &
            (df[HU_density_sqmi_at_t] >= 4000) &
            (df['ua-to-place allocation factor_max'] > 0.009)
        )

        redefine_cond2 = (
            (df[next_pop_col] >= 50000) &
            (df[HU_density_sqmi_at_t] >= 5000) &
            (df['ua-to-place allocation factor_max'] < 0)
        )

        redefine_cond3 = (
            (df[next_pop_col] >= 10000) &
            (df[HU_density_sqmi_at_t] >= 10000) &
            (df['ua-to-place allocation factor_max'] < 0)
        )

        choices = ['urban', 'urban', 'urban']
        default = df[citytype_at_t_1]

        # Timing the np.select version
        start_time = time.time()
        df[citytype_at_t] = np.select([redefine_cond1, redefine_cond2, redefine_cond3], choices, default=default)

        print("Places that shifted to urban")
        print(df[df[citytype_at_t_1] != df[citytype_at_t]][['State', 'NAMELSAD']])
        print(df[citytype_at_t].value_counts(), df[citytype_at_t_1].value_counts())
        print(df.shape)
        
        return df[['GEOID', per_cap_mass_at_t, surface_Res_at_t ,  HU_density_sqmi_at_t, ua_population_at_t, citytype_at_t]]
    
    else:
        return df[['GEOID', per_cap_mass_at_t, surface_Res_at_t]] 

