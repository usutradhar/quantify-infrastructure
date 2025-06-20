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

def process_roadway_length_data():

    """### Read roadway lengths"""
    # folder_path = r'D:\Work\Box Sync\Quantify Infrastructure\Streets_df\All states'
    folder_path = r'data\osm\All states'

    # Get a list of all CSV files in the folder
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # Initialize an empty DataFrame to store concatenated data
    concatenated_df = pd.DataFrame()

    # Loop through each CSV file and concatenate them
    for file in csv_files:
        # print(file)
        file_path = os.path.join(folder_path, file)

        # Read the CSV file into a DataFrame
        df_state = pd.read_csv(file_path, index_col =0)
        # to check how many places are excluded
        # print(df_state.shape[0])
        # count = count + df_state.shape[0]
        df_state['GEOID'] = df_state['GEOID'].astype(str).str.rjust(7, '0')

        # Concatenate the DataFrame to the existing data
        concatenated_df = pd.concat([concatenated_df, df_state], ignore_index=True)
    print(concatenated_df.shape)

    # 13 missing places due to missing geometry
    # Total no of places for 50 states in 2020
    # 31249+13 = 31262

    concatenated_df[['GEOID', 'NAMELSAD',  'secondary', 'tertiary', 'unclassified', 'residential', 'cl_tertiary', 'cl_unclassified', 'cl_residential', 'cl_service',
                    'lane_m_tertiary', 'lane_m_unclassified', 'lane_m_residential', 'lane_m_service', 'lane_m_living_street']]

    df = concatenated_df.copy()

    # Calculate total roadway length and total centerline roadway length
    roadway_columns = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'living_street']
    cl_roadway_columns = ['cl_motorway', 'cl_trunk', 'cl_primary', 'cl_secondary', 'cl_tertiary', 'cl_unclassified', 'cl_residential']
    lane_m_roadway_columns = ['lane_m_motorway', 'lane_m_trunk', 'lane_m_primary','lane_m_secondary', 'lane_m_tertiary', 'lane_m_unclassified',
                            'lane_m_residential', 'lane_m_service', 'lane_m_track','lane_m_footway', 'lane_m_cycleway', 'lane_m_living_street']
    cl_local_columns = ['cl_unclassified', 'cl_residential', 'cl_service'] # 'cl_secondary'
    lane_m_local_columns =['lane_m_unclassified', 'lane_m_residential', 'lane_m_service','lane_m_living_street'] # 'lane_m_secondary', 'lane_m_track','lane_m_footway', 'lane_m_cycleway',
    footway_columns = ['footway','n_residential','n_footway']

    df['total_length'] = df[roadway_columns].sum(axis=1)
    df['cl_all_classes'] = df[cl_roadway_columns].sum(axis=1)
    df['lane_m_all_classes'] = df[lane_m_roadway_columns].sum(axis=1)
    df['cl_local_length'] = df[cl_local_columns].sum(axis=1)
    df['cl_total_length_2020'] = df[lane_m_local_columns].sum(axis=1)
    df['walkway_length'] = df[footway_columns].sum(axis=1)

    # df['road_area_m2'] = df['motorway'] * 13.6 + df['trunk'] * 9.6 + df['primary'] * 6.0 + df['secondary'] * 5.3 + df['tertiary'] * 4.9 +\
    #       df['unclassified'] * 4.5 + df['residential'] * 4.5
    df['road_lanearea_m2'] = df['lane_m_all_classes'] * 3.6

    df['pct_cl_local'] = df['cl_local_length'] * 100 / df['cl_all_classes']
    df['pct_local'] = df['cl_total_length_2020'] * 100 / df['lane_m_all_classes']
    df['pct_walkway'] = df['walkway_length']*100/df['cl_total_length_2020'] # This value need to be checked since residential (auto) can have walkways too

    print(f"Total cities with zero local roadways:==== {df[(df['cl_total_length_2020']==0)].shape[0]}")
    print(f"CDPs with zero local roadways:==== {df[(df['NAMELSAD'].str.contains('CDP')) & (df['cl_total_length_2020']==0)].shape[0]}")
    print(f"CDPs with local roadways:==== {df[(df['NAMELSAD'].str.contains('CDP')) & (df['cl_total_length_2020']!=0)].shape[0]}")
    print(f"Total non-zero local lane-meter available for {df[(df['cl_total_length_2020']!=0)].shape[0]} cities")
    # df[(df['NAMELSAD'].str.contains('CDP')) &(df['cl_total_length_2020']!=0)]

    df[(df['NAMELSAD'].str.contains('CDP')) &(df['cl_total_length_2020']!=0)][['GEOID', 'NAMELSAD', 'total_length', 'cl_all_classes', 'lane_m_all_classes', 'cl_total_length_2020','pct_local']].sort_values(by=['pct_local']).round(2)

    print(f"Total centerline length of local roadways: {df['cl_local_length'].sum().round(2)}")
    print(f"Total lane meter length of local roadways: {df['cl_total_length_2020'].sum().round(2)}")

    # NaNs values in roadway length dataframe
    df.isna().sum().sum()

    """### Import population and attributes data from depopulation study"""

    df_population = pd.read_csv(r'data\population\forecasted_trend.csv', index_col  = 0)
    df_population['GEOID'] = df_population['GEOID'].astype(str).str.rjust(7,'0')

    df_attributes = pd.read_csv(r'data\population\df_attributes.csv', index_col  = 0)
    df_attributes['GEOID'] = df_attributes['GEOID'].astype(str).str.rjust(7,'0')

    df_pop_attr = df_population.merge(df_attributes[['GEOID', 'REGION', 'city type', 'weighted_HU_density_sqmi','median_income']], on = 'GEOID')
    print(df_population.shape, df_attributes.shape, df_pop_attr.shape)

    # Total population for SSP 2 for each 10 yr interval
    print('Total population in millions for 51 states: ===')
    df_population[['CensusPop_20','ssp22020', 'ssp22030', 'ssp22040', 'ssp22050', 'ssp22060','ssp22070', 'ssp22080', 'ssp22090', 'ssp22100']].sum() / 1000000

    """### Merge roads with population"""

    roads_with_pop_all = df.merge(df_pop_attr[['GEOID', 'State', 'REGION', 'ALAND', 'label', 'future trend from SSP 2', 'CensusPop_20', 'city type','weighted_HU_density_sqmi','median_income',
                                        'ssp22020', 'ssp22030', 'ssp22040', 'ssp22050','ssp22060', 'ssp22070', 'ssp22080',
                                        'ssp22090','ssp22100', 'ssp12020', 'ssp12030', 'ssp12040', 'ssp12050', 'ssp12060','ssp12070',
                                                    'ssp12080', 'ssp12090', 'ssp12100', 'ssp42020', 'ssp42030', 'ssp42040', 'ssp42050', 'ssp42060',
                                                    'ssp42070', 'ssp42080', 'ssp42090', 'ssp42100']], on = 'GEOID', how='left')
    print(roads_with_pop_all.shape)

    print(roads_with_pop_all[roads_with_pop_all['GEOID'].str.startswith('214800')][['GEOID', 'NAMELSAD','CensusPop_20', 'ssp22020', 'ssp22030', 'ssp22040', 'ssp22050']])
    # 2010 Census
    # Louisville/Jefferson County metro government (balance), Kentucky	597337
    # 2020 Census
    # Louisville city, Kentucky	246161
    # Louisville/Jefferson County metro government (balance), Kentucky	386884

    # 246161+386884 = 633045

    print('Total population in millions for 50 states: ===')
    print(roads_with_pop_all[['CensusPop_20', 'ssp22020', 'ssp22030', 'ssp22040', 'ssp22050', 'ssp22060','ssp22070', 'ssp22080', 'ssp22090', 'ssp22100']].sum() / 1000000)

    roads_with_pop_all['percentchangeinpop'] = np.abs((roads_with_pop_all['ssp42020'] - roads_with_pop_all['CensusPop_20'])/roads_with_pop_all['CensusPop_20'])
    # cities that have their 2020 census population with 1% variation of the projected ssp22020 population are included in the analysis
    roads_with_pop = roads_with_pop_all[(roads_with_pop_all['percentchangeinpop'] <=.01)]
    print(roads_with_pop.shape)
    # roads_with_pop_all[roads_with_pop_all['percentchangeinpop'] > 0.01][['GEOID','NAMELSAD', 'CensusPop_20', 'ssp22020',]].sort_values(by='ssp22020')

    roads_with_pop['road_density_m-sqm'] = roads_with_pop['cl_total_length_2020'] /roads_with_pop['ALAND']
    roads_with_pop['road_density'] = roads_with_pop['road_lanearea_m2'] * 100/roads_with_pop['ALAND']

    # Exclude total zero roadways
    print(f"Total cities with nonzero local roadway lane meter {roads_with_pop[roads_with_pop['cl_total_length_2020']!=0].shape[0]}")

    """### Model comparision Pruned Nonpruned F_test"""

    roads_with_pop['per_cap_mass_at_2020'] =  roads_with_pop['cl_total_length_2020'] / roads_with_pop['CensusPop_20']

    stocks_with_pop = roads_with_pop[roads_with_pop['CensusPop_20'] != 0]
    stocks_with_pop = roads_with_pop[roads_with_pop['per_cap_mass_at_2020'] != 0]
    # stocks_with_pop = stocks_with_pop[stocks_with_pop['city type'] == 'suburban']
    print(stocks_with_pop.shape)

    infra_col ='cl_total_length_2020'
    pop_col = 'CensusPop_20'
    df = stocks_with_pop.copy()

    ### ============== PRUNING ============== ###
    # Set the threshold to the 90th percentile
    threshold_u = df[infra_col].quantile(0.9)
    threshold_l = df[infra_col].quantile(0.1)
    # Filter the DataFrame to include only points in the top 10%
    df_pruned = df[df[infra_col] < threshold_u]
    df_pruned = df_pruned[df_pruned[infra_col] > threshold_l]
    print(df_pruned.shape)

    y = np.log(df[infra_col].astype(float))
    x = np.log(df[pop_col].astype(float))

    y_pruned = np.log(df_pruned[infra_col].astype(float))
    x_pruned = np.log(df_pruned[pop_col].astype(float))

    m01 = sm.ols("y ~ x", data={"y": y, "x": x}).fit()
    print(m01.params)
    print("R-squared value" , m01.rsquared)
    m02 = sm.ols("y_pruned ~ x_pruned", data={"y_pruned": y_pruned, "x_pruned": x_pruned}).fit()
    anova_results= anova_lm(m01, m02)
    print(m02.params)
    print("R-squared value" , m02.rsquared)
    print(anova_results)
    print('\n')

    if anova_results['Pr(>F)'][1] < 0.005:
        print(f"Since {anova_results['Pr(>F)'][1]} < 0.005, difference is significant")
        print("Built pruned model")

    pd.set_option('mode.chained_assignment', None) # To stop SettingWithCopy warning
    list_of_city_types = ['urban', 'suburban', 'periurban', 'rural']

    print('Total NaNs in population and roadway length:===')
    print(roads_with_pop[['CensusPop_20', 'ssp22040', 'cl_total_length_2020']].isna().sum())
    print("Places with no roadway:===", roads_with_pop[roads_with_pop['cl_total_length_2020'] == 0].shape[0])
    print("Places with zero population in census 2020:===", roads_with_pop[roads_with_pop['CensusPop_20'] == 0].shape[0])
    print("Places with no available population forecast:===", roads_with_pop[roads_with_pop['ssp22040'].isnull()].shape[0])
    print(roads_with_pop.shape)

    roads_clean = roads_with_pop.dropna(subset=['CensusPop_20', 'ssp22040', 'total_length', 'cl_total_length_2020']).reset_index(drop=True)
    print(roads_clean.shape)
    roads_clean = roads_clean[roads_clean['cl_total_length_2020']!=0]
    print(roads_clean.shape)
    current_roadway_column = 'cl_total_length_2020' # 'road_area_m2' #

    roads_clean['per_cap_mass_at_2020'] = roads_clean[current_roadway_column] / roads_clean['CensusPop_20'].round(0)
    print("Shape of the clean dataset with nonzero values:==")
    print(roads_clean.shape)
    print(roads_clean[(roads_clean['per_cap_mass_at_2020'] > 10000) | (roads_clean['total_length'] < 500)][['GEOID','NAMELSAD','CensusPop_20','total_length', 'cl_total_length_2020','per_cap_mass_at_2020']])
    # roads_clean = roads_clean.merge(df_urban_rural_conn, on ='GEOID')
    roads_clean = roads_clean[(roads_clean['per_cap_mass_at_2020'] <= 10000) & (roads_clean['total_length'] >= 500)]
    print("Shape of the clean dataset with newly defined urban rural continuum at each time interval values:==")
    print(roads_clean.shape)

    roads_clean.groupby(['city type','REGION'])[['pct_local', 'pct_cl_local','per_cap_mass_at_2020']].median().sort_values(['city type','pct_cl_local']).round(2)

    roads_with_pop.shape, df.shape

    roads_clean['city_type_order'] = roads_clean['city type'].map({'urban': 1, 'suburban': 2, 'periurban': 3, 'rural': 4, 'not enough data': 5})
    roads_clean['REGION_order'] = roads_clean['REGION'].map({'Northeast': 1, 'Midwest': 2, 'West': 3, 'South': 4})

    roads_clean[roads_clean['per_cap_mass_at_2020']>1000][['GEOID', 'NAMELSAD', 'city type', 'REGION', 'ALAND', 'CensusPop_20','per_cap_mass_at_2020']].sort_values('per_cap_mass_at_2020').shape

    roads_clean.groupby('REGION')[['lane_m_motorway', 'lane_m_trunk', 'lane_m_primary', 'lane_m_secondary', 'lane_m_tertiary',
        'lane_m_unclassified', 'lane_m_residential', 'lane_m_service',
        'lane_m_living_street']].sum().sum(axis=1)

    roads_clean['pct_local_lane-m'] = roads_clean['cl_total_length_2020'] / roads_clean[['lane_m_motorway', 'lane_m_trunk', 'lane_m_primary', 'lane_m_secondary', 'lane_m_tertiary',
        'lane_m_unclassified', 'lane_m_residential', 'lane_m_service', 'lane_m_living_street']].sum(axis=1)

    roads_clean.groupby('REGION')['pct_local_lane-m'].describe()

    roads_clean[~roads_clean['NAMELSAD'].str.contains('CDP')].sort_values('pct_local_lane-m')[['GEOID', 'NAMELSAD','CensusPop_20','cl_total_length_2020','pct_local_lane-m', 'REGION', 'city type']].head(12)

    roads_clean.shape

    log_data = np.log(roads_clean[roads_clean['city type'] == 'rural']['per_cap_mass_at_2020'])
    # sns.histplot(np.log(data['per_cap_mass_at_2020']), alpha = 0.8, color= 'grey', linewidth=0.2)
    # citytype_name = ['urban', 'suburban', 'periurban', 'rural']

    fig, ax = plt.subplots(figsize = (6,4))

    # Plot
    log_data.plot(kind = "hist", density = True, alpha = 0.65, bins = 25) # change density to true, because KDE uses density
        # Plot KDE
    log_data.plot(kind = "kde")

    # Quantile lines
    quant_5, quant_25, quant_50, quant_75, quant_95 = log_data.quantile(0.05), log_data.quantile(0.25), log_data.quantile(0.5), log_data.quantile(0.75), log_data.quantile(0.95)
    quants = [[quant_5, 0.6, 0.16], [quant_50, 1, 0.36],  [quant_95, 0.45, 0.45]] # [quant_25, 0.8, 0.26], [quant_75, 0.8, 0.46],
    for i in quants:
        ax.axvline(i[0], alpha = i[1], ymax = i[2], linestyle = ":")

    # X
    ax.set_xlabel("Per capita RL (m) in log scale")
        # Limit x range to 0-4
    x_start, x_end = 0, 15
    # ax.set_xlim(x_start, x_end)

    # Y
    ax.set_ylabel("")

    # Annotations
    ax.text(quant_5-.1, 0.17, "5$^{th}$", size = 10, alpha = 0.8)
    # ax.text(quant_25-.13, 0.27, "25th", size = 11, alpha = 0.8)
    ax.text(quant_50-.13, 0.37, "50$^{th}$", size = 10, alpha = 1)
    # ax.text(quant_75-.13, 0.47, "75th", size = 11, alpha = 0.8)
    ax.text(quant_95-.25, 0.45, "95$^{th}$ Percentile", size = 10, alpha =1)

    # Overall
    ax.grid(False)
    # Remove ticks and spines
    ax.tick_params(left = False, bottom = False)
    for ax, spine in ax.spines.items():
        spine.set_visible(False)

    # plt.show(block=False)

    def find_example_scale_parameters(df, infra_col, pop_col):
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
        sns.set_style('ticks')
        fig, ax = plt.subplots(dpi=300)
        # the size of A4 paper
        fig.set_size_inches(5,3.5)
        # Check plot how the log-log plot looks?
        sns.regplot(ax=ax,x='log_pop_col', y='log_infra_col', data=df, line_kws={"color": "chocolate"}, color="steelblue",
                    y_jitter=.03, ci =None, scatter_kws={"s": 1})
        ax.set(xlabel='P = log(Population)', ylabel = 'Y = log(Infrastructure)')
        ax.set_ylim(4,17)
        # fig = ax.get_figure()
        # plt.savefig(r'D:\Work\Box Sync\Applications\Senseble lab MIT\Interview docs\scalingRelation.png', transparent=True,  bbox_inches='tight')

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

    find_example_scale_parameters(roads_clean[roads_clean['city type'] == 'suburban'],  'cl_total_length_2020', 'CensusPop_20')

    return roads_clean

