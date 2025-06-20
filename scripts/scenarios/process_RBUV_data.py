# process the data to fit in the scaling model

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

### Buildings data
### Surface and Volume from GHSL

def process_residential_builtup_volume_data():
    # reading data from multiple years
    def read_Data(path, merged_column_names, builtarea):
        df_merged = pd.DataFrame(columns = ['GEOID'])
        if builtarea == 'surface':
            divideby = 1 # area in sqmeter
        elif builtarea == 'volume':
            divideby = 1 # volume in m^3
        else:
            print('Check input format')

        merged_column_names.append(builtarea)

        pattern = os.path.join(path, f"*csv")
        for file in glob.glob(pattern):
            df = pd.read_csv(file, dtype={'GEOID':str})
            df['TOTALAREA_SQKM'] = df[['ALAND', 'AWATER']].sum(axis=1)/1000000
            df[builtarea] = df['sum']/divideby
            df = df.sort_values(by='GEOID').reset_index(drop=True)
            # print(df.columns)
            # print(file[-9:-4])
            print('Shape of the dataframe:', df.shape, 'and Total built area (volume) in squaremeters (m^3): ', df[builtarea].sum())
            df_merged = df_merged.merge(df, on='GEOID', suffixes = ('', file[-9:-4]), how = 'outer')

        return df_merged

    column_names = ['GEOID', 'NAMELSAD', 'sum']

    df_surface = read_Data(r'data\ghsl\Surface\\', column_names, 'surface')
    df_volume = read_Data(r'data\ghsl\Volume\\', column_names, 'volume')

    # To check the consecutive changes in area/ volumne in each 10 yr interval
    df_surface['change_80_90'] = (df_surface['surface_1990'] - df_surface['surface'])
    df_surface['change_90_00'] = (df_surface['surface_2000'] - df_surface['surface_1990'])
    df_surface['change_00_10'] = (df_surface['surface_2010'] - df_surface['surface_2000'])
    df_surface['change_10_20'] = (df_surface['surface_2020'] - df_surface['surface_2010'])

    df_volume['Vchange_80_90'] = (df_volume['volume_1990'] - df_volume['volume'])
    df_volume['Vchange_90_00'] = (df_volume['volume_2000'] - df_volume['volume_1990'])
    df_volume['Vchange_00_10'] = (df_volume['volume_2010'] - df_volume['volume_2000'])
    df_volume['Vchange_10_20'] = (df_volume['volume_2020'] - df_volume['volume_2010'])

    """### Non residential buildings data from GHSL for the year 2020"""

    df_non_res_S = pd.read_csv(r'data\ghsl\Non_res_surface\BuiltArea_GHSL_S_nres_2020.csv')
    df_non_res_S['GEOID'] = df_non_res_S['GEOID'].astype(str).str.rjust(7, '0')
    df_non_res_S.rename(columns = {'sum':'surface_nonRes_2020'}, inplace= True)
    df_non_res_S.columns
    df_non_res_V = pd.read_csv(r'data\ghsl\Non_res_surface\BuiltArea_GHSL_V_nres_2020.csv')
    df_non_res_V['GEOID'] = df_non_res_V['GEOID'].astype(str).str.rjust(7, '0')
    df_non_res_V.rename(columns = {'sum':'volume_nonRes_2020'}, inplace= True)
    df_non_res_V.columns

    """### Merge built are with nonResidential: Find residential only"""

    # Merge surface with volume and non residential
    # Here surface refers to total building are: res + non-res
    df = df_volume.merge(df_surface[['GEOID','surface', 'surface_1990', 'surface_1995', 'surface_2000', 'surface_2010', 'surface_2015', 'surface_2020',
                                    'change_80_90', 'change_90_00', 'change_00_10', 'change_10_20']], on = 'GEOID')
    df = df.merge(df_non_res_S[['GEOID', 'surface_nonRes_2020']], on ='GEOID')
    df = df.merge(df_non_res_V[['GEOID', 'volume_nonRes_2020']], on ='GEOID')
    df['surface_Res_2020'] = df['surface_2020'] - df['surface_nonRes_2020']
    df['volume_Res_2020'] = df['volume_2020'] - df['volume_nonRes_2020']

    # Residential height in meter
    df['height_building'] = df['volume_2020'] / df['surface_2020']
    df['floors'] = df['height_building']// 3.048

    df[['NAMELSAD','surface_2020', 'volume_2020','surface_nonRes_2020','surface_Res_2020','volume_nonRes_2020', 'volume_Res_2020']].describe()/1000

    df[df['volume_2020']/df['surface_2020'] >= 3][['NAMELSAD','surface_2020', 'volume_2020','surface_nonRes_2020','surface_Res_2020']].sort_values(by = 'surface_2020')

    print("Number of missing values in built area data: ===")
    df.isna().sum().sum()

    # Area of the place in sqkm
    df['landAreaSqkm'] = df['ALAND']/1000000
    # Percent of total area that has buildings
    df['percent_built_surface'] = df['surface_2020'] * 100/df['ALAND']

    df.isna().sum().sum()

    """### Add housing units and age from ACS"""

    df_YearBuilt = pd.read_csv(r'data\housing_data_ACS\HousingDataCleaned.csv', index_col = 0)
    df_YearBuilt['GEOID'] = df_YearBuilt['GEOID'].str[9:]

    df_YearBuilt[['GEOID','complete_plumbing', 'Year_Built']]

    df_YearBuilt[df_YearBuilt['GEOID'] == '0652582'][['GEOID', ' !!Total:', 'HUs_occupied', 'HUs_vaccant', 'NAME', 'complete_plumbing', 'Year_Built', 'Percent_Built']]

    df_YearBuilt_selected = df_YearBuilt[['GEOID', ' !!Total:', 'HUs_occupied', 'HUs_vaccant', 'NAME', 'HUs_Total', 'YB_>=_2020', 'YB_2010_2019', 'YB_2000_2009',
                'YB_1980_1999', 'YB_1960_1979', 'YB_1940_1959', 'YB_<=_1939','Year_Built', 'Percent_Built']]

    df_YearBuilt.columns

    # Area of the place in sqkm
    df['landAreaSqkm'] = df['ALAND']/1000000
    # Percent of total area that has buildings
    df['percent_built_surface'] = df['surface_2020'] * 100/df['ALAND']

    df.isna().sum().sum()

    """### Merge with surface-volume with HUs and age"""

    df_buildings = df.merge(df_YearBuilt_selected, on = 'GEOID')

    # get an weighted age for each place
    df_buildings['weighted_avg_age'] = ((2024-1939)*df_buildings['YB_<=_1939'] + (2024-1950)*df_buildings['YB_1940_1959'] + (2024-1970)*df_buildings['YB_1960_1979'] +
                                    (2024-1990)*df_buildings['YB_1980_1999'] + (2024-2004)*df_buildings['YB_2000_2009'] + (2024-2014)*df_buildings['YB_2010_2019'] +
                                    (2024-2022)*df_buildings['YB_>=_2020'])/100

    df_buildings[['STATEFP', 'NAMELSAD','YB_>=_2020','YB_2010_2019', 'YB_2000_2009', 'Year_Built', 'Percent_Built', 'weighted_avg_age']].isna().sum()

    """### Import population and attributes data from depopulation study"""

    df_population = pd.read_csv(r'data\population\forecasted_trend.csv', index_col  = 0)
    df_population['GEOID'] = df_population['GEOID'].astype(str).str.rjust(7,'0')

    df_attributes = pd.read_csv(r'data\population\df_attributes.csv', index_col  = 0)
    df_attributes['GEOID'] = df_attributes['GEOID'].astype(str).str.rjust(7,'0')

    df_pop_attr = df_population.merge(df_attributes[['GEOID', 'REGION', 'city type', 'weighted_HU_density_sqmi',
                                                    'ua-to-place allocation factor_max', 'population_ua_min',
                                                    'population_ua_max', 'median_income']], on = 'GEOID')

    df_attributes.shape, df_population.shape

    df_attributes.columns

    df_population[(df_population['ssp42020'] -df_population['CensusPop_20']) > 0][['NAMELSAD', 'CensusPop_20','ssp22020', 'ssp22030', 'ssp22040',
                                                                                'ssp22050', 'ssp22060','ssp22070', 'ssp22080', 'ssp22090', 'ssp22100']]

    df_population[['ssp22020', 'ssp22030', 'ssp22040', 'ssp22050','ssp22060', 'ssp22070', 'ssp22080', 'ssp22090','ssp22100', 'ssp12020', 'ssp12030', 'ssp12040', 'ssp12050', 'ssp12060','ssp12070',
                    'ssp12080', 'ssp12090', 'ssp12100', 'ssp42020', 'ssp42030', 'ssp42040', 'ssp42050', 'ssp42060','ssp42070', 'ssp42080', 'ssp42090',
                    'ssp42100']] =df_population[['ssp22020', 'ssp22030', 'ssp22040', 'ssp22050','ssp22060', 'ssp22070', 'ssp22080', 'ssp22090','ssp22100', 'ssp12020', 'ssp12030', 'ssp12040', 'ssp12050',
                    'ssp12060','ssp12070','ssp12080', 'ssp12090', 'ssp12100', 'ssp42020', 'ssp42030', 'ssp42040', 'ssp42050', 'ssp42060', 'ssp42070', 'ssp42080', 'ssp42090', 'ssp42100']].round(0)

    # Total population for SSP 2 for each 10 yr interval
    print('Total population in millions for 51 states: ===')
    df_population[['CensusPop_20','ssp22020', 'ssp22030', 'ssp22040', 'ssp22050', 'ssp22060','ssp22070', 'ssp22080', 'ssp22090', 'ssp22100']].sum() / 1000000

    """### Merge buildings with population"""

    building_with_pop_all = df_buildings.merge(df_pop_attr[['GEOID', 'State', 'REGION', 'ua-to-place allocation factor_max', 'population_ua_min',
                                                        'population_ua_max','label', 'future trend from SSP 2', 'CensusPop_20', 'city type','weighted_HU_density_sqmi','median_income',
                                                    'ssp22020', 'ssp22030', 'ssp22040', 'ssp22050','ssp22060', 'ssp22070', 'ssp22080',
                                                    'ssp22090','ssp22100', 'ssp12020', 'ssp12030', 'ssp12040', 'ssp12050', 'ssp12060','ssp12070',
                                                    'ssp12080', 'ssp12090', 'ssp12100', 'ssp42020', 'ssp42030', 'ssp42040', 'ssp42050', 'ssp42060',
                                                    'ssp42070', 'ssp42080', 'ssp42090', 'ssp42100']], on = 'GEOID', how='left')

    building_with_pop_all[building_with_pop_all['GEOID'].str.startswith('214800')][['GEOID', 'NAMELSAD','CensusPop_20', 'ssp22020', 'ssp22030', 'ssp22040', 'ssp22050']]

    building_with_pop_all.shape, df_buildings.shape, df_population.shape, df_attributes.shape

    building_with_pop_all[['CensusPop_20','ssp22020', 'ssp22030', 'ssp22040', 'ssp22050','ssp22060', 'ssp22070', 'ssp22080','ssp22090','ssp22100',
                        'ssp12020', 'ssp12030', 'ssp12040', 'ssp12050', 'ssp12060','ssp12070', 'ssp12080', 'ssp12090', 'ssp12100','ssp42020',
                        'ssp42030', 'ssp42040', 'ssp42050', 'ssp42060','ssp42070', 'ssp42080', 'ssp42090', 'ssp42100']].isna().sum().sum()

    building_with_pop_all['percentchangeinpop'] = np.abs((building_with_pop_all['ssp42020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20'])

    print("Shape of building dataframe", building_with_pop_all.shape[0])
    print("No of places within 1% variation in 2020 census population and ssp scenario 1, 2 and 4")
    print("ssp1", building_with_pop_all[np.abs((building_with_pop_all['ssp12020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) < 0.01].shape[0])
    print("ssp2", building_with_pop_all[np.abs((building_with_pop_all['ssp22020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) < 0.01].shape[0])
    print("ssp4", building_with_pop_all[np.abs((building_with_pop_all['ssp42020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) < 0.01].shape[0])
    print("No of places over 1% variation in 2020 census population and ssp scenario 1, 2 and 4")
    print("ssp1", building_with_pop_all[np.abs((building_with_pop_all['ssp12020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) >= 0.01].shape[0])
    print("ssp2", building_with_pop_all[np.abs((building_with_pop_all['ssp22020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) >= 0.01].shape[0])
    print("ssp4", building_with_pop_all[np.abs((building_with_pop_all['ssp42020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) >= 0.01].shape[0])
    print("No of places over 5% variation in 2020 census population and ssp scenario 1, 2 and 4")
    print("ssp1", building_with_pop_all[np.abs((building_with_pop_all['ssp12020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) >= 0.05].shape[0])
    print("ssp2", building_with_pop_all[np.abs((building_with_pop_all['ssp22020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) >= 0.05].shape[0])
    print("ssp4", building_with_pop_all[np.abs((building_with_pop_all['ssp42020'] - building_with_pop_all['CensusPop_20'])/building_with_pop_all['CensusPop_20']) >= 0.05].shape[0])

    # excluding cities that varies over 10% in population forecast in 2020 for ssp4
    building_with_pop = building_with_pop_all[(building_with_pop_all['percentchangeinpop'] <=.01)]

    # building_with_pop[building_with_pop['city type'] == 'not enough data'].head() #shape[0] # ['CensusPop_20'].sum()

    building_with_pop.groupby('city type')[['Percent_Built','weighted_HU_density_sqmi']].describe()

    building_with_pop['weighted_HU_density_sqmi'].isna().sum()

    condition = building_with_pop[' !!Total:'] != 0
    building_with_pop.loc[condition,'HU_density_sqmile'] = building_with_pop[' !!Total:'] / (3.86102e-7 * building_with_pop['ALAND'])
    building_with_pop.loc[condition,'avg_HU_size_Res_sqm'] = (building_with_pop['volume_Res_2020'] / building_with_pop[' !!Total:']) # measures avg res unit volume

    building_with_pop[['weighted_HU_density_sqmi','HU_density_sqmile','avg_HU_size_Res_sqm']].describe()

    print(building_with_pop.shape)

    print('Total NaNs in population and built area:===')
    print(building_with_pop[['CensusPop_20', 'ssp12040', 'ssp22040', 'ssp42040', 'surface_Res_2020']].isna().sum())
    print("Places with no built area:===", building_with_pop[building_with_pop['surface_Res_2020'] == 0].shape[0])
    print("Places with zero population in census 2020:===", building_with_pop[building_with_pop['CensusPop_20'] == 0].shape[0])
    print("Places with no available population forecast:===", building_with_pop[building_with_pop['ssp22040'].isnull()].shape[0])

    building_with_pop['per_cap__nonRes_mass_at_2020'] =  building_with_pop['volume_nonRes_2020'] / building_with_pop['CensusPop_20']
    building_with_pop['per_cap__total_mass_at_2020'] =  building_with_pop['volume_2020'] / building_with_pop['CensusPop_20']
    building_with_pop['per_cap_mass_at_2020'] =  building_with_pop['volume_Res_2020'] / building_with_pop['CensusPop_20']
    # building_with_pop['avg_HU_size_sqm_Res'] = building_with_pop['surface_Res_2020'] / building_with_pop['HUs_Total']
    stocks_with_pop = building_with_pop[building_with_pop['CensusPop_20'] != 0]
    stocks_with_pop = stocks_with_pop[stocks_with_pop['city type'] == 'suburban']
    print(stocks_with_pop.shape)

    building_with_pop['per_cap__nonRes_mass_at_2020'] =  building_with_pop['volume_nonRes_2020'] / building_with_pop['CensusPop_20']
    building_with_pop['per_cap__total_mass_at_2020'] =  building_with_pop['volume_2020'] / building_with_pop['CensusPop_20']
    building_with_pop['per_cap_mass_at_2020'] =  building_with_pop['volume_Res_2020'] / building_with_pop['CensusPop_20']
    # building_with_pop['avg_HU_size_sqm_Res'] = building_with_pop['surface_Res_2020'] / building_with_pop['HUs_Total']
    stocks_with_pop = building_with_pop[building_with_pop['CensusPop_20'] != 0]
    stocks_with_pop = stocks_with_pop[stocks_with_pop['city type'] == 'suburban']
    print(stocks_with_pop.shape)

    building_with_pop['city_type_order'] = building_with_pop['city type'].map({'urban': 1, 'suburban': 2, 'periurban': 3, 'rural': 4, 'not enough data': 5})
    building_with_pop['REGION_order'] = building_with_pop['REGION'].map({'Northeast': 1, 'Midwest': 2, 'West': 3, 'South': 4})

    building_with_pop['STATEFP'] = building_with_pop['STATEFP'].astype(str).str.rjust(2,'0')
    building_with_pop = building_with_pop[(building_with_pop['STATEFP'] != '02') & (building_with_pop['STATEFP'] != '60') & (building_with_pop['STATEFP'] != '66') & 
                                      (building_with_pop['STATEFP'] != '69') & (building_with_pop['STATEFP'] != '72') & (building_with_pop['STATEFP'] != '78')]

    data = building_with_pop[building_with_pop['city type'] != 'not enough data']

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharey=True)

    # Boxplot for Value1 and Value2 in the first subplot
    data[data['city type'] == 'urban'].groupby(['REGION_order','REGION'])['per_cap_mass_at_2020'].describe().T.boxplot(ax=axs[0, 0], vert=False, sym='+',showfliers=False,
                                                                            flierprops={'markersize': 4, 'markerfacecolor': 'fuchsia'})
    axs[0, 0].set_title('Urban')
    axs[0, 0].invert_yaxis()
    # # Boxplot for Value1 and Value3 in the second subplot
    data[data['city type'] == 'suburban'].groupby(['REGION_order','REGION'])['per_cap_mass_at_2020'].describe().T.boxplot(ax=axs[0, 1], vert=False, sym='+',showfliers=False,
                                                                            flierprops={'markersize': 4, 'markerfacecolor': 'fuchsia'})
    axs[0, 1].set_title('Suburban')
    # # Boxplot for Value2 and Value3 in the third subplot
    data[data['city type'] == 'periurban'].groupby(['REGION_order','REGION'])['per_cap_mass_at_2020'].describe().T.boxplot(ax=axs[1, 0], vert=False, sym='+',showfliers=False,
                                                                            flierprops={'markersize': 4, 'markerfacecolor': 'fuchsia'})
    axs[1, 0].set_title('Periurban')
    # # Boxplot for all three values in the fourth subplot
    data[data['city type'] == 'rural'].groupby(['REGION_order','REGION'])['per_cap_mass_at_2020'].describe().T.boxplot(ax=axs[1, 1], vert=False, sym='+',showfliers=False,
                                                                            flierprops={'markersize': 4, 'markerfacecolor': 'fuchsia'})
    axs[1, 1].set_title('Rural')
    # Adjust layout
    plt.tight_layout()

    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=11)
        # ax.tick_params(axis='both', which='minor', labelsize=11)
        # ax.axis("off")
        ax.grid(False)

    log_data = np.log(building_with_pop['per_cap_mass_at_2020'])
    fig, ax = plt.subplots(figsize = (6,4))

    # Plot
    # Plot histogram
    log_data.plot(kind = "hist", density = True, alpha = 0.65, bins = 45) # change density to true, because KDE uses density
    # Plot KDE
    log_data.plot(kind = "kde")

    # Quantile lines
    quant_5, quant_25, quant_50, quant_75, quant_95 = log_data.quantile(0.05), log_data.quantile(0.25), log_data.quantile(0.5), log_data.quantile(0.75), log_data.quantile(0.95)
    quants = [[quant_5, 0.6, 0.16], [quant_50, 1, 0.36],  [quant_95, 0.7, 0.7]] # [quant_25, 0.8, 0.26], [quant_75, 0.8, 0.46],
    for i in quants:
        ax.axvline(i[0], alpha = i[1], ymax = i[2], linestyle = ":")


    # X
    ax.set_xlabel("Per capita RBUV (m$^3$) in log scale")
    # Limit x range to 0-4
    x_start, x_end = -3, 19
    ax.set_xlim(x_start, x_end)
    # ax.set_xscale('log')

    # Y
    ax.set_ylim(0, 1)

    # Annotations
    ax.text(quant_5-.1, 0.17, "5$^{th}$", size = 10, alpha = 0.8)
    # ax.text(quant_25-.13, 0.27, "25th", size = 11, alpha = 0.8)
    ax.text(quant_50-.13, 0.37, "50$^{th}$", size = 10, alpha = 1)
    # ax.text(quant_75-.13, 0.47, "75th", size = 11, alpha = 0.8)
    ax.text(quant_95-.25, 0.57, "95$^{th}$ Percentile", size = 10, alpha =1)

    # Overall
    ax.grid(False)
    # Remove ticks and spines
    ax.tick_params(left = False, bottom = False)
    for ax, spine in ax.spines.items():
        spine.set_visible(False)

    # plt.show(block=False)

    """### Model comparision Pruned Nonpruned F_test"""

    building_with_pop['per_cap_mass_at_2020'] =  building_with_pop['volume_Res_2020'] / building_with_pop['CensusPop_20']
    # building_with_pop['avg_HU_size_sqm_Res'] = building_with_pop['surface_Res_2020'] / building_with_pop['HUs_Total']
    stocks_with_pop = building_with_pop[building_with_pop['CensusPop_20'] != 0]
    stocks_with_pop = stocks_with_pop[stocks_with_pop['city type'] == 'suburban']
    print(stocks_with_pop.shape)

    infra_col ='surface_Res_2020'
    pop_col = 'CensusPop_20'
    df = stocks_with_pop.copy()


    ### ============== PRUNING ============== ###
    # Set the threshold to the 90th percentile
    threshold = df[pop_col].quantile(0.9)
    # Filter the DataFrame to include only points in the to 10%
    df_pruned = df[df[pop_col] > threshold]
    print(df_pruned.shape)

    y = np.log(df[infra_col].astype(float))
    x = np.log(df[pop_col].astype(float))

    y_pruned = np.log(df_pruned[infra_col].astype(float))
    x_pruned = np.log(df_pruned[pop_col].astype(float))

    m01 = sm.ols("y ~ x", data={"y": y, "x": x}).fit()
    m02 = sm.ols("y_pruned ~ x_pruned", data={"y_pruned": y_pruned, "x_pruned": x_pruned}).fit()
    anova_results= anova_lm(m01, m02)
    print(anova_results)
    print('\n')

    if anova_results['Pr(>F)'][1] < 0.005:
        print(f"Since {anova_results['Pr(>F)'][1]} < 0.005, difference is significant")
        print("Built seperate models")

    return building_with_pop
