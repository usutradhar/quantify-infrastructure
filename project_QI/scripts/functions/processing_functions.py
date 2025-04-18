import time
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import geopandas as gpd

# ========== Extracting roadway network from osm: Convert csv extracted ============================# 
# Convert df to gdf
# https://stackoverflow.com/questions/71907567/valueerror-geodataframe-does-not-support-multiple-columns-using-the-geometry-co
# Coordinate system of osm: http://download.geofabrik.de/osm-data-in-gis-formats-free.pdf
# must have a geometry column in csv
# 
def csv_2_gdf(filename):
    df = pd.read_csv(filename, encoding='cp1252')
    gdf = gpd.GeoDataFrame(df.loc[:, [c for c in df.columns if c != "geometry"]], 
                           geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
                           crs='EPSG:4326',)
    return gdf



# ======== convert excel sheets to seperate csvs and save 
# source: https://stackoverflow.com/questions/73378839/loop-through-excel-sheets-and-save-each-sheet-into-a-csv-based-on-a-condition
def xlsxTocsv(excel_file, save_filepath): 
    workbook = pd.read_excel(excel_file, sheet_name = None)
    
    for sheet_name in workbook.keys():
        header = 0 
        newdf = pd.read_excel(excel_file, sheet_name = sheet_name, header=header)
        # TODO: handle the case where sheet name is not a valid file name
        newdf.to_csv(save_filepath + f"{sheet_name}.csv", decimal = '.', index = False)
        


# combine stateID with SHRP id to create unique ids
def combine_state_SHRP_ID(df):
    return df['STATE_CODE'].astype(str).str.rjust(2,'0') + '_' +  df['SHRP_ID'].astype(str)
    

# ================ change date time to date time ==============
def process_csv(filename):
    df = pd.read_csv(filename)
    df['STATE_SHRP_ID'] = combine_state_SHRP_ID(df)

    # Check whether a date column exists
    if 'SURVEY_DATE' in df.columns:
        df['survey_Year'] = pd.to_datetime(df['SURVEY_DATE']).dt.year
    if 'CONSTRUCTION_DATE' in df.columns:
        df['construction_Year'] = pd.to_datetime(df['CONSTRUCTION_DATE']).dt.year
    else:
        pass
    return df



def calculate_pca_tsne(dataframe, n_components_pca=2, n_components_tsne=2, perplexity=30, random_state=42):
    """
    Function to calculate PCA and t-SNE values for a given dataframe.

    Parameters:
    - dataframe: pandas DataFrame, input data
    - n_components_pca: int, number of components for PCA (default=2)
    - n_components_tsne: int, number of components for t-SNE (default=2)
    - perplexity: float, perplexity parameter for t-SNE (default=30)
    - random_state: int, random seed for reproducibility (default=42)

    Returns:
    - pca_result: pandas DataFrame, result of PCA
    - tsne_result: pandas DataFrame, result of t-SNE
    """

    # Perform PCA
    pca = PCA(n_components=n_components_pca, random_state=random_state)
    pca_result = pca.fit_transform(dataframe)

    # Perform t-SNE
    tsne = TSNE(n_components=n_components_tsne, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(dataframe)

    # Create DataFrames for results
    pca_columns = [f'PC{i+1}' for i in range(n_components_pca)]
    tsne_columns = [f't-SNE{i+1}' for i in range(n_components_tsne)]

    pca_result = pd.DataFrame(data=pca_result, columns=pca_columns)
    tsne_result = pd.DataFrame(data=tsne_result, columns=tsne_columns)

    return pca_result, tsne_result


import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as sm
from statsmodels.stats.anova import anova_lm

def find_scale_parameters(df, infra_col, pop_col):
    df['log_pop_col'] = np.log(df[pop_col])
    df['log_infra_col'] = np.log(df[infra_col])
    
    # Check plot how the log-log plot looks?
    # ADD A QUESTION ASKING WHETHER A PLOT OR NOT???
    sns.regplot(x='log_pop_col', y='log_infra_col', data=df, 
           y_jitter=.03, ci =None, scatter_kws={"s": 1})
    x = df['log_pop_col']
    y = df['log_infra_col']

    #run anova model to find confidence intervals
    # Fit the regression model
    model = sm.ols("y ~ x", data={"y": y, "x": x}).fit()

    # Display regression summary
    # print(model.summary())
    # Calculate confidence intervals using ANOVA
    anova_result = anova_lm(model)
    conf_int = model.conf_int(alpha=0.05, cols=None)

    # Display ANOVA results and confidence intervals
    print("\nConfidence Intervals: 'log(a)' and 'b' values---")
    print(conf_int)
    return conf_int




# Function to categorize values based on quartile range
def categorize_by_quartiles(grouped_df, target_column):
    # Calculate quartile values
    Q1 = grouped_df.quantile(0.25)
    Q3 = grouped_df.quantile(0.75)

    # Categorize values based on quartiles
    def categorize_value(value):
        if value < Q1:
            return 'Pressurized'
        elif value > Q3:
            return 'Oversized'
        else:
            return 'Balanced'

    # Apply categorization to target_column
    return target_column.apply(categorize_value)



    
    