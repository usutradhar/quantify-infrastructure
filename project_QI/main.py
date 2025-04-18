# import all packages
import sys
from io import StringIO
import numpy as np
import pandas as pd
import glob, os
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import math
import statsmodels.formula.api as sm
from statsmodels.stats.anova import anova_lm

from scripts.scenarios.process_RBUV_data import process_residential_builtup_volume_data
from scripts.scenarios.process_RL_data import process_roadway_length_data

import sys
# from scripts.functions.functions_scaling_vect import find_scale_parameters, find_next_stock, process_stock_at_t
from scripts.scenarios.project_infrastructure_ssp2 import project_infrastructure_ssp2


def main():

    # Redirect stdout to capture output
    original_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output

    building_with_pop = process_residential_builtup_volume_data()
    roads_clean = process_roadway_length_data()

    print("Running RBUV for scenario beta mean:")
    project_infrastructure_ssp2(input_df = building_with_pop,
                      check_na_columns = ['CensusPop_20', 'ssp22040', 'volume_Res_2020'],
                      current_stock_column = 'volume_Res_2020',
                      current_pop_column = 'CensusPop_20',
                      project_pop_columns = ['ssp22030', 'ssp22040', 'ssp22050', 'ssp22060', 'ssp22070', 'ssp22080', 'ssp22090', 'ssp22100'],
                      case = 'mean', infrastructure = 'RBUV',
                      output_path = r'outputfiles\outputs_beta_mean\\')
    
    print("RBUV will generate city types for each decade")
    print("Using new city types run RL for scenario 2")
    project_infrastructure_ssp2(input_df = roads_clean,
                    #   check_na_columns = ['CensusPop_20', 'ssp22040', 'volume_Res_2020'],
                      current_stock_column = 'cl_total_length_2020', # 'volume_Res_2020',
                      current_pop_column = 'CensusPop_20',
                      project_pop_columns = ['ssp22030', 'ssp22040', 'ssp22050', 'ssp22060', 'ssp22070', 'ssp22080', 'ssp22090', 'ssp22100'],
                      case = 'mean', infrastructure = 'RL',
                      output_path = r'outputfiles\outputs_beta_mean\\')


    # print("Running RBUV for scenario beta minimum:")
    # project_infrastructure_ssp2(input_df = building_with_pop,
    #                   check_na_columns = ['CensusPop_20', 'ssp22040', 'volume_Res_2020'],
    #                   current_stock_column = 'volume_Res_2020',
    #                   current_pop_column = 'CensusPop_20',
    #                   project_pop_columns = ['ssp22030', 'ssp22040', 'ssp22050', 'ssp22060', 'ssp22070', 'ssp22080', 'ssp22090', 'ssp22100'],
    #                   case = 'min', infrastructure = 'RBUV',
    #                   output_path = r'outputfiles\outputs_beta_min\\')


    # print("RBUV will generate city types for each decade")
    # print("Using new city types run RL for scenario 2")
    # project_infrastructure_ssp2(input_df = roads_clean,
    #                 #   check_na_columns = ['CensusPop_20', 'ssp22040', 'volume_Res_2020'],
    #                   current_stock_column = 'cl_total_length_2020', # 'volume_Res_2020',
    #                   current_pop_column = 'CensusPop_20',
    #                   project_pop_columns = ['ssp22030', 'ssp22040', 'ssp22050', 'ssp22060', 'ssp22070', 'ssp22080', 'ssp22090', 'ssp22100'],
    #                   case = 'min', infrastructure = 'RL',
    #                   output_path = r'outputfiles\outputs_beta_min\\')
    
    # print("Running RBUV for scenario beta maximum:")
    # project_infrastructure_ssp2(input_df = building_with_pop,
    #                   check_na_columns = ['CensusPop_20', 'ssp22040', 'volume_Res_2020'],
    #                   current_stock_column = 'volume_Res_2020',
    #                   current_pop_column = 'CensusPop_20',
    #                   project_pop_columns = ['ssp22030', 'ssp22040', 'ssp22050', 'ssp22060', 'ssp22070', 'ssp22080', 'ssp22090', 'ssp22100'],
    #                   case = 'max', infrastructure = 'RBUV',
    #                   output_path = r'outputfiles\outputs_beta_max\\')
    
    # print("RBUV will generate city types for each decade")
    # print("Using new city types run RL for scenario 2")
    # project_infrastructure_ssp2(input_df = roads_clean,
    #                 #   check_na_columns = ['CensusPop_20', 'ssp22040', 'volume_Res_2020'],
    #                   current_stock_column = 'cl_total_length_2020', # 'volume_Res_2020',
    #                   current_pop_column = 'CensusPop_20',
    #                   project_pop_columns = ['ssp22030', 'ssp22040', 'ssp22050', 'ssp22060', 'ssp22070', 'ssp22080', 'ssp22090', 'ssp22100'],
    #                   case = 'max', infrastructure = 'RL',
    #                   output_path = r'outputfiles\outputs_beta_max\\')

    print("All scripts done!")

    # Restore original stdout and write to file
    sys.stdout = original_stdout
    with open("output.txt", "w") as f:
        f.write(captured_output.getvalue())


if __name__ == "__main__":
    main()
