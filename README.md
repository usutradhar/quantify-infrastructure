# quantify-infrastructure
Quantifying the available housing, roadway and waterpipes in the U.S. cities in terms of volume / length

Notes abour scripts:
- Residential built up volume (RBUV) is extracted from Google Earth Engine Code editor using [extract GHSL data.txt](https://github.com/usutradhar/quantify-infrastructure/blob/main/0%20extract%20GHSL%20data.txt) script summarized per city
- Roadway network length (RL) is extracted from OpenStreetMap and coverted to dataframes summarized by roadway type for cities using [0_read_OSM_2_streetsdf_final.ipynb](https://github.com/usutradhar/quantify-infrastructure/blob/main/0_read_OSM_2_streetsdf_final.ipynb) & [0_read_OSM_2_streetsdf_CFT_final.ipynb](https://github.com/usutradhar/quantify-infrastructure/blob/main/0_read_OSM_2_streetsdf_CFT_final.ipynb)
- To measure current per capita RBUV and projcet future RBUV per decade based on population trends: [readGHSLData_volume_ssp2.ipynb](https://github.com/usutradhar/quantify-infrastructure/blob/main/readGHSLData_volume_ssp2.ipynb)
- To measure current per capita RL and projcet future RL per decade based on population trends [readRoadsPerCap_ssp2 copy.ipynb](https://github.com/usutradhar/quantify-infrastructure/blob/main/readRoadsPerCap_ssp2%20copy.ipynb)
- Final output dataset with per capita residential built volume (m3) and roadway length (m) at every decade time interval for scenario SSP 2 [df_ssp2_final.xlsx](https://github.com/usutradhar/quantify-infrastructure/blob/main/df_ssp2_final.xlsx)
- Remaining files are for plotting and summarizing the analysis results
