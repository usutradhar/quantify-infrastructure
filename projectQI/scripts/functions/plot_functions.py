# required packages
import geopandas as gpd
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


def plot_bounds(df, index_col, column_name):
    '''
    Function to plot the 50 U.S. states with Alaska and Hawaii 

    Parameters:
    - colum_name : list of column names to plot, columns are categorical 
    - colors: create a dictionary of column class and their respective colors, e.g.,
        colors = {
            'decreasing': 'red',
            'increasing': 'darkgreen', 
            'no trend': 'goldenrod'
        }
    
    '''
    # Import cartographic base maps: Cartographic boundary for plotting
    US_counties_cb = gpd.read_file(r'D:\Work\Box Sync\Trends_all states\Maps_2020\cb_2020_us_county_5m.zip') # tl_2020_us_county
    US_states_cb = gpd.read_file(r'D:\Work\Box Sync\Trends_all states\Maps_2020\cb_2020_us_state_5m.zip')  # tl_2020_us_state
    # source: https://gis.stackexchange.com/questions/141580/which-projection-is-best-for-mapping-the-contiguous-united-states
    US_states_cb = US_states_cb.to_crs('EPSG:9311')
    US_counties_cb = US_counties_cb.to_crs('EPSG:9311')
    cmap = 'viridis'
    
    # import the United States shape file
    # set state code as index, exclude states that we will never display
    gdf = df.set_index(index_col) #.drop(index=['02', 'VI', 'MP', 'GU', 'AS'])
    mm = 1/(25.4)  # milimeters in inches
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8.27,8.27))

    for ax, col in zip(axes.flatten(), column_name):
        continental_ax = ax
        alaska_ax = ax.inset_axes([.01, .01, .28, .30])
        hawaii_ax = ax.inset_axes([.30, .01, .25, .2])
               
        # Set bounds to fit desired areas in each plot
        continental_ax.set_xlim(-2257388.37, 2695859.75)
        continental_ax.set_ylim(-2546944.04, 808080.9)

        alaska_ax.set_xlim(-4383115.86156959, -1515327.22537998)
        alaska_ax.set_ylim(1458524.37417163, 3919523.37426209)

        hawaii_ax.set_xlim(-5812090.887043, -5452538.83170424)   # complete bounds 
        hawaii_ax.set_ylim(-1064618.58269969, -431238.37803499)
               
               
        US_states_cb.plot(ax=continental_ax, facecolor = 'none', edgecolor='gray', linewidth=0.1)
        df[(df[index_col] != 'Alaska') & (df[index_col] != 'Hawaii')].plot(column=col, 
             ax=continental_ax,
             cmap=cmap, #matplotlib.colors.ListedColormap(colors), 
             legend = True)

        US_states_cb[US_states_cb['STATEFP'] == '02'].plot(facecolor = 'none', edgecolor='gray', linewidth=0.1, ax=alaska_ax)
        df[df[index_col] != 'Alaska'].plot(column=col, cmap=cmap, ax=alaska_ax) 

        US_states_cb[US_states_cb['STATEFP'] == '15'].plot(facecolor = 'none', edgecolor='gray', linewidth=0.1, ax=hawaii_ax)
        df[df[index_col] != 'Alaska'].plot(column=col, cmap=cmap, ax=hawaii_ax)

        # remove ticks
        for ax in [continental_ax, alaska_ax, hawaii_ax]:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.axis('off')
            plt.tight_layout()
            plt.axis('off') 
        
    plt.subplots_adjust(wspace=0, hspace=0.02)       
    fig.patch.set_visible(False)

    # fig.savefig(filepath + filename + '.png', dpi = 300,  bbox_inches='tight')



# =========== plot 50 U.S. states ==================
def plot_50states(df, column_name, colors, filepath = None, filename = None ):
    '''
    Function to plot the 50 U.S. states with Alaska and Hawaii 

    Parameters:
    - colum_name : list of column names to plot, columns are categorical 
    - colors: create a dictionary of column class and their respective colors, e.g.,
        colors = {
            'decreasing': 'red',
            'increasing': 'darkgreen', 
            'no trend': 'goldenrod'
        }
    
    '''
    # Import cartographic base maps: Cartographic boundary for plotting
    US_counties_cb = gpd.read_file(r'D:\Work\Box Sync\Trends_all states\Maps_2020\cb_2020_us_county_5m.zip') # tl_2020_us_county
    US_states_cb = gpd.read_file(r'D:\Work\Box Sync\Trends_all states\Maps_2020\cb_2020_us_state_5m.zip')  # tl_2020_us_state
    # source: https://gis.stackexchange.com/questions/141580/which-projection-is-best-for-mapping-the-contiguous-united-states
    US_states_cb = US_states_cb.to_crs('EPSG:9311')
    US_counties_cb = US_counties_cb.to_crs('EPSG:9311')

    
    # import the United States shape file
    # set state code as index, exclude states that we will never display
    gdf = df.set_index('STATEFP') #.drop(index=['02', 'VI', 'MP', 'GU', 'AS'])
    colors = colors
    cmap = matplotlib.colors.ListedColormap([t[1] for t in sorted(colors.items())]) #Sorting by keys before converting to list

    mm = 1/(25.4)  # milimeters in inches
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8.27,8.27))

    for ax, col in zip(axes.flatten(), column_name):
        continental_ax = ax
        alaska_ax = ax.inset_axes([.01, .01, .28, .30])
        hawaii_ax = ax.inset_axes([.30, .01, .25, .2])
               
        # Set bounds to fit desired areas in each plot
        continental_ax.set_xlim(-2257388.37, 2695859.75)
        continental_ax.set_ylim(-2546944.04, 808080.9)
        continental_ax.set_title((x_labels[col]).strip('\"'), fontsize = 6, 
                     loc='center',  y=-0.05)

        alaska_ax.set_xlim(-4383115.86156959, -1515327.22537998)
        alaska_ax.set_ylim(1458524.37417163, 3919523.37426209)

        hawaii_ax.set_xlim(-5812090.887043, -5452538.83170424)   # complete bounds 
        hawaii_ax.set_ylim(-1064618.58269969, -431238.37803499)
               
               
        US_states_cb.plot(ax=continental_ax, facecolor = 'none', edgecolor='gray', linewidth=0.1)
        df[(df['STATEFP'] != '02') & (df['STATEFP'] != '15')].plot(column=col, 
             ax=continental_ax,
             cmap=cmap, #matplotlib.colors.ListedColormap(colors), 
             legend = True,
             legend_kwds={'loc': 'lower right', 'fontsize': 6,
                                              'markerscale': 0.5})
        leg = continental_ax.get_legend()
        leg.set_bbox_to_anchor((0.06, 0.03, 0.7, 0.2))
        leg.get_frame().set_alpha(0.3)

        print(df[df['STATEFP'] == '02'][col].unique())
        US_states_cb[US_states_cb['STATEFP'] == '02'].plot(facecolor = 'none', edgecolor='gray', linewidth=0.1, ax=alaska_ax)
        df[df['STATEFP'] == '02'].plot(column=col, cmap=cmap, ax=alaska_ax) 

        US_states_cb[US_states_cb['STATEFP'] == '15'].plot(facecolor = 'none', edgecolor='gray', linewidth=0.1, ax=hawaii_ax)
        print(df[df['STATEFP'] == '15'][col].unique())
        if col == "future trend from SSP 2":
            df[df['STATEFP'] == '15'].plot(column=col, cmap=matplotlib.colors.ListedColormap(['darkgreen','goldenrod']), 
                                   ax=hawaii_ax,  missing_kwds = dict(color='black'))
        else:
            US_states_cb.plot(facecolor = 'none', edgecolor='gray', linewidth=0.1, ax=hawaii_ax)
            df[df['STATEFP'] == '15'].plot(column=col, cmap=cmap, 
                                   ax=hawaii_ax,  missing_kwds = dict(color='black'))
    

        # remove ticks
        for ax in [continental_ax, alaska_ax, hawaii_ax]:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.axis('off')
            plt.tight_layout()
            plt.axis('off') 
        
    plt.subplots_adjust(wspace=0, hspace=0.02)       
    fig.patch.set_visible(False)

    fig.savefig(filepath + filename + '.png', dpi = 300,
           bbox_inches='tight')
    # fig.savefig(r'D:\Work\Box Sync\NC Figures\all states trend SSP 2 and 4' + '.pdf', dpi = 300,
    #            bbox_inches='tight')
    




import matplotlib

def quantile_map_plot(df, list_of_columns, is_county = False, state_code =None, file_path= None, file_name= None):

    # Import cartographic base maps: Cartographic boundary for plotting
    US_counties_cb = gpd.read_file(r'D:\Work\Box Sync\Trends_all states\Maps_2020\cb_2020_us_county_5m.zip') # tl_2020_us_county
    US_states_cb = gpd.read_file(r'D:\Work\Box Sync\Trends_all states\Maps_2020\cb_2020_us_state_5m.zip')  # tl_2020_us_state
    # source: https://gis.stackexchange.com/questions/141580/which-projection-is-best-for-mapping-the-contiguous-united-states
    US_states_cb = US_states_cb.to_crs('EPSG:9311')
    US_counties_cb = US_counties_cb.to_crs('EPSG:9311')

    if len(list_of_columns) > 1:
        
        fig, axes = plt.subplots(nrows=1, ncols=len(list_of_columns), figsize=(6*len(list_of_columns),12)) #18,16 #len(list_of_columns)

        for ax, col in zip(axes.flatten(), list_of_columns):
            continental_ax = ax

            if is_county==True:
                US_counties_cb[US_counties_cb['STATEFP'] == state_code].plot(ax=continental_ax, facecolor = 'none', edgecolor='gray', linewidth=0.3)
                df[df['STATEFP'] == state_code].plot(ax= continental_ax, column = col, legend = True, linewidth=0.4, legend_kwds={'loc': 'lower left'}, 
                            cmap = matplotlib.colors.ListedColormap(['olive', 'red', 'purple',]))
                
            else:
                # Set bounds to fit desired areas in each plot
                continental_ax.set_xlim(-2257388.37, 2695859.75)
                continental_ax.set_ylim(-2546944.04, 808080.90)

                hawaii_ax = ax.inset_axes([.20, .01, .25, .2])

                US_states_cb.plot(ax=continental_ax, facecolor = 'none', edgecolor='gray', linewidth=0.3)
                df.plot(ax= continental_ax, column = col, legend = True, linewidth=0.4, legend_kwds={'loc': 'lower left'}, 
                            cmap = matplotlib.colors.ListedColormap(['olive', 'red', 'purple',]))
            
                # US_states_cb[US_states_cb['STATEFP'] == '15'].plot(facecolor = 'none', edgecolor='gray', linewidth=0.2, ax=hawaii_ax)
                # print(df_map[df_map['STATEFP'] == '15'][col].unique())
                # df_map[df_map['STATEFP'] == '15'].plot(column=col, cmap = matplotlib.colors.ListedColormap(['orange', 'purple']), 
                #                             ax=hawaii_ax,  linewidth=0.3, missing_kwds = dict(color='black'))

            ax.set_title(col)
            # remove ticks
            for ax in [continental_ax]: #, hawaii_ax]:
                ax.set_yticks([])
                ax.set_xticks([])
                ax.axis('off')
                plt.tight_layout()
                plt.axis('off') 
    else:
        fig, continental_ax = plt.subplots(figsize=(16,22)) #18,16
        col = list_of_columns
        US_counties_cb.plot(ax=continental_ax, facecolor = 'none', edgecolor='gray', linewidth=0.3)
        df.plot(ax= continental_ax, column = col, legend = True, linewidth=0.4, legend_kwds={'loc': 'lower left'}, 
                        cmap = matplotlib.colors.ListedColormap(['olive', 'red', 'purple',]))
        continental_ax.axis('off')

    