# Make plots
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


# plot density distributions of current infrastruture 
def plot_distribution(data, xlims = [-10, 25], ylims = [0,1], label_text = 'x_label', p95_position=4000):
    '''
    function description: plot density distributions of current infrastruture 
    data: 
    xlims: x axis limits
    ylims: y axis limits
    label_text:

    '''      
    fig, ax = plt.subplots(figsize = (7,5))

    # Plot
    # Plot histogram
    ax = sns.histplot(data, log_scale=True, bins = 45,linewidth=0.10, alpha = .50, kde=True, kde_kws = {'cut': 0},)
    ax.lines[0].set_color('tab:orange')

    mean_val = data.mean()

    # Quantile lines
    quant_5, quant_50, quant_95 = data.quantile(0.05), data.quantile(0.5), data.quantile(0.95)
    quants = [[quant_5, 0.6, 0.25], [quant_50, 1, 0.8],  [quant_95, 0.7, 0.95]] 
    for i in quants:
        ax.axvline(i[0], alpha = i[1], ymax = i[2], linestyle = ":", color = 'royalblue', linewidth = 2)
    # X
    mean_line = ax.axvline(mean_val,  color = 'black', linewidth = 2, ymax = 0.95, linestyle = "--", )
    ax.set_xlabel(label_text, fontsize=12)
    ax.set_ylabel("No of cities", fontsize=12)
    # Limit x range to 0-4
    x_start, x_end = xlims[0], xlims[1]  #-3, 19
    # ax.set_xlim(x_start, x_end)
    # Y
    # ax.set_ylim(0, 1)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    # ax.set_ylabel("")

    # Annotations
    ax.text(quant_5-.1, 1000, "5$^{th}$", size = 12, alpha = 1)
    ax.text(quant_50-quant_5, 2500, "50$^{th}$", size = 12, alpha = 1)
    ax.text(quant_95+quant_5, p95_position, "95$^{th}$ Percentile", size = 12, alpha =1)
    # ax.text(mean_val+quant_5/2, p95_position/2, "mean",rotation = 90, size = 12, alpha =1)

    # Overall
    ax.grid(False)
    # Remove ticks and spines
    ax.tick_params(left = False, bottom = False)
    for ax, spine in ax.spines.items():
        spine.set_visible(False)
    
    plt.savefig(r'outputfiles\figures\\' + label_text + '.png', dpi = 300, transparent=True,  bbox_inches='tight')
        
    plt.show()


# Plot the extent of burden as bar charts
# plot
import matplotlib as mpl
def plot_burden_bar(df, burdencol):
    mpl.rcParams['font.family'] = 'Arial'
    palette = ['red', 'lightseagreen', 'dimgrey', 'darkkhaki']
    # Set your custom color palette
    color_codes = sns.color_palette(palette, 4)
    fig, ax = plt.subplots( dpi=300)
    # the size of A4 paper
    fig.set_size_inches(4,3)
    if burdencol[:4] == 'RBUV':
        col_name = burdencol[:4] + ' ' + burdencol[-4:]
    elif burdencol[:2] == 'RL':
        col_name = burdencol[:2] + ' ' + burdencol[-4:]
    
    df_plot = df[df[burdencol] != 'noChange']
    print(df_plot.shape)
    df_grouped = df_plot.groupby([burdencol,'city type']).size().reset_index()
    df_grouped.columns = [col_name,  'city type', 'No of cities',]
    # print(df_grouped)
    sns.barplot(data = df_grouped, y=col_name, x ='No of cities', hue="city type", hue_order = ['urban', 'suburban', 'periurban', 'rural'], palette=color_codes,ax=ax)
    ax.set(xlim=(0, 14000))
    ax.invert_xaxis()
    plt.savefig(r'outputfiles\figures\burden_plot_' + col_name +'.png', transparent=True,  bbox_inches='tight')


# Plot sankey plots of burden extent
def plot_burden_sankey(df,trend_col,infra_col, output_path = r"outputfiles\figures\\"):

    df_parcats = df[df[trend_col] != 'noChange'].reset_index(drop=True)

    infra_dim = go.parcats.Dimension(
        values=df_parcats[infra_col], #categoryorder="category ascending",
        categoryarray= df_parcats[infra_col].unique().sort_values(), label= ''
    )

    future_trend_dim = go.parcats.Dimension(
        values=df_parcats[trend_col], label=" ", # Burden " + trend_col1[-4:],
        categoryarray = ['decreasingBurden','increasingBurden'],
    )

    df_parcats['color_col'] = df_parcats['city type']
    df_parcats['color_col'] = df_parcats['color_col'].map({'urban': 1, 'suburban': 2,'periurban':3, 'rural':4,})

    # Build colorscale
    color = [x for x in df_parcats['color_col']]
    colorscale =['red', 'lightseagreen', 'dimgrey', 'darkkhaki'] #

    # create figure object
    fig = go.Figure(
        data=[
            go.Parcats(
                dimensions=[
                    future_trend_dim,             
                    infra_dim,              
                    ],
                line={"color": color, "colorscale": colorscale, 'shape': 'hspline'},
                hoveron="color",
                hoverinfo="count + probability",
                labelfont={"size": 18,"family": "arial"},
                tickfont={"size":  16,"family": "arial"},
                arrangement="freeform",
                
            )
        ]
    )

    fig.update_layout(
        height=300,
        width=500,
        font=dict(size=20, ),
        margin=dict(l=100, r=50, t=20, b=20))
    # fig.update_traces(textposition='inside')
    fig
    # fig.write_html(output_path + str(trend_col) +'.html')

    fig.write_image(output_path + 'burden_extent_' + str(trend_col) + ".png", scale=4, engine="kaleido")