import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def define_colors(one_line_gdf, labels):
    """
    set the labels color for a given colormap from maptplotlib
    :param one_line_gdf: gdf with the target geometries
    :param labels: labels resulting from the clustering
    :return: a list of RGB colors set to the one_line_gdf length
    """
    one_line_gdf['cluster'] = labels.astype('str')
    df_sans_noise = one_line_gdf[one_line_gdf['cluster'] != '-1']
    NUM_COLORS = len(pd.unique(df_sans_noise.cluster))
    colors = [plt.cm.tab10(each) for each in np.linspace(0, 1, NUM_COLORS)]
    return colors

def plot_DBSCAN(geol_map, one_line_gdf, map_type, eps, mini, save_fig, points):
    """
    plot the resulting clusters of previous clustering
    :param geol_map: background map
    :param one_line_gdf: geodataframe with the geometry
    :return:nothing, it is  just a plotting function
    """

    labels = one_line_gdf['cluster']
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    colors = define_colors(one_line_gdf, labels)
    df_sans_noise = one_line_gdf[one_line_gdf['cluster'] != '-1']
    noise = one_line_gdf[one_line_gdf['cluster'] == '-1']
    fig, ax = plt.subplots(1, 1)
    geol_map.plot(ax=ax, color='k', edgecolor='black', alpha=0.25, figsize=(10, 10))
    rgf_unique = pd.unique(df_sans_noise['cluster']).tolist()

    if save_fig:

        if not noise.empty:
            if points:
                noise.plot(ax=ax, color='black', alpha=0.05, linestyle=None, marker='.')
            else:
                noise.plot(ax=ax, edgecolor='black', alpha=0.05)
        for color, (index, group), rgf_u in zip(colors, df_sans_noise.groupby(['cluster']), rgf_unique):
            # cluster_ind = df_sans_noise[df_sans_noise['cluster'] == index]
            if points:
                group.plot(ax=ax, label=rgf_u, colors=color, linestyle=None, marker='.')
            else:
                group.plot(ax=ax, label=rgf_u, colors=color)
        ax.set_axis_off()
    else:

        if not noise.empty:
            if points:
                noise.plot(ax=ax, color='black', label='noise', legend_kwds={'loc': 'upper left'}, alpha=0.05, marker='.')
            else:
                noise.plot(ax=ax, edgecolor='black', alpha=0.05, label='noise', legend_kwds={'loc': 'upper left'})
        for color, (index, group), rgf_u in zip(colors, df_sans_noise.groupby(['cluster']), rgf_unique):
            # cluster_ind = df_sans_noise[df_sans_noise['cluster'] == index]
            # group.plot(ax=ax, label=rgf_u, colors=color)
            if points:
                group.plot(ax=ax, color=color, marker='.')
            else:
                group.plot(ax=ax, colors=color)
                # group.plot(ax=ax, color=color,alpha=0.25)
                # Alternative method to plot end points only:
                # indices = group.index
                # for ind in indices:
                #     data = list(group.geometry[ind].coords)
                #     ax.plot([data[0][0],data[1][0]],[data[0][1],data[1][1]],color=color,marker='.',linestyle='')
        plt.title('4. {} clustering'.format(map_type))
        textstr = '\n'.join((r'Number of clusters: {}'.format(n_clusters_),
                             r'eps: {}'.format(eps),
                             r'min_sample: {}'.format(mini)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.5, 0, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='center_baseline', horizontalalignment='center', bbox=props)

        plt.legend(loc='best')
    ax.set_axis_off()
    plt.show()

def plot_Fold_Axes(geol_map, one_line_gdf, map_type, eps, mini, save_fig):
    """
    plot the fold axes for each cluster
    :param geol_map: background map
    :param one_line_gdf: geodataframe with the geometry
    :return:nothing, it is  just a plotting function
    """

    labels = one_line_gdf['cluster']
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    colors = define_colors(one_line_gdf, labels)
    df_sans_noise = one_line_gdf[one_line_gdf['cluster'] != -1]
    fig, ax = plt.subplots(1, 1)
    geol_map.plot(ax=ax, color='k', edgecolor='black', alpha=0.25, figsize=(10, 10))
    rgf_unique = pd.unique(df_sans_noise['cluster']).tolist()

    if save_fig:
        for color, (index, group), rgf_u in zip(colors, df_sans_noise.groupby(['cluster']), rgf_unique):
            group.plot(ax=ax, label=rgf_u, colors=color)
        ax.set_axis_off()
    else:
        for color, (index, group), rgf_u in zip(colors, df_sans_noise.groupby(['cluster']), rgf_unique):
            group.plot(ax=ax, colors=color)
        plt.title('4. {} clustering'.format(map_type))
        textstr = '\n'.join((r'Number of clusters: {}'.format(n_clusters_),
                             r'eps: {}'.format(eps),
                             r'min_sample: {}'.format(mini)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.5, 0, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='center_baseline', horizontalalignment='center', bbox=props)

        # plt.legend(loc='best')
    ax.set_axis_off()
    plt.show()
