import fold_clustering
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def plot_poly_cluster(geol_map, geom_df, map_type, eps, mini, SAVE_FIG):
    fig, ax = plt.subplots(1, 1)
    colors = fold_clustering.define_colors(geom_df, geom_df.cluster)
    newcmp = ListedColormap(colors)
    for label, color in zip(geom_df.cluster, colors):
        dict_col = {label: color}
    geom_df['cluster'] = geom_df['cluster'].astype(int)
    if SAVE_FIG:
        geol_map.plot(ax=ax, color='k', alpha=0.1, figsize=(25, 25))
        geom_df.plot(ax=ax, column='cluster', legend=False, cmap=newcmp, categorical=True, alpha=0.5, edgecolor='black',
                     linewidth=2)
        ax.set_axis_off()
    else:
        geol_map.plot(ax=ax, color='k', alpha=0.1, figsize=(25, 25))
        geom_df.plot(ax=ax, column='cluster', legend=False, cmap=newcmp, categorical=True, alpha=0.5, edgecolor='black',
                     linewidth=2, legend_kwds={'loc': 'best'})
        ax.set_axis_off()
        # leg = ax.get_legend()
        # leg.set_bbox_to_anchor((0., 0., 0.2, 0.2))
        plt.title('5. {} convex_hull'.format(map_type))
        textstr = '\n'.join((r'eps: {}'.format(eps),
                             r'min_sample: {}'.format(mini)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        ax.text(0.45, 0, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left', bbox=props)
    plt.show()

def plot_geol_map(geol_map):
    ax = geol_map.plot(column='AGE_DEB', cmap='tab20')
    ax.set_axis_off()
    plt.show()


def plot_match(gdf, geol_map):
    ax = geol_map.plot(color='k', alpha=0.5, figsize=(25, 25), edgecolor='black')
    gdf.plot(ax=ax, legend=False, cmap='tab20c', categorical=True, alpha=0.5, edgecolor='black',
                 linewidth=6)
    # ax.set_axis_off()
    plt.show()
