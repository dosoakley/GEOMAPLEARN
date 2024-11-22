import fold_clustering
import Array
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from hdbscan import HDBSCAN
import quadrillage
import geopandas as gpd
import plot_map
import fold_detection
import warnings
from dsom import SOM
from shapely.geometry import LineString

warnings.filterwarnings("ignore")

"""
Les fonctions de ce module constituent les différentes grandes étapes du traitement de détection de "plis géologiques" 
à partir d'une carte géologique vectorisée.
Elles s'enchainent linéairement. Les résultats de l'une étant les entrées de la suivante. Comme toutes ces fonctions 
enregistrent leur résultat sur disque, on peut reprendre le traitement à n'importe quel étape.
Les répertoires et fichiers sont tous identifiés par des constantes en début de ce module. 

Toutes ces fonctions sont construites sur le même modèle : 
    - on lance le chrono et on log le départ de l'étape..
    - on charge en mémoire les données à partir de fichiers précédemment sauvegardés sur disque 
    - on effectue le traitement proprement dit, en déléguant au maximum la logique du traitement à d'autres modules 
    spécialisés
    - on sauvegarde les résultats dans un fichier sur disque
    - on affiche éventuellement (graphiquement) le résultat  
    - on stoppe le chrono et on conclue le log 
Seuls les traitements proprement dit de la fonction et de l'éventuel affichage sont codés dans des fonctions spécifiques. 
Toutes les actions redondantes sont factorisées dans la fonction "englobante" <execute_step>. 

En plus de ces fonctions, ce module est complété par une partie exécutable "__main__" réduite, qui facilite l'exécution 
d'une partie de la chaine de traitement, en particulier elle permet de commencer le traitement à n'importe quelle étape.

Cette organisation à 2 avantages :
    - renforce la lisibilité du code sans surcharger le module principal 
    - permet de gagner du temps de test 
"""

default = r'.\default_result.shp'
tmp_path_name = r'.\tmp\\'

# ------------ TO CHANGE BEFORE EACH RUN -------------------------

NUMBER_OF_SAMPLE = 20 #Number of points per quadrant of the circle used to create the grid.
START_FROM_STEP_NUMBER = 0
SAVE_FIG = False  # true if you wanna save your figure automatically
OS = True  # if TRUE = Windows, False = Linux
cluster_algorithm = 'hdbscan' #Options are dbscan or hdbscan
cluster_fold_types_separately = True #If True, do the clustering separately for synclines and anticlines.
n_neighbors_LOF = 20 #n_neighbors argument for the LocalOutlierFactor outlier removal.
axes_som_win_min_max = [10,15] #Default [10,15]. Set lower to get more nodes, higher for fewer nodes.
use_low_dip_mask = True #Exclude folds with limb dips less than 5 degrees.
use_dip_direction_mask = True #Require that anticline limbs dip outward and syncline limbs dip inward.
use_interlimb_angle_mask = False #Exclude folds with an interlimb angle of less than 5 degrees.
allow_single_cluster = True #Allow hdbscan to find only one cluster for a given fold type (syncline or anticline).
save_clusters_before_LOF = True #This saves a file of the clusters after clustering but before LocalOutlierFactor outlier removal.

# ------------------------CHOOSE your case: uncomment the target case below -----------------------------------

# MAP_NAME = "Flat_Model"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\FlatModel\FlatModel.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "Flat_Model_EggBoxTopo"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\FlatModel_EggBoxTopo\FlatModel_EggBoxTopo.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "SingleFold"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\SingleFold\SingleFold.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "SingleFold_EggBoxTopo"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\SingleFold_EggBoxTopo\SingleFold_EggBoxTopo.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "AsymmetricFold"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\AsymmetricFold\AsymmetricFold.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "SigmoidalFold"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\SigmoidalFold\SigmoidalFold.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "MultipleFolds"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\MultipleFolds\MultipleFolds.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "MultipleFolds_EggBoxTopo"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\MultipleFolds_EggBoxTopo\MultipleFolds_EggBoxTopo.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "FoldTrain"
# INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\FoldTrain\FoldTrain.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

MAP_NAME = "DoubleFold"
INPUT_VECTOR_MAP_FILE = r'..\Synthetic_Maps\DoubleFold\DoubleFold.shp'
investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')
axes_som_win_min_max = [15,20]

# MAP_NAME = "Lavelanet"
# INPUT_VECTOR_MAP_FILE = r'..\Real_Maps\Lavelanet\Lavelanet_map.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=1000,min_samples=30,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "Lavelanet_alt1"
# INPUT_VECTOR_MAP_FILE = r'..\Real_Maps\Lavelanet\Lavelanet_map.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=500,min_samples=30,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "Lavelanet_alt2"
# INPUT_VECTOR_MAP_FILE = r'..\Real_Maps\Lavelanet\Lavelanet_map.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=1000,min_samples=15,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# MAP_NAME = "Esternay"
# INPUT_VECTOR_MAP_FILE = r'..\Real_Maps\Esternay\Esternay_map.shp'
# investigated_map = fold_detection.FoldDetection(epsilon=1000,min_samples=30,min_angle=60, map_type=MAP_NAME, id_fm='RelAge')

# --------------------------------------------------------------------------------------------------------------
NUMBER_OF_SAMPLE_STRING = str(NUMBER_OF_SAMPLE)
nodes = NUMBER_OF_SAMPLE_STRING + "_nodes_"
MAP_NAME, TMP_PATH, INPUT_VECTOR_MAP_FILE, DEFAULT_SAVE_FILE = \
    fold_detection.FoldDetection.get_path_for_file(MAP_NAME, OS, INPUT_VECTOR_MAP_FILE, default, tmp_path_name)

BEAM_FILE = TMP_PATH + nodes + investigated_map.map_type + "_rays.shp"
SEGMENT_FILE = TMP_PATH + nodes + investigated_map.map_type + r"_segments.shp"
LINE_MATCH_FILE = TMP_PATH + nodes + investigated_map.map_type + r"_Match_lines.shp"
POINT_MATCH_FILE = TMP_PATH + nodes + investigated_map.map_type + r"_Match_points.shp"
LINE_CLUSTER_FILE = TMP_PATH + nodes + investigated_map.map_type + r'_cluster_lines.shp'
POINT_CLUSTER_FILE = TMP_PATH + nodes + investigated_map.map_type + r'_cluster_points.shp'
FILENAME = MAP_NAME + "_" + nodes
FOLD_AXIS_FILE = TMP_PATH + nodes + investigated_map.map_type + r'_fold_axes.shp'
FOLD_AREA_FILE = TMP_PATH + nodes + investigated_map.map_type + r'_fold_area.shp'

# --------------------------------------------------------------------------------------------------------------


def execute_step(fct, args=None, titre=None, save_file=DEFAULT_SAVE_FILE, plot_fct=None):
    local_start_time = time.time()
    if titre is None:
        print("Execution de la fonction %s".format(fct), end='')
    else:
        print(titre + "...", end='')

    if args is None:
        result = fct()
    else:
        result = fct(**args)
    if type(result) is gpd.GeoDataFrame:
        result.to_file(save_file,engine='pyogrio')
    elif type(result) is np.ndarray:
        result.tofile(save_file)
    elif type(result) is tuple:
        for i in range(len(result)):
            result[i].to_file(save_file[i],engine='pyogrio') #For GeoDataFrame
        

    if plot_fct is not None:
        plot_fct()
        plt.draw()
    print("..terminé à " + time.strftime("%H:%M:%S", time.localtime()) + ". \nTemps d'exécution : %s secondes" % (
            time.time() - local_start_time))


def beam():
    """
    Cette fonction construit un ensemble (ou faisceau) de rayons : des segments générés de diverses manières en vue de
    balayer la carte géologique donnée en référence. Ce balayage ou maillage, peut prendre diverses formes (aléatoire,
    en "toiles d'araignée", régulier dans plusieurs directions, etc. Il peut également être serré ou distandu.
    Le choix du maillage adéquat dépend du problème à résoudre.

    BeamGridSpider: construit une grille à 360° en toile d'araignée
    :return: les faisceaux de rayons sous la forme d'un GeoDataFrame
    """
    geol_map = Array.get_data_from_vector_map(INPUT_VECTOR_MAP_FILE)
    #  ----- Circular grid  ------
    beam_grid = quadrillage.CircleGrid(NUMBER_OF_SAMPLE)
    grid_gdf = beam_grid.get_arranged_rays(geol_map)

    return grid_gdf


def segment():
    """
    Cette fonction calcule les intersections entre les géométries de la carte géologique et des rayons associés à cette
    carte, précalculés à l'étape précédente.
    Chaque ligne de DataFrame généré représente un morceau de rayon. C'est un segment au sens géométrique, qui
    traverse exactement une et une seule géométrie de ma carte géologique (formation). Ce morceau est donc associé à
    la formation géologique unique traversée.
    Les morceaux constituant un même rayon sont ordonnés via la colonne "rank" du DataFrame, d'après leur proximité
    au point géographique de coordonnées (0.0, 0.0).
    :return: un GeoDataFrame constitué des segments
    """
    # local_start = time.time()
    geol_map = Array.get_data_from_vector_map(INPUT_VECTOR_MAP_FILE)
    beam_df = Array.get_data_from_vector_map(BEAM_FILE)
    gdf_beams_and_map_merged = investigated_map.merge_beams_and_map(geol_map, beam_df,geol_map)
    gdf_to_tri= gdf_beams_and_map_merged.apply(fold_detection.FoldDetection.triPoint, axis=1)
    gdf_beams_and_map_merged = fold_detection.FoldDetection.removePoint(gdf_to_tri)
    gdf_beams_and_map_merged = gpd.GeoDataFrame(gdf_beams_and_map_merged,geometry=investigated_map.intersect_geom)
    if SAVE_FIG:
        plot_map.plot_geol_map(geol_map)

    return gdf_beams_and_map_merged


def match():
    """
    1) Create numerical index and convert them in a three letters alphabetic values (three distinct digits)
    2) analyze symmetry in each beams with regex language and transform them to segments
    3) create a gdf with one geometry per matched symmetry

    :return: a gdf with one geometry per symmetry. Because the first and last position of the symmetry are the same,
    the RGF NAME of the symmetry bears the name of this formation.
    """

    gdf = Array.get_data_from_vector_map(SEGMENT_FILE)
    geol_map = Array.get_data_from_vector_map(INPUT_VECTOR_MAP_FILE)
    # load the beams grid with (n-1) nodes in gdf
    gdf = gdf.rename(columns={"index_righ": "index_right"})
    gdf = gdf.set_index(['index_right', 'level_1'])
    ###########################
    # step 1)  Create numerical index and convert them in a three letters alphabetic values (three distinct digits)
    ###########################
    investigated_map.gdf = gdf
    gdf = investigated_map.create_id()
    # loss of the segments order during the alphaIds merge in "merge_alpha_ids"
    ordered_gdf = investigated_map.gdf.groupby(level=0, group_keys=False).apply(investigated_map.order_segment_by_rank)
    ###########################
    #  step 2) analyze symmetry in segments beams by beams in a for loop
    ###########################
    line_df,point_df = investigated_map.analyze_symmetry(ordered_gdf,geol_map)
    ###########################
    #  step 3) create a gdf with one geometry per matched symmetry
    ###########################
    investigated_map.gdf = gpd.GeoDataFrame(line_df, geometry=investigated_map.geom_col)
    investigated_map.gdf2 = gpd.GeoDataFrame(point_df, geometry=investigated_map.geom_col)
    plot_map.plot_match(investigated_map.gdf,geol_map)
    return investigated_map.gdf,investigated_map.gdf2

def end_pts_distance_function(p1,p2):
    """
    Metric to use for calculating distance between points using end points.
    This is the metric Thierry was originally using, but done without creating 
    a huge distance matric to make it faster and reduce memory requirements.
    Also, unlike his implementation, it doesn't separate by geological unit.
    In each p, the order of components should be: x1, y1, x2, y2.
    """
    dist1 = np.sqrt((p1[0]-p2[0])**2.+(p1[1]-p2[1])**2.)+np.sqrt((p1[2]-p2[2])**2.+(p1[3]-p2[3])**2.)
    dist2 = np.sqrt((p1[0]-p2[2])**2.+(p1[1]-p2[3])**2.)+np.sqrt((p1[2]-p2[0])**2.+(p1[3]-p2[1])**2.)
    metric = min(dist1,dist2)
    return metric

def clustering(epsilon=3200, neighboors=415):
    """
    Regroupe les segments "match" par similarité grâce à l'agorithme de clustering DBSCAN. La similarité est d'abord
    fondée sur la distance entre 2 segments, telle que calculée à l'étape précédente.
    :param epsilon: paramètre de DBSCAN - rayon du voisinnage ("hyper-boule") considéré pour chaque segment
    :param neighboors: paramètre de DBSCAN - nombre de voisins (i.e. segments) minimum nécessaires pour constituer un
     cluster
    :return:
    """

    geo_df_lines = Array.get_data_from_vector_map(LINE_MATCH_FILE)
    geo_df_points = Array.get_data_from_vector_map(POINT_MATCH_FILE)
    #Get the locations of the points.
    px = geo_df_points['geometry'].apply(lambda x: list(x.coords)[0][0]).values
    py = geo_df_points['geometry'].apply(lambda x: list(x.coords)[0][1]).values
    points = np.array([px,py]).T
    n1x,n1y,n1z = [geo_df_points['n1x'].values,geo_df_points['n1y'].values,geo_df_points['n1z'].values]
    n2x,n2y,n2z = [geo_df_points['n2x'].values,geo_df_points['n2y'].values,geo_df_points['n2z'].values]
    nh1 = np.sqrt(n1x**2.+n1y**2.)
    nh2 = np.sqrt(n2x**2.+n2y**2.)
    phi1 = np.arctan(np.abs(n1z)/nh1) #Plunge
    phi2 = np.arctan(np.abs(n2z)/nh2) #Plunge
    ftype = np.zeros(geo_df_points.shape[0],dtype=int)
    clusters = -1*np.ones(geo_df_points.shape[0],dtype=int)
    dip_direction_mask = ((geo_df_points['fold_type']=='anticline') & (geo_df_points['DipDirs']=='apart')) | (
        (geo_df_points['fold_type']=='syncline') & (geo_df_points['DipDirs']=='together'))
    if cluster_fold_types_separately:
        fold_types = ['anticline','syncline']
    else:
        fold_types = ['all']
    for ift,fold_type in enumerate(fold_types):
        mask = np.ones(phi1.shape,dtype=bool)
        if use_low_dip_mask:
            mask = mask & (phi1 < 85.0*np.pi/180.0) & (phi2 < 85.0*np.pi/180.0)
        if use_interlimb_angle_mask:
            theta = np.arccos(n1x*n2x+n1y*n2y+n1z*n2z) #Angle between the two bedding orientations.
            theta[theta>np.pi/2] = np.pi-theta[theta>np.pi/2] #Take the acute angle.
            mask = mask & (theta > 5.0*np.pi/180.0)
        #Try adding the syncline_dips and anticline_dips here.
        if use_dip_direction_mask:
            mask = mask & dip_direction_mask #Use this.
        if fold_type != 'all':
            mask = mask & (geo_df_points['fold_type'] == fold_type)
        if mask.sum() >= investigated_map.min_samples:
            if cluster_algorithm == 'dbscan':
                clusterer = DBSCAN(eps=epsilon, min_samples=investigated_map.min_samples,n_jobs=-1)
                temp_clusters = clusterer.fit_predict(points[mask,:])
            elif cluster_algorithm == 'hdbscan':
                clusterer = HDBSCAN(min_cluster_size=investigated_map.min_samples,min_samples=investigated_map.min_samples,
                                    cluster_selection_epsilon=epsilon, allow_single_cluster=allow_single_cluster)
                temp_clusters = clusterer.fit_predict(points[mask,:])
                plt.figure()
                clusterer.condensed_tree_.plot(select_clusters=True)
                plt.draw()
            else:
                print('Error: Clustering aglorithm not recognized.')
            temp_clusters[temp_clusters!=-1] = temp_clusters[temp_clusters!=-1]+clusters.max()+1
            clusters[mask] = temp_clusters
            ftype[mask] = ift

    if save_clusters_before_LOF:
        geo_df_points['cluster'] = clusters
        geo_df_points['fold_type'] = ftype
        geo_df_lines['cluster'] = clusters
        geo_df_lines['fold_type'] = ftype
        POINT_CLUSTER_FILE_PRELOF = TMP_PATH + nodes + investigated_map.map_type + r'_cluster_points_preLOF.shp'
        LINE_CLUSTER_FILE_PRELOF = TMP_PATH + nodes + investigated_map.map_type + r'_cluster_lines_preLOF.shp'
        geo_df_points.to_file(POINT_CLUSTER_FILE_PRELOF,engine='pyogrio')
        geo_df_lines.to_file(LINE_CLUSTER_FILE_PRELOF,engine='pyogrio')
        


    #Remove outlier from clusters based on their end points.:
    x1 = geo_df_lines['geometry'].apply(lambda x: list(x.coords)[0][0]).to_numpy()
    y1 = geo_df_lines['geometry'].apply(lambda x: list(x.coords)[0][1]).to_numpy()
    x2 = geo_df_lines['geometry'].apply(lambda x: list(x.coords)[1][0]).to_numpy()
    y2 = geo_df_lines['geometry'].apply(lambda x: list(x.coords)[1][1]).to_numpy()
    endpts = np.array([np.concatenate((x1,x2)),np.concatenate((y1,y2))]).T    
    endpt_clusters = np.concatenate((clusters,clusters))
    endpt_lines = np.concatenate((geo_df_lines.index,geo_df_lines.index)) #Line numbers corresponding to each index.
    for i in np.unique(endpt_clusters): 
        if i != -1:
            clf = LocalOutlierFactor(n_neighbors=n_neighbors_LOF)
            outliers = clf.fit_predict(endpts[endpt_clusters==i,:])
            line_nums = endpt_lines[endpt_clusters==i][outliers==-1]
            clusters[line_nums]=-1
            if np.count_nonzero(clusters==i)<investigated_map.min_samples:
                clusters[clusters==i] = -1 #Remove the cluster if it got too small.
    
    geo_df_points['cluster'] = clusters
    geo_df_points['fold_type'] = ftype
    geo_df_lines['cluster'] = clusters
    geo_df_lines['fold_type'] = ftype
    return geo_df_lines,geo_df_points


def plot_clusters():
    geol_map = Array.get_data_from_vector_map(INPUT_VECTOR_MAP_FILE)
    geo_df_clusters = Array.get_data_from_vector_map(LINE_CLUSTER_FILE)
    geo_df_clusters2 = Array.get_data_from_vector_map(POINT_CLUSTER_FILE)
    fold_clustering.plot_DBSCAN(geol_map, geo_df_clusters, investigated_map.map_type, investigated_map.epsilon, investigated_map.min_samples, SAVE_FIG, False) #Lines
    fold_clustering.plot_DBSCAN(geol_map, geo_df_clusters2, investigated_map.map_type, investigated_map.epsilon, investigated_map.min_samples, SAVE_FIG, True) #Points

def plot_axes():
    geol_map = Array.get_data_from_vector_map(INPUT_VECTOR_MAP_FILE)
    geo_df_clusters = Array.get_data_from_vector_map(FOLD_AXIS_FILE)
    fold_clustering.plot_Fold_Axes(geol_map, geo_df_clusters, investigated_map.map_type, investigated_map.epsilon, investigated_map.min_samples, SAVE_FIG)

def fold_axes():
    df = Array.get_data_from_vector_map(POINT_CLUSTER_FILE)
    px = df['geometry'].apply(lambda x: list(x.coords)[0][0]).values
    py = df['geometry'].apply(lambda x: list(x.coords)[0][1]).values
    clusters = df['cluster'].values
    ftype = df['fold_type'].values
    fold_axes = []
    clusters_to_keep = []
    fold_types = []
    clusters[df['max_angle']<investigated_map.min_angle] = -1 #Don't use these ones in finding the fold axis.
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters!=-1] #-1 is the outlier indicator, so remove it from consideration.
    for i in unique_clusters:
        #Fit the fold axis with a self-organizing map:
        #The values below should probably not be hard-coded.
        points = np.array([px[clusters==i],py[clusters==i]]).T
        ftype_i = ftype[clusters==i][0] #They should all be the same.
        if np.any(ftype[clusters==i] != ftype_i):
            print('Error: Multiple fold types in a single cluster.')
        n = 2 #Number of neurons along 1st dimension
        epochs = 1000#10000
        sigma = 1.0 #This is in units of chain index, not real world distance.
        dcoef=3.0
        win_min = axes_som_win_min_max[0]
        win_max = axes_som_win_min_max[1] #The defaults are too large and end up making the axes short.
        if points.size > 0:
            som1 = SOM(n=n, dim=2, sigma=sigma, dcoef=dcoef, split_chains=False, win_min=win_min, win_max=win_max)
            som1.fit(points,epochs=epochs)
            x = som1.weights[:,0]
            y = som1.weights[:,1]
            fold_axes.append(LineString([coords for coords in zip(x, y)]))
            clusters_to_keep.append(i)
            fold_types.append(ftype_i)
    d = {'cluster':clusters_to_keep,'geometry':fold_axes,'fold_type':fold_types}
    gdf = gpd.GeoDataFrame(d,geometry='geometry',crs=investigated_map.crs)
    return gdf
            
def plot_polygone_cluster():
    geol_map = Array.get_data_from_vector_map(INPUT_VECTOR_MAP_FILE)
    geo_df_clusters = Array.get_data_from_vector_map(FOLD_AREA_FILE)
    plot_map.plot_poly_cluster(geol_map, geo_df_clusters, investigated_map.map_type, investigated_map.epsilon, investigated_map.min_samples, SAVE_FIG)

def fold():
    investigated_map.gdf = Array.get_data_from_vector_map(LINE_CLUSTER_FILE)
    geo_df_clusters = investigated_map.associate_cluster_geom()
    return geo_df_clusters

STEPS = {
    0: (
        beam, None, "Calcul des faisceaux de rayons qui vont balyer et interroger la carte géologique", BEAM_FILE,
        None),
    1: (segment, None, "Calcul des segments par intersection des rayons et de la carte géologique", SEGMENT_FILE, None),
    2: (match, None, "Calcul des segments qui correspondent à des indices de plis", [LINE_MATCH_FILE,POINT_MATCH_FILE], None),
    3: (clustering, {'epsilon': investigated_map.epsilon, 'neighboors': investigated_map.min_samples}, "Calcul des clusters par DBSCAN", 
        [LINE_CLUSTER_FILE,POINT_CLUSTER_FILE],plot_clusters),
    4: (fold_axes, None, "Fitting fold axes",FOLD_AXIS_FILE,plot_axes),
    5: (fold, None, "Calcul de l'enveloppe des structures géologiques de plis", FOLD_AREA_FILE, plot_polygone_cluster)
}

if __name__ == "__main__":
    start_time = time.time()
    steps_number = len(STEPS)

    for i in range(START_FROM_STEP_NUMBER, steps_number):
        print("Etape n°%i : " % i, end='')
        execute_step(*STEPS[i])
    plt.show()

    print("\nTemps d'exécution global : %s secondes" % (time.time() - start_time))
