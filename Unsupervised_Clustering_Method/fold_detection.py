import numpy as np
import scipy as sp
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString, Polygon
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely import wkt
import sys
import regex as reg
from scipy.spatial import ConvexHull


class FoldDetection:
    POINT_ORIGIN = Point(-20000000, -200)

    def __init__(self, epsilon, min_samples, min_angle, map_type: str, id_fm: str = 'StratNum',
                 geom_col: str = 'geometry', crs: str = 'epsg:2154',
                 intersect_geom: str = 'intersection'):
        self.geom_col = geom_col
        self.intersect_geom = intersect_geom
        self.id_fm = id_fm
        self.crs = crs
        self.gdf = None
        self.gdf2 = None
        self.map_type = map_type
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.N_to_match = 2 #Number of units to be repeated in reverse order to detect a match.
        self.min_angle = min_angle #Minimum intersection angle allowed between ray and map polygon (degrees).
        self.N_points_orientation = None#10 #Number of points to use in the calculation of bedding orientations. Should be at least 3. Set to None to not use.
        self.max_point_dist = 250 #500 #Maximum distance in which to look for the N points for calculating the bedding orientation. Set to None to not use.
        self.dem = None #Digital elevation model (2D Numpy array of elevation values)

    @staticmethod
    def remove_double_slash(path: str):
        if '//' in path:
            path = path.replace('//', '/')
        elif '\\\\' in path:
            path = path.replace('\\\\', '\\')
        return path

    @staticmethod
    def get_transform_path_from_windows_to_linux(win_path: str):
        win_path_changed = win_path.replace('\\', "/")
        win_path_changed = win_path_changed.replace('/', '', 1)
        win_path_changed = win_path_changed.replace('.', '', 1)
        win_path_changed = win_path_changed.lstrip(' ')
        win_path_changed = FoldDetection.remove_double_slash(win_path_changed)
        return win_path_changed

    @staticmethod
    def get_path_for_file(map_type, os, input_file, default, TMP_PATH=r'.\Shapefile\tmp\\'):

        if not os:
            input_file = FoldDetection.get_transform_path_from_windows_to_linux(input_file)
            TMP_PATH = FoldDetection.get_transform_path_from_windows_to_linux(TMP_PATH)
            map_type = FoldDetection.get_transform_path_from_windows_to_linux(map_type)
            default = FoldDetection.get_transform_path_from_windows_to_linux(default)
            DEFAULT_PATH = TMP_PATH + default
            DEFAULT_PATH = FoldDetection.remove_double_slash(DEFAULT_PATH)
        else:
            TMP_PATH = FoldDetection.remove_double_slash(TMP_PATH)
            default = default.replace('.', '', 1)
            DEFAULT_PATH = TMP_PATH + default
            DEFAULT_PATH = FoldDetection.remove_double_slash(DEFAULT_PATH)

        return map_type, TMP_PATH, input_file, DEFAULT_PATH

    @staticmethod
    def removePoint(gdf):

        index_to_drop = gdf[gdf['toDrop'] == 1].index
        if len(index_to_drop) != 0:
            gdf.drop(index=index_to_drop, inplace=True)
        gdf.drop(columns=['toDrop'], inplace=True)
        return gdf

    @staticmethod
    def triPoint(line):

        if "POINT" in str(line):
            # line['geometry'] = 'indesirable'
            # line.drop(index=line.index, inplace=True)
            line['toDrop'] = 1
        else:
            line['toDrop'] = 0
        return line

    def merge_beams_and_map(self, background_map_gdf, beams_gdf, geol_map):

        """
        Cette fonction calcule les intersections entre les géométries de la carte géologique et des rayons associés à cette
        carte, précalculés à l'étape précédente.
        Chaque ligne de DataFrame généré représente un morceau de rayon. C'est un segment au sens géométrique, qui
        traverse exactement une et une seule géométrie de ma carte géologique (formation). Ce morceau est donc associé à
        la formation géologique unique traversée.
        Les morceaux constituant un même rayon sont ordonnés via la colonne "rank" du DataFrame, d'après leur proximité
        au point géographique de coordonnées (0.0, 0.0).
        :param background_map_gdf: gpd of the initial geological map
        :param beams_gdf: gpd with the grid of beams
        :return: un GeoDataFrame constitué des segments
        """
        # spatial join btw beams and the background map
        join_map = gpd.sjoin(background_map_gdf, beams_gdf, how="inner", op='intersects')
        # join_map = gpd.sjoin(beams_gdf, background_map_gdf, how="inner", op='intersects')
        #  associate the geological intersection with the beams
        self.gdf = pd.merge(join_map, beams_gdf, left_on='index_right', right_index=True, suffixes=('', '_ray'))
        # intersect the intersected geological geometry with the beams geometry to save the geologcial information
        self.gdf[self.intersect_geom] = self.gdf.apply(lambda x: FoldDetection.get_intersection(x, self.geom_col, self.min_angle),
                                                        axis=1)

        # groupby the gdf by the n° of the beam
        # self.gdf = self.gdf.reset_index(drop=True)
        result = self.gdf.groupby('index_right').apply(lambda x: self.get_ordonated_intersected_geol(x))
        #Get the angles between line segments and the polygons they intersect.
        angles = result.apply(lambda x: FoldDetection.get_intersection_angles(x,self.geom_col,self.intersect_geom,self.N_points_orientation,self.max_point_dist),
                              axis=1,result_type='expand')
        result[['angle1','angle2']] = angles[[0,1]]
        result[['n1x','n1y','n1z']] = angles[[2,3,4]]
        result[['n2x','n2y','n2z']] = angles[[5,6,7]]
        #Remove the 'geometry' column, since we don't need it anymore.
        result.drop([self.geom_col], axis=1, inplace=True)
        result['rank'] = result[self.intersect_geom].groupby(level=0).cumcount() + 1
        # self.gdf = gpd.GeoDataFrame(result, geometry=self.intersect_geom).reset_index()
        self.gdf = gpd.GeoDataFrame(result, geometry=self.intersect_geom)
        return self.gdf

    @staticmethod
    def dist(segment: LineString):
        return segment.distance(FoldDetection.POINT_ORIGIN)

    @staticmethod
    def get_intersection(row, geom_col,min_angle):
        # if type(row[geom_col]) == MultiPolygon:
        #     result = [poly.intersection(row[geom_col + '_ray']) for poly in row[geom_col].geoms]
        # else:
        #     result = [row[geom_col].intersection(row[geom_col + '_ray'])]
        # results = []
        # for r in result:
        #     if type(r) == MultiLineString:
        #         results = results + [line for line in r.geoms]
        #     else:
        #         results.append(r)
        result = row[geom_col].intersection(row[geom_col + '_ray'])
        if type(result) == MultiLineString:
            results = [line for line in result.geoms]
        else:
            results = [result]
        return results
        
    @staticmethod
    def dist_point_line_seg(x1, y1, x2, y2, x3, y3): # x3,y3 is the point
        """
        Distance between a point (x3,y3) and a line segment defined by its end 
        points: (x1,y1) and (x2,y2). The inputs can be scalar floats or arrays.
        Setting return_square to True means the square of the distance is returned, rather than the distance.
        The code is modified from the one here: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        More details of the theory can be found here: https://paulbourke.net/geometry/pointlineplane/
        I've modified it slightly, in particular to make it work on arrays.
        """
        px = x2-x1
        py = y2-y1
        norm = px*px + py*py
        u =  ((x3 - x1) * px + (y3 - y1) * py) / norm        
        u[u > 1.] = 1.
        u[u < 0.] = 0.
        x = x1 + u * px
        y = y1 + u * py
        dx = x - x3
        dy = y - y3
        dist = (dx*dx + dy*dy)**.5
        return dist
    
    @staticmethod
    def normal_from_plane(x,y,z):
        #Find the normal to the best-fit plane through a set of points.
        vx,vy,vz = [x-x.mean(),y-y.mean(),z-z.mean()]
        sigma = np.array([[np.sum(vx**2),np.sum(vx*vy),np.sum(vx*vz)],
                  [np.sum(vy*vx),np.sum(vy**2),np.sum(vy*vz)],
                  [np.sum(vz*vx),np.sum(vz*vy),np.sum(vz**2)]])
        w,v = np.linalg.eig(sigma)
        n = v[:,np.argmin(w)] #Eigenvector corresponding to smallest eigenvalue is normal to the plane.
        return n
    
    # @staticmethod
    # def bootstrap_normal(x,y,z,Nboot):
    #     #Bootstrap calculation of the normal to a plane.
    #     n = np.zeros(3)
    #     ndata = x.shape[0] #x, y, and z should have the same size.
    #     inds = np.arange(ndata)
    #     for i in range(Nboot):
    #         inds_boot = np.random.choice(inds,ndata,replace=True)
    #         n += FoldDetection.normal_from_plane(x[inds_boot],y[inds_boot],z[inds_boot])
    #     n /= Nboot #This gives us the average of all the vectors.
    #     return n
    
    @staticmethod
    def get_intersection_angles(row,geom_col,intersect_col,N_points,max_point_dist):
        #This works by finding the closest points on the polygon to the intersection point.
        #It is assumed that this point is one of the points bounding the polygon face where the intersection occurs.
        #It is possible to have cases where this is not true, so I might want to use distance from the line semgents instead, 
        #but I don't think it is very likely.
        polygon = row[geom_col]
        intersect_seg = row[intersect_col]
        x,y = intersect_seg.coords.xy #Coordinates of the 2 end points of the intersection line segment.
        x,y = [np.array(x),np.array(y)]
        dx,dy = [x[1]-x[0],y[1]-y[0]]
        amp_d = np.sqrt(dx**2+dy**2)
        theta = [0,0]
        n = [[0]*3]*2 #This will hold the unit normal vectors to the planes.
        z = [0,0]
        dmin = 1e10 #Just make it huge so the first d<dmin will be true.
        for i in range(2):
            if i == 1:
                #The vector between the two points of the intersection line segment points the opposite way.
                dx,dy = [-dx,-dy]
            #Find the closest point to the intersection point on the polygon.
            if polygon.geom_type == 'MultiPolygon':
                polys = [part for part in polygon.geoms]
            else:
                polys = [polygon]
            #A polygon may have multiple parts and interior or exterior rings, 
            #so start by finding which of these lines is the closest to the intersection point, 
            #which should be the one that it intersects.
            for ip,p in enumerate(polys):
                coords_lists = [list(p.exterior.coords)]+[list(ring.coords) for ring in p.interiors] #Lists of coordinates for exterior and each interior ring.
                for ic,coords in enumerate(coords_lists):
                    xp,yp = [np.array([c[0] for c in coords]),np.array([c[1] for c in coords])]
                    d = FoldDetection.dist_point_line_seg(xp[0:-1], yp[0:-1], xp[1:], yp[1:], x[i], y[i]) #Distance of point from each segment, rather than from poitns.
                    # d = np.sqrt((x[i]-xp)**2+(y[i]-yp)**2)
                    if ((ip==0) and (ic==0)) or np.any(d<dmin):
                        ind = np.argmin(d)
                        dmin = d[ind]
                        xclose,yclose = [xp,yp]
                        if polygon.has_z:
                            zclose = np.array([c[2] for c in coords])
            #Interpolate the elevation of the intersection point from the two closest points.
            dpts = np.sqrt((xclose[ind:ind+2]-x[i])**2+(yclose[ind:ind+2]-y[i])**2) #Distance of endpoints of closest segment from the intersection point.
            seg_fraction = dpts[0]/np.sum(dpts) #Fraction of the distance along the intersected segment at which the ray intersects it.
            if polygon.has_z:
                z[i] = zclose[ind]+seg_fraction*(zclose[ind+1]-zclose[ind])
            #Find the distance of each point from the intersection point.
            seg_lengths = np.sqrt(np.diff(xclose)**2+np.diff(yclose)**2)
            dalong = np.concatenate(([0],np.cumsum(seg_lengths)))
            dalong_intersection = dalong[ind]+seg_fraction*seg_lengths[ind]
            dist_intersection = np.abs(dalong-dalong_intersection)
            dclose = dist_intersection
            #Take only the points within a certain distance of the intersection point, if one is specified.
            if not max_point_dist is None:
                if np.min(dclose) <= max_point_dist:
                    close_pts = dclose<=max_point_dist
                    xclose,yclose = [xclose[close_pts],yclose[close_pts]]
                    if polygon.has_z:
                        zclose = zclose[close_pts]
                    dclose = dclose[close_pts]
                else:
                    xclose,yclose = [np.array([xclose[ind]]),np.array([yclose[ind]])]
                    if polygon.has_z:
                        zclose = np.array([zclose[ind]])
                    dclose = np.array(dclose[ind])
            #Take only the N closest points, if a number is specified.
            if not N_points is None:
                inds_sorted = np.argsort(dclose)
                if inds_sorted.shape[0] >= N_points:
                    xclose,yclose = [xclose[inds_sorted[0:N_points]],yclose[inds_sorted[0:N_points]]]
                    if polygon.has_z:
                        zclose = zclose[inds_sorted[0:N_points]]
                    dclose = dclose[inds_sorted[0:N_points]]
                else:
                    #In this case, use the points available (which are less than N_points), but print a warning.
                    if not max_point_dist is None:
                        print('Warning: Fewer than '+str(N_points)+' points within allowed distance.')
                    else:
                        print('Warning: Fewer than '+str(N_points)+' points along contact.')
            if (not max_point_dist is None) and (not N_points is None):
                print('Warning: At least one of N_points and max_point_dist should be specified to get good results.')
            #Find the strike, dip, and angle between the ray and the strike where it intersects the polygon.
            if xclose.size==1:
                #No way to do a 3 point problem here, so just use the orientation of the polygon edge where the line intersects.
                dxp,dyp = [xclose[0]-x[i],yclose[0]-y[i]]
                strike = np.arctan(dyp/dxp)
                n[i] = np.array([np.cos(strike+np.pi/2.0),np.sin(strike+np.pi/2.0),0.0])
                # print(xclose.size)
            else:
                if polygon.has_z and xclose.size>=3:
                    #Do a three (or more) point problem.
                    #This follows the method described by Allmendinger (2020)
                    # vx,vy,vz = [xclose-xclose.mean(),yclose-yclose.mean(),zclose-zclose.mean()]
                    # sigma = np.array([[np.sum(vx**2),np.sum(vx*vy),np.sum(vx*vz)],
                    #           [np.sum(vy*vx),np.sum(vy**2),np.sum(vy*vz)],
                    #           [np.sum(vz*vx),np.sum(vz*vy),np.sum(vz**2)]])
                    #Below is an alternative using a weighted covariance. This doesn't seem to make much difference.
                    # dclose[dclose<1.0] = 1.0 #Set everything to at least 1 meter to avoid huge weights. In particular, avoid getting infinite weights for 0 meters.
                    # w = 1./dclose #weights
                    # w = w/np.sum(w) #Normalize weights so we don't have to divide by sum of weights when taking weighted mean.
                    # vx,vy,vz = [xclose-np.sum(w*xclose),yclose-np.sum(w*yclose),zclose-np.sum(w*zclose)] #Subtract out weighted mean.
                    # sigma = np.array([[np.sum(w*vx**2),np.sum(w*vx*vy),np.sum(w*vx*vz)],
                    #           [np.sum(w*vy*vx),np.sum(w*vy**2),np.sum(w*vy*vz)],
                    #           [np.sum(w*vz*vx),np.sum(w*vz*vy),np.sum(w*vz**2)]]) #Weighted covariance (ignore constant factor in front).
                    # w,v = np.linalg.eig(sigma)
                    # n[i] = v[:,np.argmin(w)] #Eigenvector corresponding to smallest eigenvalue is normal to the plane.
                    # #Try bootstrapping.
                    # Nboot = 10 #In the future, this won't be hard-coded.
                    # n[i] = FoldDetection.bootstrap_normal(xclose,yclose,zclose,Nboot)
                    n[i] = FoldDetection.normal_from_plane(xclose,yclose,zclose)
                    if n[i][2] > 0.0:
                        n[i] = -n[i] #Take the lower hemisphere vector.
                    #The equation of the plane is n[0]*x+n[1]*y+n[2]*z = 0, with the origin in the plane.
                    #The strike is found when z = 0, which gives y = -n[0]*x/n[1].
                    strike = np.arctan(-n[i][0]/n[i][1])
                    #The above equation can't handle a perfectly horizontal plane. That has no real strike anyway, 
                    #but for determining theta, we need something, so we will use the orientation of the map contact 
                    #for it.
                    if np.isnan(strike):
                        if np.all(xclose==xclose[0]): #If all x are identical, it's a vertical line, which linregress can't handle.
                            strike = np.pi/2.0
                        else:
                            result = sp.stats.linregress(xclose,yclose)
                            strike = np.arctan(result.slope)
                else:
                    # print(xclose.size)
                    #Just use the positions of the points in 2D to calculate a strike, assuming constant elevation.
                    #Fit a line to the points.
                    if np.all(xclose==xclose[0]): #If all x are identical, it's a vertical line, which linregress can't handle.
                        strike = np.pi/2.0
                    else:
                        result = sp.stats.linregress(xclose,yclose)
                        strike = np.arctan(result.slope)
                    n[i] = np.array([np.cos(strike+np.pi/2.0),np.sin(strike+np.pi/2.0),0.0])
            dxp,dyp = [np.cos(strike),np.sin(strike)]
            amp_dp = np.sqrt(dxp**2+dyp**2)
            theta[i] = np.arccos(np.dot([dx,dy],[dxp,dyp])/[amp_d*amp_dp])[0] #Use the dot product to find the angle between vectors.
            theta[i] = theta[i]*180.0/np.pi
            if theta[i] > 90.0:
                theta[i] = 180.0-theta[i] #Always use the acute angle.
        result = theta + list(n[0]) + list(n[1])
        return result
        

    def remove_duplicate_column_and_organize_geom(self, concat_df):

        concat_df = concat_df.loc[:, ~concat_df.columns.duplicated()]
        geom_to_concat_at_the_end = concat_df[self.intersect_geom]
        concat_df = concat_df.drop([self.intersect_geom], axis=1)
        concat_df = pd.concat([concat_df, geom_to_concat_at_the_end], axis=1)
        return concat_df

    def merge_col_with_tmp(self, row, tmp):
        
        dic_to_fill = {self.id_fm: pd.merge(row[self.id_fm], tmp, left_index=True, right_index=True)}
        concat_df = pd.DataFrame()
        for col in row.columns.values:
            if col != self.intersect_geom:
                dic_to_fill[col] = pd.merge(row[col], tmp, left_index=True, right_index=True)
                concat_df = pd.concat([concat_df, dic_to_fill[col]], axis=1)

        concat_df = self.remove_duplicate_column_and_organize_geom(concat_df)
        concat_df['distance_origin'] = concat_df[self.intersect_geom].apply(FoldDetection.dist)
        concat_df.sort_values(by='distance_origin', ascending=True, inplace=True)
        concat_df.reset_index(drop=True,inplace=True) #Renumber sequentially, rather than by polygon, to avoid repeated values.

        return concat_df

    @staticmethod
    def drop_layer_path_synth(group):
        """
        corrige les bugs de construction des shapefiles. En effet, certaines fois les polygones des mêmes unités ont
        été créé avec un nom de layer différent. Cette fonction corrige ce problème
        :param group:  beams
        :return: un rayon de segments sans doublons.
        """""
        if len(group) > 1:
            if "layer" and "path" in group.columns:
                subgroup = group[~group.duplicated(subset=['layer'], keep='first')]
                subgroup = subgroup.drop_duplicates(subset=['distance_origin'], keep='first')
                group = group[group['layer'].isin(subgroup['layer'].tolist())]
        return group

    def get_ordonated_intersected_geol(self, row):
        """
        Réalise le merge des listes de LineString contenues dans la colonne 'intersection' et puis les ordonne suivant la distance à l'origine.
        :param group: le sous-ensemble des géologies qui intersectent un segment particulier
        :return:
        """

        if self.intersect_geom in row:
            tmp = row[self.intersect_geom].explode()
        else:
            tmp = row[self.geom_col].explode()
        # tmp.dropna(inplace=True) #When the list of intersections was empty, the tmp entry will be nan.
        concat_df = self.merge_col_with_tmp(row, tmp)

        concat_df = FoldDetection.drop_layer_path_synth(concat_df)
        for col in concat_df.columns.values:
            # if '_x' in col or '_y' in col or 'geometry' in col or 'index' in col or 'FID' in col or 'path' in col:
            if '_x' in col or '_y' in col or 'ray' in col or 'index' in col or 'FID' in col or 'path' in col:
                concat_df = concat_df.drop([col], axis=1)

        return concat_df

    @staticmethod
    def order_segment_by_rank(result):
        """
        Réalise le merge des listes de LineString contenues dans la colonne 'intersection' et puis les ordonne suivant la
        distance à l'origine. :param group: le sous-ensemble des géologies qui intersectent un segment particulier
        :return: sorted dataframe by distance
        """
        result.sort_values(by='rank', ascending=True, inplace=True)
        return result

    @staticmethod
    def create_df_with_new_ids(group, ids, dico, dico_alpha):

        id_group = group[ids]
        num_ids = dico[id_group.iloc[0]]
        alpha_conv = dico_alpha[num_ids]

        group['alpha_ids'] = alpha_conv
        return group

    def attribute_new_ids_to_formation(self, id_list, dict_alpha_conv):
        unique_df_ids = pd.unique(self.gdf[self.id_fm])
        dict_form_ids = {}
        for iter, unique in enumerate(unique_df_ids):
            dict_form_ids[unique] = id_list[iter]

        self.gdf = self.gdf.groupby(self.id_fm,group_keys=False).apply(
            lambda x: FoldDetection.create_df_with_new_ids(x, self.id_fm, dict_form_ids, dict_alpha_conv))
        return self.gdf

    def create_id(self):
        """
        extract all formation names in RGF_NAME and create a three digit index associated to each formation.
        Formation names and ids are merged, to have a common index with the general gdf "result" for the final merge.
        Final merge made with the common column RGF_NAME
        :param result: beams df with geometries
        :return: gdf with merged alphaID associated to RGF_NAME
        """
        count = 0
        id_list = []

        for it in range(len(pd.unique(self.gdf[self.id_fm]))):
            id_list.append(count)
            count += 1
        dict_alpha_conv = FoldDetection.digits_to_alpha(id_list)

        self.gdf = self.attribute_new_ids_to_formation(id_list, dict_alpha_conv)
        return self.gdf  # beams gdf with a new column "ID" which contains three digits numerical ids

    @staticmethod
    def digits_to_alpha(digits_list):
        """
        transform unique numerical identifiers into an alphabetic character.
        :param digits_list:
        :return: str chain of unique character IDs
        """
        #Below is a string of 86 unique characters.
        #If you need more than that (due to a map with more than 86 units), you will need to expand this string.
        characters = 'abcdefghijklmnopqrstuvwxyz' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + '0123456789' + 'αβγδεζηθικλμνξοπρστυφχψω'
        if max(digits_list) > len(characters):
            print('Error: Number of units ('+str(max(digits_list))+') exceeds number of unique character IDs.')
            print('To solve this, add more unique characters to the characters string in the the digits_to_alpha function.')
        dict_alpha_conv = {}
        for n in digits_list:
            dict_alpha_conv[n] = characters[n]

        return dict_alpha_conv

    def analyze_symmetry(self, gdf, geol_map):
        """
        analyse les symmétries par rayons dans les segments. Efface les faux négatifs,
        :param self: gdf with beams data
        :return: gdf of analyzed match that represents fold, ordered by match_id
        """
        # extract the complete liste of segment succession in a chain string
        # gdf = FoldDetection.RemoveQuaternaryAndDuplicates(gdf)
        gdf2 = gdf.groupby('index_right',group_keys=False).apply(lambda x: FoldDetection.RemoveQuaternaryAndDuplicates(x,self.id_fm))
        final_ids_str_list = FoldDetection.list_to_final_str(gdf2)
        line_df = pd.DataFrame()
        point_df = pd.DataFrame()
        line_list = []
        point_list = []
        print('checkPoint before big LOOP')
        num_id = 0  # id of the individual matches
        check_num_id = []  # list to check if a match was already given to a previous match
        for index, it in enumerate(pd.unique(gdf2.index.get_level_values(0))):
            # iteration sur la chaîne de charactère principale de beam et check des symmétries avec expr. régulières
            match_pal_list = FoldDetection.iteration_pal(final_ids_str_list[index],self.N_to_match)
            select = (gdf2.loc[gdf2.index.get_level_values('index_right') == it])
            #  remove false positive
            selected_id, num_id, check_num_id = FoldDetection.trie_match(select, match_pal_list, num_id, check_num_id, self.N_to_match)
            FoldDetection.check_pal(selected_id, match_pal_list, self.id_fm)  # check the validity of the match
            #Find the center point for each fold.
            if selected_id.size>0:
                for i in selected_id['match_id'].unique():
                    mask = selected_id['match_id'] == i
                    indices = selected_id[mask].index
                    i1 = indices[0]
                    i2 = indices[-1]
                    #Find the contact location of the first two units at the beginning (start point) and end (end point) of the match.
                    seg1 = list(selected_id.geometry[i1].coords)
                    seg2 = list(selected_id.geometry[indices[1]].coords)
                    if (np.round(seg1[1]) == np.round(seg2[0])).all():
                        start_pt = seg1[1]
                        end_pt = list(selected_id.geometry[i2].coords)[0]
                        start_angle = selected_id.angle2[i1]
                        end_angle = selected_id.angle1[i2]
                        start_normal = [selected_id.n2x[i1],selected_id.n2y[i1],selected_id.n2z[i1]]
                        end_normal = [selected_id.n1x[i2],selected_id.n1y[i2],selected_id.n1z[i2]]
                    elif (np.round(seg1[0]) == np.round(seg2[1])).all():
                        start_pt = seg1[0]
                        end_pt = list(selected_id.geometry[i2].coords)[1]
                        start_angle = selected_id.angle1[i1]
                        end_angle = selected_id.angle2[i2]
                        start_normal = [selected_id.n1x[i1],selected_id.n1y[i1],selected_id.n1z[i1]]
                        end_normal = [selected_id.n2x[i2],selected_id.n2y[i2],selected_id.n2z[i2]]
                    else:
                        print('Warning: Segment end points do not match.')
                        start_pt = seg1[0] #Since we have to do something.
                        end_pt = list(selected_id.geometry[indices[-1]].coords)[1]
                        start_angle = selected_id.angle1[i1]
                        end_angle = selected_id.angle2[i2]
                        start_normal = [selected_id.n1x[i1],selected_id.n1y[i1],selected_id.n1z[i1]]
                        end_normal = [selected_id.n2x[i2],selected_id.n2y[i2],selected_id.n2z[i2]]
                    midpt = ((start_pt[0]+end_pt[0])/2.0,(start_pt[1]+end_pt[1])/2.0)
                    #For now, use the midpt as the axis_pt. In the future, I will change this.
                    axis_pt = midpt
                    #Calculate the fold axis orientation.
                    axis_orientation = np.cross(start_normal,end_normal)
                    amplitude = np.linalg.norm(axis_orientation)
                    # axis_orientation /= amplitude #Normalize
                    if amplitude != 0.0: #If it does, normalizing will produce nan
                        axis_orientation /= amplitude #Normalize
                    if axis_orientation[2] > 0.0:
                        axis_orientation *= -1.0 #Use the lower hemisphere version.
                    #Determine whether the two normals point towards or away from each other.
                    # theta = np.arccos(np.dot(start_normal[0:2],end_normal[0:2])/(
                        # np.linalg.norm(start_normal[0:2])*np.linalg.norm(end_normal[0:2]))) #Angle between horizontal components of normals.
                    line_vector = [end_pt[0]-start_pt[0],end_pt[1]-start_pt[1]] #2D vector from start_pt to end_pt.
                    theta1 = np.arccos(np.dot(start_normal[0:2],line_vector)/(
                        np.linalg.norm(start_normal[0:2])*np.linalg.norm(line_vector))) #Angle between horizontal component of normal and line.
                    theta2 = np.arccos(np.dot(end_normal[0:2],line_vector)/(
                        np.linalg.norm(end_normal[0:2])*np.linalg.norm(line_vector))) #Angle between horizontal component of normal and line.
                    anticline_dips = (theta1 < np.pi/2.) & (theta2 > np.pi/2.) #Downward normals point together, so limbs dip apart, as in an anticline.
                    syncline_dips = (theta1 > np.pi/2.) & (theta2 < np.pi/2.) #Downward normal point apart, so limbs dip together, as in a syncline.
                    #Create the points and lines.                 
                    line = selected_id.iloc[indices[0]] #For the line, use the geological information from the first segment.
                    point = selected_id.iloc[indices[0]]
                    line['geometry'] = LineString([start_pt,end_pt])
                    # point['geometry'] = Point(midpt)
                    point['geometry'] = Point(axis_pt)
                    line['length'] = np.sqrt((start_pt[0]-end_pt[0])**2+(start_pt[1]-end_pt[1])**2)
                    strat_num = selected_id[self.id_fm][indices] #Younger units should have lower numbers.
                    if (strat_num[indices[1]] > strat_num[indices[0]]): #Gets older toward the center.
                    # if (strat_num[indices[1]] > strat_num[indices[0]]) and anticline_dips: #Gets older toward the center.
                        point['fold_type'] = 'anticline'
                        # if (strat_num[indices[1]] >= 6):
                        #     n6 += 1
                        #     print(n6)
                    elif (strat_num[indices[1]] < strat_num[indices[0]]): #Gets younger toward the center.
                    # elif (strat_num[indices[1]] < strat_num[indices[0]]) and syncline_dips: #Gets younger toward the center.
                        point['fold_type'] = 'syncline'
                    else:
                        point['fold_type'] = 'unknown'
                        # if (strat_num[indices[1]] >= 6):
                        #     n6 += 1
                        #     print(n6)
                    if anticline_dips:
                        point['DipDirs'] = 'apart'
                    elif syncline_dips:
                        point['DipDirs'] = 'together'
                    else:
                        point['DipDirs'] = 'same'
                    if (strat_num[indices[1]]-strat_num[indices[0]] == 1) or (strat_num[indices[0]]-strat_num[indices[1]] == 1):
                        point['conformable'] = True
                        #I also might want to try allowing some age gap, so ages don't have to be perfectly conformable.
                    else:
                        point['conformable'] = False
                    point['max_angle'] = max(start_angle,end_angle) #The largest (closer to a right angle) of the angles the ends of the line make with the contact.
                    # point['ax'] = axis_orientation[0] #Include the axis orientation.
                    # point['ay'] = axis_orientation[1]
                    # point['az'] = axis_orientation[2]
                    #Replace the n1 and n2 normals, which come from a single segment, with the start_normal and end_normal instead.
                    point['n1x'], point['n1y'], point['n1z'] = start_normal[0],start_normal[1],start_normal[2]
                    point['n2x'], point['n2y'], point['n2z'] = end_normal[0],end_normal[1],end_normal[2]
                    line_list.append(line)
                    point_list.append(point)
        line_df = pd.DataFrame(line_list)
        point_df = pd.DataFrame(point_list)
        print('CheckPoint after the big LOOP')
        return line_df,point_df

    @staticmethod
    def RemoveQuaternaryAndDuplicates(gdf_part,id_fm):
        """
        Remove Quaternary (or otherwise unwanted) units, which are marked with a 
        Strat_Order value of 0.
        Combine any duplicate units within a beam.
        """

        n = gdf_part.index.get_level_values('index_right')[0] #Since we grouped by this, all values should be the same.
        quat = gdf_part[id_fm].values == 0
        alpha_ids = gdf_part['alpha_ids'].values
        duplicates = alpha_ids[1:] == alpha_ids[0:-1]
        if any(quat) or any(duplicates):
            #Get some more information about the segments on this beam.            
            inds = np.asarray(gdf_part.index.get_level_values('level_1'))
            nrows = inds.size
            start_coords = np.array([list(gdf_part.geometry.values[i].coords)[0] for i in range(nrows)])
            end_coords = np.array([list(gdf_part.geometry.values[i].coords)[1] for i in range(nrows)])
            diff1 = np.round(start_coords[1:]-end_coords[0:-1])
            diff2 = np.round(start_coords[0:-1]-end_coords[1:])
            if  (diff1 == 0).all():
                order = 1 #start_coords of one segment equal end_coords of previous.
            elif (diff2 == 0).all():
                order = 2 #end_coords of one segment equal start_coords of previous.
            else:
                print('Warning: Not all segment end points match.')
                RMS1 = np.sqrt(np.mean(diff1**2))
                RMS2 = np.sqrt(np.mean(diff2**2))
                if RMS1 < RMS2:
                    order = 1
                else:
                    order = 2
            #Remove the Quaternary units and extend the remaining segments so that they meet up.
            Quat_contact_next = np.zeros(nrows,dtype=bool) #Contact after this segment is where A Quaternary unit was removed.
            i = 0
            while i < nrows: #Do this rather than a for loop, so that we can remove entries without messing up what i refers to.
                if quat[i]:
                    row_ind = (n,inds[i])
                    if nrows > 1: #If this is the only unit and we're removing it, then there's nothing left to rearrange.
                        prev_row_ind = (n,inds[i-1])
                        if i == 0: #First segment. Extend the next one to where this one started.
                            if order == 1:
                                start_coords[i+1,:] = start_coords[i,:]
                            else:
                                end_coords[i+1,:] = start_coords[i,:]
                        elif i == nrows-1: #End segment. Extend the previous one to where this one ends.
                            if order == 1:
                                end_coords[i-1,:] = end_coords[i,:]
                            else:
                                start_coords[i-1,:] = start_coords[i,:]
                        else: #Middle segment. Take the midpoint between the segments on either side.
                            midpt = (start_coords[i,:]+end_coords[i,:])/2.0
                            if order == 1:
                                start_coords[i+1,:] = midpt
                                end_coords[i-1,:] = midpt
                            else:
                                midpt = (start_coords[i,:]+end_coords[i,:])/2.0
                                start_coords[i-1,:] = midpt
                                end_coords[i+1,:] = midpt
                        #Label the previous unit as being before a Quaternary unit.
                        if i > 0:
                            Quat_contact_next[i-1] = True
                    #Remove this point from everything.
                    gdf_part.drop(row_ind,inplace = True)
                    start_coords = np.delete(start_coords, i, axis=0)
                    end_coords = np.delete(end_coords, i, axis=0)
                    alpha_ids = np.delete(alpha_ids, i)
                    inds = np.delete(inds,i)
                    quat = np.delete(quat, i)
                    Quat_contact_next = np.delete(Quat_contact_next, i)
                    nrows -= 1
                else:
                    i += 1
            #Where a Quaternary unit was removed, the bedding orientation left at the new contact comes from the contacts with the Quaternary unit, which is meaningless.
            #Instead, we will calculate the bedding orientations by averaging those of the contacts on either side of this one.
            #This isn't perfect, but if the contacts are part of a conformable succession, it is likely to be a decent estimate, and I don't have a better idea.
            #This will still give bad results if a single unit is between Quaternary units on both sides.
            #However, it will work in most cases and is probably good enough.
            #Also, it doesn't account for the case in which the Quaternary unit that was removed was the first or last one.
            #However, in that case, the "contact" will be with the edge of the map and is largely meaningless anyway.
            terms1 = ['angle1','n1x','n1y','n1z']
            terms2 = ['angle2','n2x','n2y','n2z']
            for i in range(nrows-1):
                if Quat_contact_next[i]:
                    row_ind = (n,inds[i])
                    next_row_ind = (n,inds[i+1])
                    if order == 1:
                        for j in range(len(terms1)):
                            term = (gdf_part[terms1[j]][row_ind] + gdf_part[terms2[j]][next_row_ind])/2
                            gdf_part[terms2[j]][row_ind] = term
                            gdf_part[terms1[j]][next_row_ind] = term
                    else:
                        for j in range(len(terms1)):
                            term = (gdf_part[terms2[j]][row_ind] + gdf_part[terms1[j]][next_row_ind])
                            gdf_part[terms1[j]][row_ind] = term
                            gdf_part[terms2[j]][next_row_ind] = term
            #Remove the duplicates and combine adjacent segments.
            if inds.size > 0:
                duplicates = np.concatenate(([False],alpha_ids[1:] == alpha_ids[0:-1])) #Recalculate in case quaternary removal added any.
                i = 0
                while i < nrows:
                    if duplicates[i]:
                        row_ind = (n,inds[i])
                        if nrows > 1: #If this is the only unit and we're removing it, then there's nothing left to rearrange.
                            prev_row_ind = (n,inds[i-1]) #i=0 will never be a duplicate, so no risk of i-1 being negative.
                            #Extend previous segment, which this is a duplicate of, up to the end of this segment.
                            if order == 1:
                                end_coords[i-1,:] = end_coords[i,:]
                            else:
                                start_coords[i-1,:] = start_coords[i,:]
                            #For items that are recorded at both ends of a segment, I need to choose the ones from the far ends of the combined segments.
                            if order == 1:
                                gdf_part['angle2'][prev_row_ind] = gdf_part['angle2'][row_ind]
                                gdf_part['n2x'][prev_row_ind] = gdf_part['n2x'][row_ind]
                                gdf_part['n2y'][prev_row_ind] = gdf_part['n2y'][row_ind]
                                gdf_part['n2z'][prev_row_ind] = gdf_part['n2z'][row_ind]
                            else:
                                gdf_part['angle1'][prev_row_ind] = gdf_part['angle1'][row_ind]
                                gdf_part['n1x'][prev_row_ind] = gdf_part['n1x'][row_ind]
                                gdf_part['n1y'][prev_row_ind] = gdf_part['n1y'][row_ind]
                                gdf_part['n1z'][prev_row_ind] = gdf_part['n1z'][row_ind]
                            
                        #Remove this point from everything.
                        gdf_part.drop(row_ind,inplace = True)
                        start_coords = np.delete(start_coords, i, axis=0)
                        end_coords = np.delete(end_coords, i, axis=0)
                        alpha_ids = np.delete(alpha_ids, i)
                        inds = np.delete(inds,i)
                        Quat_contact_next = np.delete(Quat_contact_next, i)
                        nrows -= 1
                        
                        #Recalculate duplicates with the remaining points.
                        duplicates = np.concatenate(([False],alpha_ids[1:] == alpha_ids[0:-1]))
                    else:
                        i += 1
            #Replace geometries of kept segments with their updated geometries.
            for i,ind in enumerate(inds):
                row_ind = (n,ind)
                gdf_part.geometry[row_ind] = LineString([Point(start_coords[i,:]),Point(end_coords[i,:])])
        
        return gdf_part

    @staticmethod
    def list_to_final_str(beams_gpd):
        index_0 = pd.unique(beams_gpd.index.get_level_values(0))
        list_id = []
        for it in index_0:
            test = (beams_gpd['alpha_ids'].iloc[beams_gpd.index.get_level_values('index_right') == it])
            test = test.tolist()
            test = ''.join(test)
            test = str(test)
            list_id.append(test)

        return list_id

    @staticmethod
    def iteration_pal(pal_string,N_to_match: int):
        """
        detect les palyndromes sur une chaîne de caractère à partir d'une expression régulière "expr"

        :param pal_string:
        :return: dictionnaire d'objets regex représentant les match regex(span,match)
        """
        #Note on regular expression character matching: Previous, we had used three character codes which were matched with \w{3}.
        #Now, I'm using single character codes. These can be matched with \w, assuming all possible character are word characters, 
        #or with . for an even larger range of possible characters.
        pal_string = str(pal_string)
        pal_string_copie = str(pal_string)
        # chaîne principale
        if N_to_match == 1:
            expr = r"(\w)(\w)+?\1"
            # expr = r"(.).+?\1"
            #This expr looks for a character, 1 or more intervening characters, and then the first character again.
        elif N_to_match == 2:
            expr = r"(\w)(\w)(\w)*?\2\1"
            # expr = r"(.)(.).*?\2\1"
            #This expr looks for 2 characters, with 0 or more intervening characters, and then the first 2 characters in reverse order.
            expr2 = r"(\w)(\w)\1"
            # expr2 = r"(.).\1"
            #expr2 looks for a character, exactly 1 intervening character, and then the first character again.
        else:
            print('N_to_match > 2 is not allowed.')
        
        match_list = {}
        indice = 0  # indice correspond à l'index du match dans la chaîne principale
        if N_to_match == 2:
            #In this case, also consider a match where the same contact is repeated, but there's only one middle unit.
            #We need to consider this case first because for a given index, it will be the shorter solution and, therefore, the preferred one.
            while True:
                result = reg.search(expr2, pal_string)
                if result is not None:
                    span = result.span()
                    indice = indice + span[0]  # span[0] c'est l'index début du match, spam[1] la fin
                    match_list[indice] = result
                    indice = indice + 1
                    pal_string = pal_string[span[0] + 1:]
                else:
                    break
            pal_string = pal_string_copie #Reset it for the next search.
            indice = 0 #Reset this as well.
        while True:
            result = reg.search(expr, pal_string)
            if result is not None:
                span = result.span()
                indice = indice + span[0]  # span[0] c'est l'index début du match, spam[1] la fin
                if indice not in match_list.keys():
                    #The not in keys part only matters for N_to_match == 2, where we may have already found an expr2 match here.
                    match_list[indice] = result
                indice = indice + 1
                # on pousse le curseur de trois vers la gauche dans la chaîne principale,
                # c.a.d à la prochaine formation du rayon
                pal_string = pal_string[span[0] + 1:]
            else:
                break

        return match_list

    @staticmethod
    def trie_match(beam_info, match_list, match_id: int, check_num, N_to_match: int):
        """
        associe les match beam par beam en associant la position des matchs
        dans les chaines de caractères avec la position les beams
        1) associe le match repéré à la géométrie concernée
        2) créer un match id unique dépendant de l'index du beam étudié
        3) remove false positive match:
            - les matchs qui commencent et finissent par une formation agée de moins de 3 Ma
            - les matchs qui ont une longueur plus petite que trois (ça ne devrait jamais arrivé)
            - les matchs composés de trois fois le même terme e.g.(aie, aie, aie)
            - les matchs dont tout sauf les extrêmités sont composés de quaternaire
        :param check_num:
        :param match_id:
        :param beam_info: is the analyzed beam
        :param match_list: the associate list of match to beam_info in dic object format
        :return: a cleaned df withouth false positive
        """
        iteration = 0
        if 'distance_o' in beam_info.columns:
            beam_info.sort_values(by='distance_o', ascending=True, inplace=True)
        else:
            beam_info.sort_values(by='distance_origin', ascending=True, inplace=True)
        #Reindex beam_info so it doesn't have repeated values.
        beam_info.reset_index(drop=True,inplace=True)
        # new_selection_df created parce que des fois new_selection_df
        # n'est pas définis parce que la liste des matchs est vide
        new_selection_df = pd.DataFrame()
        # clean_match_df  = pd.DataFrame()
        clean_match_geo_list = []
        ###########################
        #  (step 1): Associe le match repéré à la géométrie associée
        ###########################

        for index1, value1 in match_list.items():
            iteration = iteration + 1
            index2 = value1.span()
            span1 = index2[1]
            span0 = index2[0]

            pos1 = int(index1)  # index de début du match
            longueur = int(span1 - span0)
            selection = beam_info.iloc[pos1:pos1 + longueur, :]  # select the match in the gdf with geometry
            selection = pd.DataFrame(selection)

            selection['match_id'] = match_id
            new_selection_df = pd.concat([new_selection_df, selection], sort=False)
            check_num = FoldDetection.track_similar_ids(check_num, match_id)
            ###########################
            #  (step 3): remove false positive match
            ###########################
            # remove match that len < 3 (three not the heart emoticon)
            # remove match with three times the same formation (quat, quat, quat)
            new_selection_df = FoldDetection.remove_smaller_len_than_three(selection, new_selection_df, match_id)
            # # remove selection with first or last entries ages smaller than 3 Ma.
            # new_selection_df = FoldDetection.remove_less_than_three(selection, new_selection_df, match_id,N_to_match)
            # # Enlève les matchs dont tout sauf les extrêmités sont composés de quaternaire
            # new_selection_df = FoldDetection.remove_quat_heart(selection, new_selection_df, match_id)
            if new_selection_df is None:
                # it will probably never happen
                print('Il n y a pas de match dans ce rayon, ce qui est problématique')
            # unique_match_id = pd.unique(new_selection_df['match_id'])
            # if match_id in unique_match_id:
            clean_match_geo_list.extend(
                (new_selection_df[new_selection_df['match_id'] == match_id]).values.tolist())
            match_id = match_id + 1
        columns = new_selection_df.columns.values
        clean_match_df = pd.DataFrame(data=[line for line in clean_match_geo_list], columns=columns)

        return clean_match_df, match_id, check_num  # warning here it returns a df not a gdf

    @staticmethod
    def track_similar_ids(list_ids, num_id: int):
        if num_id in list_ids:
            # print(num_id + 'were given 2 times')
            sys.exit('two similar ids were given to different matches in "tri_match() at id #' + str(num_id))
        list_ids.append(num_id)
        return list_ids

    @staticmethod
    def remove_smaller_len_than_three(selection, concat_df, num_id):
        """
        Drop match (selection) if there is only one component (three times the same Fm)
        Drop match (selection) if the len of the match is smaller than three (this case should never happend)
        :param selection: the selected regex match in df format
        :param concat_df: the future final result gdf
        :param num_id: the created match_id that will be assiciated with the segment
        :return: cleaned gdf
        """
        unique_match_id = pd.unique(concat_df.match_id)
        unique_id = pd.unique(selection['alpha_ids'])

        #  Drop match (selection) if there is only one component (three times the same Fm)
        if len(unique_id) == 1:
            if num_id in unique_match_id:  # avoid error if somehow the match is already errase
                concat_df = concat_df.drop(concat_df[concat_df['match_id'] == num_id].index)
                print('remove match with only one component')
            return concat_df

        # remove match if the selection is smallest than 3 (the minimum)
        if len(selection) < 3:
            if num_id in unique_match_id:  # avoid error if somehow the match is already errase
                concat_df = concat_df.drop(concat_df[concat_df['match_id'] == num_id].index)
                print('remove match with a len < 3')
            return concat_df

        return concat_df

    @staticmethod
    def coords(geom, geom_col):
        x = geom[geom_col].apply(lambda x: list(x.coords))
        return x

    @staticmethod
    def match_par_liste_df(group, geom_col):
        """
        group all the geometry of one segment (match), in one geometry
        i.e. the beggining of the first and the end of the last terms
        then associate this geometry with selected variable that define this formatin
        :param groupe_id: individual match (segment) grouped by df
        :return: df with one geometry per match associated with selected attributes.
        """
        ###########################
        #  extract the targeted formation (first and last one)
        ###########################
        # if float(first_elem.AGE_DEB)!= float(last_elem.AGE_DEB):
        #     sys.exit('MATCH invalide, first elem != last elem. \n first_elem: ' + float(first_elem.AGE_DEB) + ' last_elem: ' + float(last_elem.AGE_DEB))

        first_pos = group.head(1)
        last_pos = group.tail(1)

        if first_pos.alpha_ids.to_list() != last_pos.alpha_ids.to_list():
            sys.exit(
                'incorrect match:' + str(first_pos['alpha_ids'].values) + ", last_pos: " + str(
                    last_pos['alpha_ids'].values))

        first_pos_df = pd.DataFrame(first_pos)
        last_pos_df = pd.DataFrame(last_pos)

        list_Line = []
        for geom1, geom2 in zip(first_pos_df[geom_col], last_pos_df[geom_col]):
            list_Line.append(geom1)
            list_Line.append(geom2)
        multi_line = MultiLineString(list_Line)
        one_line_geom = first_pos_df
        one_line_geom.drop(columns=geom_col, inplace=True)
        one_line_geom[geom_col] = str(multi_line)
        one_line_geom[geom_col] = one_line_geom[geom_col].apply(wkt.loads)

        gdf = gpd.GeoDataFrame(one_line_geom, geometry=geom_col)

        return one_line_geom

    @staticmethod
    def getIndex_sec(group):
        first_df = pd.DataFrame(group.head(1))
        last_df = pd.DataFrame(group.tail(1))
        first_df['match_extremity'] = 0
        last_df['match_extremity'] = 1
        concat_df = pd.concat([first_df, last_df])
        return concat_df

    @staticmethod
    def unary(group):
        group['geometry'] = group['geometry'].unary_union
        return group

    @staticmethod
    def check_pal(gdf_matched, dict_list_match, id_fm):
        if not gdf_matched.empty:
            id_match = pd.unique(gdf_matched['match_id'])
            for it in id_match:
                selection_df = gdf_matched[gdf_matched['match_id'] == it]
                first = selection_df.head(1)
                last = selection_df.tail(1)
                if first[id_fm].to_numpy() != last[id_fm].to_numpy():
                    sys.exit('coucou les matchs ne matches pas')
        return None

    def associate_cluster_geom(self):
        """
        associate the geometry with their cluster labels. and return a geodataframe of it
        :param union_cluster: series result of the previous tri match function
        :param origina_cluster_gdf: original dataframe loaded for the section "clustering
        :return: geodataframe with polygon geom + name cluster
        """
        
        clusters = self.gdf['cluster'].unique()
        clusters = clusters[clusters != -1]
        hulls = []
        fold_types = []
        for c in clusters:
            cluster_df = self.gdf[self.gdf['cluster'] == c]
            x = np.ravel([cluster_df[self.geom_col][i].xy[0] for i in cluster_df.index])
            y = np.ravel([cluster_df[self.geom_col][i].xy[1] for i in cluster_df.index])
            points = np.array([x,y]).T
            hull = ConvexHull(points)
            vertx = points[hull.vertices,0]
            verty = points[hull.vertices,1]
            coords = [(vertx[i],verty[i]) for i in range(vertx.size)]+[(vertx[0],verty[0])] #Duplicate last point to close the polygon.
            hulls.append(Polygon(coords))
            fold_types.append(cluster_df['fold_type'][cluster_df.index[0]]) #This should be the same for all members of the cluster, so just use the first one.
        d = {'cluster':clusters,'fold_type':fold_types,'geometry':hulls}
        hull_df = gpd.GeoDataFrame(d,geometry='geometry',crs=self.crs)
        return hull_df
