import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

class CircleGrid:
    """
    This is based on the Circle class from Thierry's code, but instead of just making beams 
    as diameters of the circle, it makes beams as chords between all points on 
    the circle.
    The number_samples is the number of points in a quarter circle. So it is analagous to 
    number_samples being the number of points on one side of a rectangle in some 
    of the old functions, but it means that the total number of points around 
    the circle will be 4 times number_samples.
    """

    def __init__(self, number_samples: int = 16, geom_col: str = 'geometry', crs: str = 'epsg:2154'):
        self.number_samples = number_samples
        self.geom_col = geom_col
        self.crs = crs
        self.ymax = None
        self.ymin = None
        self.xmax = None
        self.xmin = None
        self.gdf = None

    @staticmethod
    def get_bbox_rectangle(xmin, ymin, xmax, ymax):
        bbox = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
        return bbox

    @staticmethod
    def get_mass_center(bbox):
        center_point = bbox.centroid
        return center_point

    @staticmethod
    def get_middle_bbox(min, max):
        mid = (min + max) / 2
        return mid

    @staticmethod
    def createPoint(x, y):
        return Point(x, y)

    def get_radius(self):
        mid_y = CircleGrid.get_middle_bbox(self.ymin, self.ymax)
        mid_x = CircleGrid.get_middle_bbox(self.xmin, self.xmax)
        midpt = CircleGrid.createPoint(mid_x, mid_y)
        radii = []
        for x in [self.xmin,self.xmax]:
            for y in [self.ymin,self.ymax]:
                Point = CircleGrid.createPoint(x,y)
                radii.append(Point.distance(midpt))
        radius = max(radii)
        radius = radius*1.1 #This avoids getting vertices exactly at the corners.
        return radius

    @staticmethod
    def create_line(X, Y):
        ligne = LineString([Point(X[0], Y[0]), Point(X[1], Y[1])])
        return ligne

    @staticmethod
    def circleBeams(x, y):
        dico_pair = []
        xall = np.zeros((2,np.sum(np.arange(0,len(x)))))
        yall = np.zeros((2,np.sum(np.arange(0,len(x)))))
        n = 0
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                x1 = np.array([x[i], x[j]])
                y1 = np.array([y[i], y[j]])
                ligne = CircleGrid.create_line(x1, y1)
                dico_pair.append(ligne)
                xall[:,n] = x1
                yall[:,n] = y1
                n = n+1
        return dico_pair

    # def get_beams_in_circle(self, circle):
    def get_beams_in_circle(self, center,radius,theta):
        # x, y = circle.exterior.coords.xy
        # middle = len(x) / 2 - 1
        # middle = (int(np.round(middle, 0)))
        x = center[0]+radius*np.cos(theta)
        y = center[1]+radius*np.sin(theta)
        dico_pair = CircleGrid.circleBeams(x, y)
        df = pd.DataFrame([line for line in dico_pair],columns=[self.geom_col])
        gdf = gpd.GeoDataFrame(df, geometry=self.geom_col, crs=self.crs)
        return gdf

    def get_arranged_rays(self, working_map):
        """
         Fonction qui balaie la zone d'étude de rayons sous tout ces angles, à égale distance
        :param working_map: shp data
        :return: geodataframe de rayons
        """
        self.xmin, self.ymin, self.xmax, self.ymax = working_map.total_bounds

        # define bounding box of the map
        bbox = CircleGrid.get_bbox_rectangle(self.xmin, self.ymin, self.xmax, self.ymax)
        center_point = CircleGrid.get_mass_center(bbox)
        radius = self.get_radius()
        theta_step = np.pi/2.0/self.number_samples
        theta = np.arange(0,2*np.pi,theta_step)
        xc,yc = center_point.xy
        center = [xc[0],yc[0]]
        gdf = self.get_beams_in_circle(center, radius, theta)
        return gdf