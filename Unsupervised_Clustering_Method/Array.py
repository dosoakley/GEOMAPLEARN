"""
Working version
readme:
result: metadata coming from the .shp file transfered to the beams
strati: metadata coming from the .shp file to the beams used for the color representation. Strati est une copie de result
join_map: original variable that includes the .shp file


"""
import geopandas as gp
from shapely.geometry import Point

POINT_ORIGIN = Point(0, 0)


def get_data_from_vector_map(vector_file_path):
    geodata = gp.GeoDataFrame.from_file(vector_file_path, geom_col='geometry', crs='epsg:2154',engine='pyogrio')

    return geodata
