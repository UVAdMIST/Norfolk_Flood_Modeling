# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:38:42 2018

@author: Faria
"""


import shapefile
import pandas as pd

gis_proj_dir = "D:\WAZE\Shapefiles"
#gis_main_dir = "C:/Users/Jeff/Google Drive/research/Hampton Roads Data/Geographic Data/"
#elev_raster = '{}Raster/USGS Nor DEM/mosaic/nor_mosaic.tif'.format(gis_main_dir)

def read_shapefile_attribute_table(sf_name):
    sf = shapefile.Reader(sf_name)
    records = sf.records()
    df = pd.DataFrame(records)
    sf_field_names = [i[0] for i in sf.fields]
    df.columns = sf_field_names[1:]
    df.reset_index(inplace=True)
    return df

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import arcpy
#from gis_utils import read_shapefile_attribute_table, gis_proj_dir, gis_main_dir, elev_raster 
import numpy as np

import pandas as pd

from arcpy import env
from arcpy.sa import Raster, Ln, Tan, RemapValue, Reclassify, PathDistance, PathAllocation, \
    ExtractMultiValuesToPoints, FocalStatistics, NbrRectangle

arcpy.CheckOutExtension("spatial")
env.overwriteOutput = True
env.workspace = gis_proj_dir
#

    
def join_flooded_pts_with_rd_attributes(flood_pts='flooded_points.shp', road_lines='Street_Norfolk.shp', 
        out_file_name = 'fld_pts_rd_data.shp'):
    print "SpatialJoin_analysis"
    arcpy.SpatialJoin_analysis(flood_pts, road_lines, out_file_name, match_option='CLOSEST')

#def sample_road_points():
#    fld_pts = 'D:/WAZE/Shapefiles/fld_pts_rd_data.shp'
#    rd_pts = 'D:/WAZE/Shapefiles/rd_far_fld.shp'
#    fld_pt_df = read_shapefile_attribute_table(fld_pts)
#    rd_pts_df = read_shapefile_attribute_table(rd_pts)
#    cls = fld_pt_df.groupby('VDOT').agg(['count']).sum().sort_values(ascending=False)
#    cls = cls / cls.sum() * 100
#    cls = get_rd_classes(rd_pts)
#    num_samples = (cls * 750 / 100).round()
#
#    l = []
#    for c, n in num_samples.iteritems():
#        d = rd_pts_df[rd_pts_df['VDOT'] == c]
#        if d.shape[0] > n:
#            idx = d.sample(n=int(n)).index
#        else:
#            print "there are too few points. only {} when it's asking for {} for VDOT {}".format(
#                d.shape[0],
#                n,
#                c
#            )
#            idx = d.index
#        l.append(pd.Series(idx))
#    sampled = pd.concat(l)
#    out_file_name = 'sampled_road_pts.shp'
#    where_clause = '"FID" IN ({})'.format(",".join(map(str, sampled.tolist())))
#    print "Select_analysis"
#    arcpy.Select_analysis(rd_pts, out_file_name, where_clause)
#    return sampled

def make_rand_road_pts():
    """
    makes the 'rd_far_fld.shp' file which is points on roads that are spaced at least 300 ft from
    each other and at least 200 ft from any flood points
    :return: 
    """
    road_shapefile = "nor_roads_centerlines.shp"
    arcpy.Densify_edit(road_shapefile, densification_method='DISTANCE', distance=30)
    road_pts_file = 'rd_pts_all_1.shp'
    arcpy.FeatureVerticesToPoints_management(road_shapefile, road_pts_file)
    rand_rd_pts_file = 'rand_road.shp'
    rand_rd_pts_lyr = 'rand_road_lyr'
    arcpy.CreateRandomPoints_management(gis_proj_dir, rand_rd_pts_file, road_pts_file,
                                        number_of_points_or_field=50000,
                                        minimum_allowed_distance=200)
    print "rand_rd_points_file"
    fld_pts_file = 'flooded_points.shp'
    fld_pts_buf = 'fld_pt_buf.shp'
    arcpy.Buffer_analysis(fld_pts_file, fld_pts_buf, buffer_distance_or_field="200 Feet",
                          dissolve_option='ALL')
    print "buffer"
    arcpy.MakeFeatureLayer_management(rand_rd_pts_file, rand_rd_pts_lyr)
    arcpy.SelectLayerByLocation_management(rand_rd_pts_lyr, overlap_type='WITHIN',
                                           select_features=fld_pts_buf,
                                           invert_spatial_relationship='INVERT')
    rd_pts_outside_buf = 'rd_far_fld.shp'
    arcpy.CopyFeatures_management(rand_rd_pts_lyr, rd_pts_outside_buf)
    arcpy.JoinField_management(rd_pts_outside_buf, in_field='CID', join_table=road_pts_file,
                               join_field='FID')
    print "rd_points_outside_buf"
    
def main():
    # merge_flood_non_flood()
    # add_flood_pt_field()
    # add_is_downtown_field()
    # extract_values_to_points()
    # join_fld_pts_with_basin_attr()
    # add_is_in_hague()
    # update_db()
    
    make_rand_road_pts()
    join_flooded_pts_with_rd_attributes()
#    sample_road_points()
#    merge_flood_non_flood()

if __name__ == "__main__":
    main()
