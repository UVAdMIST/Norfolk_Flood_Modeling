# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 14:38:42 2018

@author: Faria
"""


import shapefile
import pandas as pd

import os
import sys

import arcpy
import numpy as np
from collections import Counter

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from arcpy import env
from arcpy.sa import Raster, Ln, Tan, RemapValue, Reclassify, PathDistance, PathAllocation, \
    ExtractMultiValuesToPoints, FocalStatistics, NbrRectangle
gis_proj_dir = "D:\WAZE\Shapefiles"

arcpy.CheckOutExtension("spatial")
env.overwriteOutput = True
env.workspace = gis_proj_dir

def read_shapefile_attribute_table(sf_name):
    sf = shapefile.Reader(sf_name)
    records = sf.records()
    df = pd.DataFrame(records)
    sf_field_names = [i[0] for i in sf.fields]
    df.columns = sf_field_names[1:]
    df.reset_index(inplace=True)
    return df
    
def join_flooded_pts_with_rd_attributes(flood_pts='flooded_points.shp', road_lines='Street_Norfolk.shp', 
        out_file_name = 'fld_pts_rd_data.shp'):
    print "SpatialJoin_analysis"
    arcpy.SpatialJoin_analysis(flood_pts, road_lines, out_file_name, match_option='CLOSEST')

def get_road_cls_prop():
    rd_shapefile = "D:\WAZE\Shapefiles/nor_roads_centerlines.shp"
    rd_pts_shapefile = 'D:/WAZE/Shapefiles/rd_far_fld.shp'
    fld_pts_shapefile = 'D:/WAZE/Shapefiles/flooded_points.shp'

    rd_df = read_shapefile_attribute_table(rd_shapefile)
    rd_pts_df = read_shapefile_attribute_table(rd_pts_shapefile)
    fld_pts_df = read_shapefile_attribute_table(fld_pts_shapefile)
    c = Counter( rd_df['VDOT'] )
    VDOT_cls = c.items() 
    print VDOT_cls
    
    c_pts = Counter( rd_pts_df['VDOT'] )
    VDOT_cls_pts = c_pts.items() 
    print VDOT_cls_pts
        
    prop = []
    for i in range(len(VDOT_cls)):
        print VDOT_cls[i][1]
        p = float(VDOT_cls[i][1])*100/len(rd_df['VDOT'])
        print p
        prop.append(p)
        
    num_samples = []
    for i in prop:
        n = i*len(fld_pts_df['Date'])/100
        num_samples.append(int(round(n)))
        return c_pts, VDOT_cls_pts, num_samples

def SelectRandomByCount (layer_shp, layer_df):
    import random
    pts_lyr = 'pts_lyr'
    pts_lyr_vdot = 'pts_lyr_VDOT'
    arcpy.MakeFeatureLayer_management(layer_shp, pts_lyr)               #create layer of road points shapefile

    vdotFldName = arcpy.ListFields(layer_shp, "VD*")[0].name
    delimVdotFld = arcpy.AddFieldDelimiters (layer_shp, vdotFldName)
    c, VDOT_cls_pts, num_samples = get_road_cls_prop()
    vdot = []
    for i in range(0,len(c)):
        v= VDOT_cls_pts[i][0]
        vdot.append(int(v))
#    vdotsStr = ", ".join (map (str, vdot))

    for i,j in zip(vdot,num_samples):
        sql_vdot = "{0} IN ({1})".format(delimVdotFld,i)

        arcpy.MakeFeatureLayer_management(layer_shp, pts_lyr_vdot, sql_vdot) #selecting points with VDOT class 1 to 10
        oids = [oid for oid, in arcpy.da.SearchCursor (pts_lyr_vdot, "OID@")]
        oidFldName = arcpy.Describe (layer_shp).OIDFieldName

        delimOidFld = arcpy.AddFieldDelimiters (layer_shp, oidFldName)    
        randOids = random.sample (oids, j)
        oidsStr = ", ".join (map (str, randOids))

        sql = "{0} IN ({1})".format (delimOidFld, oidsStr)
        z = arcpy.SelectLayerByAttribute_management (pts_lyr, "", sql)
        arcpy.CopyFeatures_management(z, "D:\WAZE\Shapefiles/z%s.shp"%(i))
    
    shplist =  arcpy.ListFeatureClasses('z*.shp') 
    print "Merge_management"
    arcpy.Merge_management(shplist, os.path.join(gis_proj_dir, 'sampled_rd_pts.shp'))    
    
def merge_flood_non_flood():
    flood_pts = 'D:\WAZE\Shapefiles/fld_pts_rd_data.shp'
    non_flood_pts = 'D:\WAZE\Shapefiles\sampled_rd_pts.shp'
    out_file_name = 'D:\WAZE\Shapefiles/fld_nfld_pts.shp'
    print "Merge_management"
    arcpy.Merge_management([flood_pts, non_flood_pts], out_file_name) 

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
                                        minimum_allowed_distance='200 Feet')
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
    make_rand_road_pts()
    join_flooded_pts_with_rd_attributes()
    get_road_cls_prop()
    rd_pts_shapefile = 'D:/WAZE/Shapefiles/rd_far_fld.shp'
    rd_pts_df = read_shapefile_attribute_table(rd_pts_shapefile)
    SelectRandomByCount (rd_pts_shapefile,rd_pts_df)
    merge_flood_non_flood()


if __name__ == "__main__":
    main()
