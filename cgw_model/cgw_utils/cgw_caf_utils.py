# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:08:17 2016

@author: kbefus
"""

#import numpy as np
#import shapefile
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.prepared import prep
from shapely.ops import unary_union
#import pandas as pd

from cgw_model.cgw_utils import cgw_feature_utils as cfu
cfu.speedups.enable()

def find_land_water(ws_path,in_domain_poly,find_land=True,find_water=False,
                    unq_col = 'FID'):
    '''
    Locate watersheds within the current domain, specified by in_domain_poly
    that are "land" and/or "water".
    
    Land polygons are used to define the active
    model cells that are within the overall buffered domain but not coastal waters.
    Coastal "nearshore" GHB cells are assigned where the buffered domain extends
    beyond the land_poly_merged polygon calculated in this function. This initial
    assignment is then checked using the elevation cutoff in ______ function.
    
    Parameters
    ----------
    
    ws_path:
                path to NOAA CAF watershed shapefile
    
    in_domain_poly:
                shapely polygon of active domain that includes buffering (i.e., Modflow model domain)
    
    find_land:  
                boolean switch to use function to find land watersheds
                (used in assigning boundary conditions for Modflow)
                
    find_water: 
                boolean switch to use function to find coastal waterbodies
                (used in assigning zones for ZoneBudget post-processing of model results)
    
    unq_col:
                column from input polygon to use as identifier for features
    
    Outputs
    ----------
    
    land_outputs:
        list of land_poly_merged,land_polys,iout_land,land_only_ids
        
        1) land_poly_merged: merged shapely polygon of domain
        
        2) land_polys: list of shapely polygons of watersheds inside domain
        
        3) iout_land: list of indexes for watershed locations (for accessing feature attributes)
        
        4) land_only_ids: list of identifiers of watershed polygons
    
    water_outputs:
        list of water_only_polys,iout_water,water_only_ids
    
        1) water_only_polys: list of shapely polygons of water features in domain
        
        2) iout_water: list of indexes of those water features
        
        3) water_only_ids: list of identifiers of those water features
        
    '''
    shp_col = 'Shapes'
    caf_df = cfu.shp_to_pd(ws_path,save_shps=True,shp_col=shp_col)
    if unq_col in ['FID']:
        # Create FID column
        caf_df['FID'] = caf_df.index.values
    poly_col = 'Polys'
    
    # ---- Find Land polygons ----
    if find_land:
        # Select catchments
        land_water_col = 'P_TYPE'
        land_water_class_col = 'POLYCLAS'
        
        classes_to_use = ['Coastal Drainage Area','Estuarine Drainage Area','Fluvial Drainage Area']#,'Portion of a US Drainage outside US']
        coastal_features = (caf_df[land_water_col] == 'land') & (caf_df[land_water_class_col].isin(classes_to_use)) & \
                        (caf_df['NAME'] != 'Canadian Territory')  # returns true where column equals 'land' or 'water'
        caf_land_df = caf_df[coastal_features].copy()
        caf_land_df[poly_col] = [Polygon(ishp.points) for ishp in caf_land_df[shp_col].values.tolist()]
        
        # Find internal catchments
        prepared_polygon = prep(in_domain_poly)
        iout_land=[ishp for ishp,ws_shp in enumerate(caf_land_df[poly_col].values.tolist()) if prepared_polygon.intersects(ws_shp)]
        land_poly_merged = unary_union([cfu.buffer_invalid(atemp) for atemp in caf_land_df[poly_col].values[iout_land]]) # can buffer to clean topology errors
        land_polys = caf_land_df[poly_col].values[iout_land]
        land_only_ids = caf_land_df[unq_col].values[iout_land]
        poly_inds = caf_land_df.index.values[iout_land]
    else:
        land_poly_merged,land_polys,poly_inds,land_only_ids = [],[],[],[]
    
    # ---- Find water polygons ----
    if find_water:
        # Select water catchments
        water_features = caf_df[land_water_col] == 'water'   # returns true where column equals 'land' or 'water'
        caf_water_df = caf_df[water_features].copy()
        caf_water_df[poly_col] = [Polygon(ishp.points) for ishp in caf_water_df[shp_col].values.tolist()]
        
        # Find internal catchments
        iout_water=[ishp for ishp,ws_shp in enumerate(caf_water_df[poly_col].values.tolist()) if prepared_polygon.intersects(ws_shp)]
        water_only_polys = [orient(temppoly) for temppoly in caf_water_df[poly_col].values[iout_water]]
        water_only_ids = caf_water_df[unq_col].values[iout_water]
        water_only_inds = caf_water_df.index.values[iout_water]
    else:
        water_only_polys,water_only_inds,water_only_ids = [],[],[]
        
    return [land_poly_merged,land_polys,poly_inds,land_only_ids],[water_only_polys,water_only_inds,water_only_ids]    
    