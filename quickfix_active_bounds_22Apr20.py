#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:58:39 2020

@author: kbefus
"""
import os,sys
import argparse
import glob
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.spatial import cKDTree as KDTree
from shapely.ops import unary_union
from shapely import speedups
speedups.enable()
res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
sys.path.insert(1,code_dir)


import rasterio
from cgw_model.cgw_utils import cgw_raster_utils as cru
from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_feature_utils as cfu

#%%

model_dir = os.path.join(r'/mnt/762D83B545968C9F/model_lmsl_noghb')
ca_regions = ['soca']
model_nums = [42,44,46,48,55] # code for saving active model boundaries is outdated and occassionally flips the domain in the northing direction

outdir_fmt = 'output*'
for ca_region in ca_regions:
    outdirs = glob.glob(os.path.join(model_dir,ca_region,outdir_fmt))
    activebound_dir = os.path.join(model_dir,ca_region,'active_bounds')
    for outdir in outdirs:
        _,datum,res,sl,kh = os.path.basename(outdir).split('_')
        sl = float(sl[2:-1])
        kh = float(kh[2:])
        
        for modelnum in model_nums:
            modelnum_dir = glob.glob(os.path.join(outdir,'{}*{}*'.format(ca_region,modelnum)))[0]
            model_name = os.path.basename(modelnum_dir)
            ct_fname = os.path.join(modelnum_dir,'{}_celltypes.tif'.format(model_name))

            active_out_fname = os.path.join(activebound_dir,'{}.shp'.format(model_name))
            
            with rasterio.open(ct_fname) as src:
                gt = src.profile['transform'].to_gdal()
                cs = src.profile['crs'].to_wkt()
                z_out = src.read()[0]
                z_out[z_out==src.nodata] = np.nan
            
            in_dict = {'Z':z_out,'in_proj':cs,
                       'out_shp':active_out_fname,
                       'gt':gt}
            cfu.raster_to_polygon_gdal(**in_dict)
        

