#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:08:30 2019

@author: kbefus
"""

import sys,os
import numpy as np
import glob
import pandas as pd
import geopandas as gpd
#import dask.array as da
import rasterio
from rasterio import mask
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile
from scipy.spatial import cKDTree as KDTree

res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
sys.path.insert(1,code_dir)

#%%
def xy_from_affine(tform=None,nx=None,ny=None):
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
    return X,Y

def read_geotiff(in_fname,band=0):
    with rasterio.open(in_fname) as src:
        data = src.read()[band]
        data[data==src.nodata]=np.nan
        ny,nx = data.shape
        X,Y = xy_from_affine(src.transform,nx,ny)
        profile = src.profile
    return X,Y,data,profile

#%%
# ----------- Region directory information -----------
research_dir_orig = os.path.join(res_dir,'ca_slr')
data_dir_orig = os.path.join(research_dir_orig,'data')
research_dir = r'/mnt/762D83B545968C9F'
output_dir = os.path.join(research_dir,'data','outputs_fill_gdal_29Oct19')

results_dir = os.path.join(research_dir,'results','no_ghb','wt_analysis')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

model_types = ['model_lmsl_noghb','model_mhhw_noghb']


id_col = 'Id'
sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m
Kh_vals = [0.1,1.,10.]
datum_type = 'MHHW'
cell_spacing = 10. # meters
file_fmt = '{0}_{1}_{2}_Kh{3:3.2f}_slr{4:3.2f}m'
head_fmt = '{}_head.tif'
wt_fmt = '{}_wtdepth.tif'
cell_fmt = '{}_celltypes.tif'
marine_value = -500.
other_nan_val = -9999.
active_date = '6Nov19'

col_fmt = '{0}_count_sl{1:3.2f}_Kh{2:3.2f}_{3}_{4}'

wt_col = 'wtdepth'
#dx = 20
#dist_inland_bins = np.arange(0,1e4+dx,dx)
#dist_inland_bins = np.hstack([dist_inland_bins,np.inf])
#%%
out_hist_data = []
out_hist_cols = []
out_stats = []
for linear_resp_bool in [False,True]:
    for model_type in model_types:
        datum_type = model_type.split('_')[1].upper()
        scenario_type = '_'.join(model_type.split('_')[1:])
        
        if linear_resp_bool:
            d2 = '_'.join([datum_type,'linresp'])
        else:
            d2 = datum_type
        
        wt_dir = os.path.join(output_dir,model_type,'wt')
            
        county_dirs = [idir for idir in glob.glob(os.path.join(wt_dir,'*')) if os.path.isdir(idir)]
    
        for Kh in Kh_vals:    
            print('------------ Kh = {} ---------------'.format(Kh))
            kh_dir = 'Kh{0:3.2f}mday'.format(Kh)
            kh_dir=kh_dir.replace('.','p')
            
            type_all = {}
            
            for county_dir in county_dirs:
                county_name = os.path.basename(county_dir)
                print('------- {} --------'.format(county_name))
                
                for isl,sl in enumerate(sealevel_elevs):
                    print('--- SL = {} ----'.format(sl))
                    
                    if sl not in list(type_all.keys()):
                        type_all.update({sl:{}})
                    
                    # Load water table depth tifs only for modern sl
                    if sl==0.0:
                        tempname = file_fmt.format(county_name,'wt',scenario_type,Kh,sl)
                        tempname = tempname.replace('.','p')
                        wt_fname = os.path.join(county_dir,kh_dir,'{}.tif'.format(tempname))
                        
                        x,y,wt_sl0,profile = read_geotiff(wt_fname)
                        with np.errstate(invalid='ignore'):
                            wt_sl0[(wt_sl0<0) & (wt_sl0!=marine_value)]=0 # set negative water tables to zero
                            wt_sl0[wt_sl0<=other_nan_val] = np.nan
                        
                        # Assign marine mask
                        marine_mask = wt_sl0 == marine_value
                        wt_sl0[marine_mask] = np.nan

                        # Calculate distance inland raster
                        notnan_or_marine = ~np.isnan(wt_sl0)
                        marine_tree = KDTree(np.c_[x[marine_mask],y[marine_mask]])
                        dist,marine_inds = marine_tree.query(np.c_[x[notnan_or_marine],y[notnan_or_marine]])
                        dist_inland_array = np.nan*np.ones_like(wt_sl0)
                        dist_inland_array[notnan_or_marine] = dist.copy()
                            
                    
                    # Load shapefile wt bins
                    cdir_wt = os.path.join(output_dir,model_type,'shp',os.path.basename(county_dir))
                    if linear_resp_bool and sl==0:
                        # Load original, not lin, modeled output for sl=0
                        temp_fname = '{0}_{1}_slr{2:3.2f}m_Kh{3:3.2f}mday_emergent'.format(county_name,scenario_type,sl,Kh) 
                        temp_fname = temp_fname.replace('.','p')
                        shp_name = os.path.join(cdir_wt,kh_dir,'{}.shp'.format(temp_fname))
                        shp_df = gpd.read_file(shp_name)
                    else:
                        temp_fname = '{0}_{1}_slr{2:3.2f}m_Kh{3:3.2f}mday_emergent'.format(county_name,scenario_type,sl,Kh)
                        if linear_resp_bool:
                           kh_dir2 = '_'.join(['linresponse',kh_dir])
                           temp_fname = '{}_lin'.format(temp_fname)
                        else:
                            kh_dir2 = kh_dir
                        temp_fname = temp_fname.replace('.','p')
                        shp_name = os.path.join(cdir_wt,kh_dir2,'{}.shp'.format(temp_fname))
                        shp_df = gpd.read_file(shp_name)
                        
                    unique_types = shp_df[wt_col].unique()
                    
                    with MemoryFile() as memfile:
                        with memfile.open(**profile) as dataset:
                            dataset.write(dist_inland_array[None,:])
                    
                            for temp_type in unique_types:
                                temp_shp = shp_df[shp_df[wt_col]==temp_type].copy()
                                
#                                type_hist_list = np.zeros_like(dist_inland_bins[:-1])
                                for ifeature in temp_shp.geometry.values:
                                    # Sample distance array using the feature
                                    mask_dist,tform = mask.mask(dataset,[ifeature],crop=True)
                                    mask_dist = mask_dist.squeeze()
                                    mask_dist = mask_dist[mask_dist>other_nan_val].copy()
#                                    counts,edges = np.histogram(mask_dist[~np.isnan(mask_dist)],bins=dist_inland_bins)
#                                    type_hist_list += counts # sum for each feature in bin
#                                    type_all.extend(mask_dist[~np.isnan(mask_dist)])
                                    
                                    if temp_type not in list(type_all[sl].keys()):
                                        type_all[sl].update({temp_type:[]})
                                    
                                    type_all[sl][temp_type].extend(mask_dist[~np.isnan(mask_dist)])
                                    
#                                if 'bin_left' not in out_hist_cols:
#                                    left,right = edges[:-1],edges[1:]
#                                    out_hist_cols.extend(['bin_left','bin_right'])
#                                    out_hist_data.extend([left,right])
                                
#                                # Store in main list for saving to csv
#                                out_hist_data.append(type_hist_list)
#                                
                                
#                                out_hist_cols.append(col_fmt.format(county_name,sl,Kh,d2,temp_type))
#                            
#                                # save basic stats on dist data
                                
                                
    
            # Combine for all ca for each sea level
            for isl,sl in enumerate(sealevel_elevs):
                for temp_type in unique_types:
                    
                    temp_out = np.array(type_all[sl][temp_type])
                    if len(temp_out)>0:
                        out_stats.append(['All',sl,Kh,d2,temp_type,
                                          np.nanmedian(temp_out),np.nanmean(temp_out),
                                          np.nanmax(temp_out),np.nanmin(temp_out),
                                          np.nanstd(temp_out),temp_out.shape[0]])
                    else:
                        out_stats.append(['All',sl,Kh,d2,temp_type,other_nan_val,other_nan_val,
                                          other_nan_val,other_nan_val,other_nan_val,0])
                    
                    # Remove data as the rows are made in effort to free up memory
                    type_all[sl][temp_type] = None
                    
                    
            temp_out = None

type_all=None
# Save outputs
#out_fname = os.path.join(results_dir,'wtdepth_bins_distinland_hists_{}.csv'.format(active_date))
#out_df = pd.DataFrame(np.array(out_hist_data).T,columns=out_hist_cols)
#out_df.to_csv(out_fname,index_label='type')

out_fname2 = os.path.join(results_dir,'wtdepth_bins_distinland_AllCAonly_stats_{}.csv'.format(active_date))
out_cols2 = ['County','Sea_level_m','Kh_mday','Datum_Model',
             'WT_bin','Median_dist_m','Mean_dist_m','Max_dist_m',
             'Min_dist_m','Std_dist_m','Count_dist']
out_df2 = pd.DataFrame(out_stats,columns=out_cols2)
out_df2.to_csv(out_fname2,index=False)
                    
                        



