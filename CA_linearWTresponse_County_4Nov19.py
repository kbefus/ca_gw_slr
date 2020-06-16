#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:03:45 2019

Calculate and save linear response head, wt, and shp files
Run after export_results_to_counties_*.py

@author: kbefus
"""

import sys,os
import numpy as np
import time,glob 
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask.array as da
from affine import Affine
import rasterio

from astropy.convolution import convolve, Gaussian2DKernel

res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
sys.path.insert(1,code_dir)

from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_raster_utils as cru
from cgw_model.cgw_utils import cgw_feature_utils as cfu
import zipfile

#%%
kernel = Gaussian2DKernel(1)

def zipfolder(foldername, target_dir, prefix=None,condition=None): 
    ''' Source: https://stackoverflow.com/a/10480441           
    '''
    
    if prefix is None:
        fname=foldername + '.zip'
    else:
        fname = os.path.join(os.path.dirname(foldername),prefix+os.path.basename(foldername)+'.zip')
        
    with zipfile.ZipFile(fname, 'w', zipfile.ZIP_DEFLATED,compresslevel=9) as zipobj:
        rootlen = len(target_dir) + 1
        for base, dirs, files in os.walk(target_dir):
            for file in files:
                fn = os.path.join(base, file)
                if condition is not None:
                    if not condition(fn):
                        continue
                zipobj.write(fn, fn[rootlen:])
#%%
research_dir_orig = os.path.join(res_dir,'ca_slr')
data_dir_orig = os.path.join(research_dir_orig,'data')


skip_county = ['San_Benito']
research_dir = r'/mnt/762D83B545968C9F'
main_model_dir = os.path.join(research_dir,'model')
data_dir = os.path.join(research_dir,'data')
results_dir = os.path.join(research_dir,'results')
active_date = '27Dec19'
output_dir = 'outputs_fill_gdal_{}'.format(active_date)
dem_dir = os.path.join(data_dir_orig,'gw_dems11Feb19')

model_types = ['model_lmsl_noghb','model_mhhw_noghb'][1:]

output_fmt = '{0}_{1}_{2}_Kh{3:3.1f}_slr{4:3.2f}m'
sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m
Kh_vals = [0.1,1.,10.][1:2]
marine_mask_val = -500.
out_field= 'fbin_m'
new_marine_value = -1
flood_bins = [-np.inf,marine_mask_val,0,1,2,5,np.inf]

new_ftype = 'linresponse_{}'

# Load county data
county_fname = os.path.join(data_dir_orig,'gis','CA_coastal_Counties_TIGER2016_UTM10N.shp')
ind_joined_df = gpd.read_file(county_fname).sort_values('NAME')

run_analysis=True
#%%
# Loop through counties
all_ftype_county_dirs=[]
for indtemp,irow in list(ind_joined_df.iterrows()):
    county_dir = '_'.join(irow['NAME'].split())
    print('----- {} -------'.format(county_dir))
    if county_dir in skip_county:
        continue    
    for model_type in model_types:
        datum_type = model_type.split('_')[1].upper()
        scenario_type = '_'.join(model_type.split('_')[1:])
        for Kh in Kh_vals: # ----------- Loop through K --------------
            print('------------ Kh = {} ---------------'.format(Kh))
            kh_dir = 'Kh{0:3.1f}mday'.format(Kh)
            kh_dir=kh_dir.replace('.','p')
            dem = None
            for sl in sealevel_elevs: # - Loop through Sea levels --------
                print('--- SL = {} ----'.format(sl))
                for ftype in ['wt','head']: # Loop through file types
                    if ftype == 'wt':
                        # Use fill-corrected wtdepths
                        in_dir= os.path.join(data_dir,output_dir,model_type,ftype,county_dir,kh_dir)
                    else:
                        in_dir= os.path.join(data_dir,output_dir,model_type,ftype,county_dir,kh_dir)
                    
                    outtemp = output_fmt.format(county_dir,ftype,scenario_type,Kh,sl)
                    outtemp=outtemp.replace('.','p')
                    in_fname = os.path.join(in_dir,''.join([outtemp,'.tif']))
                    if run_analysis:
                        if sl == 0:
                            if ftype == 'wt':
                                # Load water table data
                                wt = xr.open_rasterio(in_fname,chunks=(1e3,1e3))[0]
                                nodata = wt.nodatavals[0]
                                wt=wt.where(wt!=nodata)
                                wt_sl0 = wt.copy()
            #                    wt=wt.where(wt!=marine_value)
                            else:
                                # Load head data
                                head = xr.open_rasterio(in_fname,chunks=(1e3,1e3))[0]
                                nodata = head.nodatavals[0]
                                head=head.where(head!=nodata)
                                head_sl0 = head.copy()
            #                    head=head.where(head!=marine_value)
                            
                            continue
                        else:
                            if ftype == 'wt':
                                # Load water table data
                                wt = xr.open_rasterio(in_fname,chunks=(1e3,1e3))[0]
                                with rasterio.open(in_fname) as temp:
                                    wt_profile = temp.profile
                                temp=None
                                nodata = wt.nodatavals[0]
                                wt=wt.where(wt!=nodata)
                        
                        if sl != 0:
                            #head+wt_depth = land elevation
                            dem = wt_sl0+head_sl0
                            
                            meta2 = wt_profile.copy()
                             
                            # Decrease thickness of vadose zone by amount of slr
                            shifted_wt = wt_sl0 - sl # modern water table depth - sea_level_rise
    #                        with np.errstate(invalid='ignore'):
    #                            shifted_wt = xr.where(shifted_wt<0,0,shifted_wt) # all wt<0 depth set to 0 = emergent gw
                            shifted_wt = xr.where(wt==marine_mask_val,marine_mask_val,shifted_wt) # set marine innundation
                            
                            # Calculate hydraulic head (i.e., wt elevation)
                            shifted_wt_head = dem-shifted_wt
                            shifted_wt_head = xr.where(wt==marine_mask_val,marine_mask_val,shifted_wt_head) # set marine innundation
                        else:
                            shifted_wt = wt
                            shifted_wt_head = head
                    
                           
                    # Save outputs
                    for ftype in ['wt','head']:
                        path_to_county_dir = os.path.join(data_dir,output_dir,model_type,ftype,county_dir)
                        if path_to_county_dir not in all_ftype_county_dirs:
                            all_ftype_county_dirs.append(path_to_county_dir)
                        if run_analysis:
                            out_dir= os.path.join(path_to_county_dir,new_ftype.format(kh_dir))
                            if not os.path.isdir(out_dir):
                                os.makedirs(out_dir)
                            
                            outtemp = output_fmt.format(county_dir,new_ftype.format(ftype),scenario_type,Kh,sl)
                            outtemp=outtemp.replace('.','p')
                            out_fname = os.path.join(out_dir,''.join([outtemp,'.tif']))
                            
                            if ftype == 'wt':
                                meta2.update({'dtype':shifted_wt.dtype})
                                shifted_wt=shifted_wt.values
    #                            with np.errstate(invalid='ignore'):
    #                                shifted_wt[(shifted_wt<0) & (shifted_wt>-2)] = 0. # set near-zero values to zero
    #                                shifted_wt[(shifted_wt<0) & (shifted_wt!=marine_mask_val)] = marine_mask_val # all more negative values to marine
                    
                                with rasterio.open(out_fname,'w',**meta2) as dest:
                                    dest.write(np.expand_dims(shifted_wt,axis=0))
                                
                                # Also make shapefile with bins
                                out_shp_dir = os.path.join(data_dir,output_dir,model_type,'shp',county_dir,new_ftype.format(kh_dir))
                                if not os.path.isdir(out_shp_dir):
                                    os.makedirs(out_shp_dir)
                                
                                tempout='{0}_{1}_slr{2:3.2f}m_Kh{3:3.1f}mday_wtbins_lin'.format(county_dir,scenario_type,sl,Kh)
                                tempout = tempout.replace('.','p')
                                out_shp_fname = os.path.join(out_shp_dir,''.join([tempout,'.shp']))
                                
                                marine_mask2 = shifted_wt==marine_mask_val
                                shifted_wt[marine_mask2] = np.nan # set to nan for convolve
                                
                                out_rast2 = convolve(shifted_wt,kernel,fill_value=np.nan)
            
                                out_rast2[marine_mask2] = marine_mask_val # reset marine mask
    #                            with np.errstate(invalid='ignore'):
    #                                shifted_wt[(shifted_wt<0) & (shifted_wt>-2)] = 0.
    #                                shifted_wt[(shifted_wt<0) & (shifted_wt!=marine_mask_val)] = marine_mask_val
                                
                                out_bin_rast = np.nan*np.zeros_like(shifted_wt,dtype=np.float)
                                bwtd = np.digitize(out_rast2.squeeze(),bins=flood_bins,right=True)
                                
                                for i,fb in enumerate(flood_bins):
                                    # Find bins in binned_wt_depths that match flood_bins
                                    out_bin_rast[bwtd==i] = fb
                                
            #                    out_bin_rast[bwtd==7] = -10
                                out_bin_rast[np.isinf(out_bin_rast)] = 6 # > than final flood_bin, >5 m
                                out_bin_rast[np.isnan(shifted_wt)] = -10
                                out_bin_rast[marine_mask2] = new_marine_value
                                
                                in_dict={'XY':None,'Z':out_bin_rast,'in_proj':meta2['crs'].wkt,'gt':meta2['transform'].to_gdal(),
                                         'out_shp':out_shp_fname,'field_name':out_field}
                                cfu.raster_to_polygon_gdal(**in_dict)
                                
                                out_bin_rast=None
                                bwtd=None
                                
                                # Cull some unwanted data
                                shp_df = gpd.read_file(out_shp_fname)
                                shp_df = shp_df.loc[shp_df[out_field]>-10]
                                shp_df['type']='terrestrial'
                                shp_df.loc[shp_df[out_field]==new_marine_value,'type']='marine/tidal'
                                
                                shp_df['wtdepth'] = '>5 m; deep'
                                shp_df.loc[shp_df[out_field]==new_marine_value,'wtdepth']='marine/tidal'
                                shp_df.loc[shp_df[out_field]==0,'wtdepth']='0 m; emergent'
                                shp_df.loc[shp_df[out_field]==1,'wtdepth']='0-1 m; very shallow'
                                shp_df.loc[shp_df[out_field]==2,'wtdepth']='1-2 m; shallow'
                                shp_df.loc[shp_df[out_field]==5,'wtdepth']='2-5 m; moderate'
                                
                                # Fix "invalid" polygons
                                valid_polys = shp_df.is_valid
                                shp_df.loc[~valid_polys,'geometry'] = shp_df.loc[~valid_polys].buffer(0)
                                shp_df.to_file(out_shp_fname,crs_wkt=meta2['crs'].wkt)
                                                
                                
                                
    #                            write_dict = {'fname':out_fname,'X':None,'Y':None,
    #                                          'geodata':geodata,
    #                                          'Vals':shifted_wt.values,'proj_wkt':wt.crs}
                            else:
    #                            write_dict = {'fname':out_fname,'X':None,'Y':None,
    #                                          'geodata':geodata,
    #                                          'Vals':shifted_wt_head.values,'proj_wkt':head.crs}
                                meta2.update({'dtype':shifted_wt_head.dtype})
                                with rasterio.open(out_fname,'w',**meta2) as dest:
                                    dest.write(np.expand_dims(shifted_wt_head.values,axis=0))
                            
#                        cru.write_gdaltif(**write_dict)
#                    
                    
#%% Write zip files

#condition = lambda a: 'linresponse' in a and '0.01' not in a                    
#for icounty_dir in all_ftype_county_dirs:
#    zipfolder(icounty_dir,icounty_dir,prefix='linresponse_',condition=condition)          
                
                    