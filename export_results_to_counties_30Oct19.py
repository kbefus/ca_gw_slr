#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:56:46 2019

Export masked_wt_depths and masked heads by county
1) Load model boundaries and counties
2) For each county find intersecting models
3) Loop through K and SLR scenarios
4) Make new raster domain from county shape
5) Insert model outputs into raster domain and save
6) Create shps of wt bins

Run after merge_gwhead_*.py

@author: kbefus
"""
import sys,os
import numpy as np
import time
import glob

res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
#code_dir = r'C:\Users\kbefus\OneDrive - University of Wyoming\ca_slr\scripts'
sys.path.insert(1,code_dir)

from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_utils import cgw_raster_utils as cru

import geopandas as gpd
import pandas as pd
from rasterio import mask
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile
from rasterio.crs import CRS
import rasterio
import affine
from astropy.convolution import convolve, Gaussian2DKernel
#%%
kernel = Gaussian2DKernel(1)

research_dir_orig = os.path.join(res_dir,'ca_slr')
research_dir = r'/mnt/762D83B545968C9F'
data_dir = os.path.join(research_dir,'data')
data_dir_orig = os.path.join(research_dir_orig,'data')
active_date = '27Dec19'
out_dir = os.path.join(data_dir,'masked_wt_depths_gdal_{}'.format(active_date))
output_dir = 'outputs_fill_gdal_{}'.format(active_date)
dem_dir = os.path.join(data_dir_orig,'gw_dems11Feb19')

model_types = ['model_lmsl_noghb','model_mhhw_noghb'][:1]


id_col = 'Id'
sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m
cell_spacing = 10. # meters
marine_mask_val = -500.
Kh_vals = [0.1,1.,10.][1:2]
utm10n = 3717
utm11n = 3718
nodata_val = -9999
ncrs =  {'init': 'epsg:{}'.format(utm10n)}
scrs = {'init': 'epsg:{}'.format(utm11n)}
crs_dict = {'norca':ncrs,'paca':ncrs,'sfbay':ncrs,
            'cenca':ncrs,'soca':scrs}
out_field= 'fbin_m'
new_marine_value = -1
flood_bins = [-np.inf,marine_mask_val,0,1,2,5,np.inf]
#min_sl = 1.5502 # minimum of MHHW (need to change to other value for MSL)


save_head = True
save_wtdepth = True

wt_fmt = '{}*wtdepth.tif'
head_fmt = '{}*head.tif'
dirname_fmt = 'slr_{0:3.2f}_m_Kh{1:3.1f}mday' # all outputs to this dir format
output_fmt = '{0}_{1}_{2}_Kh{3:3.1f}_slr{4:3.2f}m'

shp_date = '11Feb19'
nmodel_domains_shp = os.path.join(data_dir_orig,'ca_{}_slr_gw_domains_{}.shp'.format('n',shp_date))
ndomain_df = gpd.read_file(nmodel_domains_shp)
ndomain_df.crs = ncrs

smodel_domains_shp = os.path.join(data_dir_orig,'ca_{}_slr_gw_domains_{}.shp'.format('s',shp_date))
sdomain_df = gpd.read_file(smodel_domains_shp)
sdomain_df.crs = scrs
sdomain_df=sdomain_df.to_crs(ncrs)

all_modeldomains_df = pd.concat([ndomain_df,sdomain_df],ignore_index=True)

county_fname = os.path.join(data_dir_orig,'gis','CA_coastal_Counties_TIGER2016_UTM10N.shp')
ind_joined_df = gpd.read_file(county_fname)
ind_joined_df.crs = ncrs

model_county_df = gpd.sjoin(ind_joined_df,all_modeldomains_df,how='inner',op='intersects')
grouped_county_df = model_county_df.groupby(by=['NAME'])
#%%
icount=0
icount2=0
for icounty,idf in grouped_county_df:
    print('-------------- {} ---------------'.format(icounty))
    county_dir = '_'.join(idf['NAME'].values[0].split())
    
    icount+=1 # for bug checking
#    if icount>1:
#        continue
    
    # Make new grid extents and transform
    
    if 'soca' not in idf['ca_region'].unique():
        temp_crs = ncrs
        county_geom = idf.iloc[0].geometry
    elif 'soca' in idf['ca_region'].unique() and len(idf['ca_region'].unique()) ==1:
        temp_crs = scrs
        if hasattr(idf,'to_frame'):
            temp_df = gpd.GeoDataFrame(idf.to_frame().T,
                                             geometry=[idf.geometry],
                                             crs = ncrs)
        else:
            temp_df = idf.copy()
        temp_df = temp_df.to_crs(scrs) # convert back to scrs
        county_geom = temp_df.iloc[0].geometry
    else:
        temp_crs = ncrs
        county_geom = idf.iloc[0].geometry

    
    left, bottom, right, top = county_geom.bounds
    xres = cell_spacing
    yres = cell_spacing
    overlap_transform = affine.Affine(xres, 0.0, left,
                                      0.0, -yres, top)
    overlap_height = int(np.round((top-bottom)/yres))
    overlap_width = int(np.round((right-left)/xres))
    
    out_profile = {'driver':'GTiff','crs':temp_crs,'count':1,
                    'height': overlap_height,'dtype':rasterio.float32,
                    'width': overlap_width,'nodata':nodata_val,'compress':'lzw'}
    vrt_options = {'resampling': Resampling.bilinear,
                    'transform': overlap_transform,
                    'crs':temp_crs,
                    'height': overlap_height,
                    'width': overlap_width,'nodata':nodata_val,
                    'tolerance':0.001,'warp_extras':{'ALL_TOUCHED':'TRUE','NUM_THREADS':'10',
                                                     'SAMPLE_GRID':'YES','SAMPLE_STEPS':'100',
                                                     'SOURCE_EXTRA':'10'}}
    vrt_options2 = vrt_options.copy()
    vrt_options2.update({'resampling':Resampling.nearest})
    
    model_names = ['_'.join([itemp[1]['ca_region'],'{0:02d}'.format(itemp[1]['Id'])]) for itemp in idf.iterrows()]
    
    # Calculate fill amount for all dems
    orig_dems = [os.path.join(dem_dir,'{}_dem.tif'.format(imod)) for imod in model_names]
#    landfel_dems = [os.path.join(dem_dir,'{}_dem_landfel.tif'.format(imod)) for imod in model_names]
    out_dem = np.nan*np.zeros([overlap_height,overlap_width])
    for idem in orig_dems:
        # Construct county-based original dem
        with rasterio.open(idem) as srcdem:
            with WarpedVRT(srcdem,**vrt_options) as vrt1:
                temp_data = vrt1.read()[0]
                temp_data[temp_data==vrt1.nodata] = np.nan
                out_dem[~np.isnan(temp_data)] = temp_data[~np.isnan(temp_data)].copy()
            temp_data=None
            
            # Use nearest interpolation to fill in edges
            with WarpedVRT(srcdem,**vrt_options2) as vrt2:
                temp_data = vrt2.read()[0]
                temp_data[temp_data==vrt2.nodata] = np.nan
                out_dem[np.isnan(out_dem)& (~np.isnan(temp_data))] = temp_data[np.isnan(out_dem) & (~np.isnan(temp_data))]
            temp_data=None
    
    for model_type in model_types:
        print('---------- {} --------------'.format(model_type))
        datum_type = model_type.split('_')[1].upper()
        scenario_type = '_'.join(model_type.split('_')[1:])
        for Kh in Kh_vals:
            print('------------ Kh = {} ---------------'.format(Kh))
            kh_dir = 'Kh{0:3.1f}mday'.format(Kh)
            kh_dir=kh_dir.replace('.','p')
            for sealevel_elev in sealevel_elevs:
                print('--- SL = {} ----'.format(sealevel_elev))
                slr_dir = os.path.join(out_dir,model_type,
                                   dirname_fmt.format(sealevel_elev,Kh))
                
                for ftype,ftype_fmt in zip(['head'],[head_fmt]):
    #                icount2 +=1
    #                if icount2>1:
    #                    continue
                    outtemp = output_fmt.format(county_dir,ftype,scenario_type,Kh,sealevel_elev)
                    outtemp=outtemp.replace('.','p')
                    out_filename = os.path.join(data_dir,output_dir,model_type,ftype,county_dir,kh_dir,''.join([outtemp,'.tif']))
                    if os.path.isfile(out_filename):
                        continue
                    else:
                        
                        if not os.path.isdir(os.path.dirname(out_filename)):
                            os.makedirs(os.path.dirname(out_filename))
                        # Initialize raster as nans
                        out_raster = np.nan*np.zeros([overlap_height,overlap_width])
                        marine_mask = np.zeros([overlap_height,overlap_width],dtype=bool)
                        # Loop through rasters to fill in remaining data
                        for model_name in model_names:
                            print('---------- {} -------------'.format(model_name))
                            model_name2 = '_'.join([model_name.split('_')[0],str(int(model_name.split('_')[1]))])
                            temp_fname = glob.glob(os.path.join(slr_dir,ftype_fmt.format(model_name2)))[0]
                            with rasterio.open(temp_fname) as model_rast:
                                with MemoryFile() as memfile:
                                    with memfile.open(**model_rast.profile) as dataset:
                                        # need to set marine areas to nan
                                        head_temp = model_rast.read().squeeze()
                                        head_temp[head_temp==marine_mask_val] = np.nan
                                        dataset.write(np.expand_dims(head_temp,0))
                                        
                                        with WarpedVRT(dataset,**vrt_options) as vrt:
                                            temp_data = vrt.read()[0]
                                            temp_data[temp_data==vrt.nodata] = np.nan
                                            out_raster[~np.isnan(temp_data)] = temp_data[~np.isnan(temp_data)].copy()
                                            temp_data=None
                                        vrt.close()
                                    vrt=None        
                                    dataset.close()
                                    dataset=None
                                    # Use nearest interpolation to fill in edges
                                    with WarpedVRT(model_rast,**vrt_options2) as vrt:
                                        temp_data = vrt.read()[0]
                                        temp_data[temp_data==vrt.nodata] = np.nan
                                        marine_mask[temp_data==marine_mask_val] = True
                                        out_raster[np.isnan(out_raster)& (~np.isnan(temp_data))] = temp_data[np.isnan(out_raster) & (~np.isnan(temp_data))]
        #                                if ftype is 'head':
                                            # Correct for near-sea level interpolation errors, 1.5502 is min(MHHW sea levels)
            #                                out_raster[out_raster<min_sl] = temp_data[out_raster<min_sl]
                                        temp_data = None
                                    vrt.close()
                                vrt=None
                                memfile.close()
                            model_rast=None
        
                        # Calculate wt_depth from dem-head
                        wt_depth = out_dem-out_raster
        #                wt_depth[wt_depth<0] = 0.0 # force all water tables "above" land surface to be zero/at surface (e.g., at filled waterbodies)
                        wt_depth[marine_mask] = marine_mask_val
                        
                        if save_head:
                            out_raster[marine_mask] = marine_mask_val                    
                            # Crop to county boundaries
                            out_profile_temp = out_profile.copy()
                            out_profile_temp['dtype'] = out_raster.dtype
                            out_profile_temp['transform'] = overlap_transform
                            with MemoryFile() as memfile:
                                with memfile.open(**out_profile_temp) as dataset:
                                    dataset.write(np.expand_dims(out_raster,0))
                                    out_rast,tform = mask.mask(dataset,[county_geom],all_touched=True,crop=True)
                                    out_rast[out_rast==dataset.nodata]=np.nan
                            memfile.close()
                            
                            out_profile2 = out_profile.copy()
                            out_profile2['transform'] = tform
                            _,out_profile2['height'],out_profile2['width'] = out_rast.shape
                            out_profile2['dtype'] = out_rast.dtype
            
                            with rasterio.open(out_filename,'w',**out_profile2) as county_rast:
                                county_rast.write(out_rast)
                        
                        # --- Water table depth analyses and shapefile creation ---
                        if save_wtdepth:
                            outtemp = output_fmt.format(county_dir,'wt',scenario_type,Kh,sealevel_elev)
                            outtemp=outtemp.replace('.','p')
                            wtout_filename = os.path.join(data_dir,output_dir,model_type,'wt',county_dir,kh_dir,''.join([outtemp,'.tif']))
                            if not os.path.isdir(os.path.dirname(wtout_filename)):
                                os.makedirs(os.path.dirname(wtout_filename))
                            
                            # Crop to county boundaries
                            out_profile_temp = out_profile.copy()
                            out_profile_temp['dtype'] = wt_depth.dtype
                            out_profile_temp['transform'] = overlap_transform
                            with MemoryFile() as memfile:
                                with memfile.open(**out_profile_temp) as dataset:
                                    dataset.write(np.expand_dims(wt_depth,0))
                                    wt_depth,tform = mask.mask(dataset,[county_geom],all_touched=True,crop=True)
                                    wt_depth[wt_depth==dataset.nodata]=np.nan
                            memfile.close()
                            
                            out_profile2 = out_profile.copy()
                            out_profile2['transform'] = tform
                            _,out_profile2['height'],out_profile2['width'] = wt_depth.shape
                            out_profile2['dtype'] = wt_depth.dtype
        #                    with np.errstate(invalid='ignore'):
        #                        wt_depth[(wt_depth<0) & (wt_depth>-2)] = 0. # set near-zero values to zero
        #                        wt_depth[(wt_depth<0) & (wt_depth!=marine_mask_val)] = marine_mask_val # all more negative values to marine
                            
                            with rasterio.open(wtout_filename,'w',**out_profile2) as county_rast:
                                county_rast.write(wt_depth)
                        
                        # Save shp of wt depths
                        out_shp_dir = os.path.join(data_dir,output_dir,model_type,'shp',county_dir,kh_dir)
                        if not os.path.isdir(out_shp_dir):
                            os.makedirs(out_shp_dir)
        
                        tempout='{0}_{1}_slr{2:3.2f}m_Kh{3:2.1f}mday_wtbins'.format(county_dir,scenario_type,sealevel_elev,Kh)
                        tempout = tempout.replace('.','p')
                        out_shp_fname = os.path.join(out_shp_dir,''.join([tempout,'.shp']))
                                    
                        wt_depth = wt_depth.squeeze()
                        marine_mask2 = wt_depth==marine_mask_val
                        wt_depth[marine_mask2] = np.nan # set to nan for convolve
                        
                        out_rast2 = convolve(wt_depth,kernel,fill_value=np.nan)
        
                        out_rast2[marine_mask2] = marine_mask_val # reset marine mask
        #                with np.errstate(invalid='ignore'):
        #                    wt_depth[(wt_depth<0) & (wt_depth>-2)] = 0.
        #                    wt_depth[(wt_depth<0) & (wt_depth!=marine_mask_val)] = marine_mask_val
                        
                        out_bin_rast = np.nan*np.zeros_like(wt_depth,dtype=np.float)
                        bwtd = np.digitize(out_rast2.squeeze(),bins=flood_bins,right=True)
                        
                        for i,fb in enumerate(flood_bins):
                            # Find bins in binned_wt_depths that match flood_bins
                            out_bin_rast[bwtd==i] = fb
                        
        #                    out_bin_rast[bwtd==7] = -10
                        out_bin_rast[np.isinf(out_bin_rast)] = 6 # > than final flood_bin, >5 m
                        out_bin_rast[np.isnan(wt_depth)] = -10
                        out_bin_rast[marine_mask2] = new_marine_value
                        
                        in_dict={'XY':None,'Z':out_bin_rast,'in_proj':CRS(temp_crs).wkt,'gt':overlap_transform.to_gdal(),
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
                        shp_df.to_file(out_shp_fname,crs_wkt=CRS(temp_crs).wkt)
                                    
                    
                
            
            



