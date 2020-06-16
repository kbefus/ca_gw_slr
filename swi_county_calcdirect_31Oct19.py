#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:11:37 2019

@author: kbefus
"""

import os,sys
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

rho_f = 1000
rho_s = 1025
density_ratio = rho_f/(rho_s-rho_f) #between 33-50

def ghybenherzberg(head=None,alpha=density_ratio):
    '''
    
    From Werner and Simmons 2009
    '''
    swi_depth = alpha*head
    return swi_depth


#%%

research_dir_main = os.path.join(res_dir,'ca_slr')
data_dir_orig = os.path.join(research_dir_main,'data')
research_dir = r'/mnt/762D83B545968C9F'
data_dir = os.path.join(research_dir,'data')

active_date = '27Dec19'
swi_date = active_date
swi_dir = os.path.join(data_dir,'swi_outputs_{}'.format(swi_date))

save_dir = os.path.join(data_dir_orig,'swi_outputs_{}'.format(swi_date),'county_data_{}'.format(swi_date))


output_dir = 'outputs_fill_gdal_{}'.format(active_date)
head_dir = os.path.join(data_dir,output_dir)
head_fmt = '{0}_{1}_Kh{2:3.1f}_slr{3:3.2f}m'

model_types = ['model_lmsl_noghb','model_mhhw_noghb']
analytical_types = ['ghybenherzberg']

cs_sl_sal = 'NAD83'
sl_fname = os.path.join(data_dir_orig,'sea_level','CA_sl_{}_12Feb18.txt')
sal_fname = os.path.join(data_dir_orig,'salinity','CA_sal_12Feb18.txt')
sal_data,_ = cru.read_txtgrid(sal_fname)
sal_data = np.array(sal_data) # lon, lat, density
buffer0 = 1e3 # m buffer around domain

sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m
cell_spacing = 10. # meters
marine_mask_val = -500.
Kh_vals = [0.1,1.,10.]


max_thick = 50.
cell_types = {-2:'marine/tidal',1:'swi_footprint',2:'fresh_footprint'}

linear_resp_bool = True
county_sl_data = {}
for analytical_type in analytical_types:
    for model_type in model_types:
        print('---------- {} --------------'.format(model_type))
        datum_type = model_type.split('_')[1].upper()
        scenario_type = '_'.join(model_type.split('_')[1:])
        
        # Load sea_level and density data for all of CA
        sl_data,_ = cru.read_txtgrid(sl_fname.format(datum_type.upper()))
        sl_data = np.array(sl_data) # lon, lat, sl
        
        county_dirs = glob.glob(os.path.join(head_dir,model_type,'head','*'))
        county_dirs = [idir for idir in county_dirs if os.path.isdir(idir) and 'Error' not in idir]
        
        for county_dir in county_dirs:
            county_name = os.path.basename(county_dir)
            print('----------- {} -------------'.format(county_name))
            for Kh in Kh_vals:
                print('------------ Kh = {} ---------------'.format(Kh))
                kh_dir = 'Kh{0:3.1f}mday'.format(Kh)
                kh_dir=kh_dir.replace('.','p')
                if linear_resp_bool:
                    kh_dir = '_'.join(['linresponse',kh_dir])
                for sealevel_elev in sealevel_elevs:
                    print('--- SL = {} ----'.format(sealevel_elev))
                    if linear_resp_bool:

                        if sealevel_elev == 0:
                            continue
                        temp_name = head_fmt.format(county_name,'linresponse_head_{}'.format(scenario_type),Kh,sealevel_elev)
                    else:
                        temp_name = head_fmt.format(county_name,'head_{}'.format(scenario_type),Kh,sealevel_elev)
                        
                    temp_name = temp_name.replace('.','p')
                    head_fname = os.path.join(county_dir,kh_dir,'{}.tif'.format(temp_name))
                    x,y,head,profile = read_geotiff(head_fname)
                    active_proj = profile['crs'].to_wkt()
                    marine_mask = head==marine_mask_val
                    landmarinecells=marine_mask.astype(int)
                    landmarinecells[np.isnan(head)]=-100
                    head[marine_mask]=np.nan

                    if county_name in list(county_sl_data.keys()):
                        sl_mean = county_sl_data[county_name]['sl_mean']
                        den_mean = county_sl_data[county_name]['den_mean']
                        
                    else:
                        sl_xy_proj = cru.projectXY(sl_data[:,:2],inproj=cs_sl_sal,outproj=active_proj)
                        sl_data_proj = np.column_stack([sl_xy_proj,sl_data[:,2]])
                        sal_xy_proj = cru.projectXY(sal_data[:,:2],inproj=cs_sl_sal,outproj=active_proj)
                        sal_data_proj = np.column_stack([sal_xy_proj,sal_data[:,2]])
                        
                        temp_extent = [x.min(),x.max(),
                                       y.min(),y.max()]
                        if county_name in ['Napa',]:
                            buffer_temp = 1.2e4
                        elif county_name in ['Contra_Costa','Marin','Alameda','Santa_Clara']:
                            buffer_temp = 8e3
                        else:
                            buffer_temp = buffer0
                        
                        inpts = (sl_data_proj[:,:1]<=temp_extent[1]+buffer_temp) & (sl_data_proj[:,:1]>=temp_extent[0]-buffer_temp) \
                                & (sl_data_proj[:,1:2]<=temp_extent[3]+buffer_temp) & (sl_data_proj[:,1:2]>=temp_extent[2]-buffer_temp)
                        sl_values = sl_data_proj[inpts.ravel(),2:]
                        sl_mean = np.nanmean(sl_values)
                        
                        inpts2 = (sal_data_proj[:,:1]<=temp_extent[1]+buffer_temp) & (sal_data_proj[:,:1]>=temp_extent[0]-buffer_temp) \
                                & (sal_data_proj[:,1:2]<=temp_extent[3]+buffer_temp) & (sal_data_proj[:,1:2]>=temp_extent[2]-buffer_temp)
                        den_vals = sal_data_proj[inpts2.ravel(),2:]
                        den_mean = np.nanmean(den_vals)
                        
                        county_sl_data.update({county_name:{'sl_mean':sl_mean,
                                                            'sl_count':len(sl_values),
                                                            'sl_min':np.nanmin(sl_values),
                                                            'sl_max':np.nanmax(sl_values),
                                                            'den_mean':den_mean,
                                                            'den_count':len(den_vals),
                                                            'den_min':np.nanmin(den_vals),
                                                            'den_max':np.nanmax(den_vals)}})
#                    g_dict = {'xi':(x,y),'method':'nearest'}   
#                    sea_level = cru.griddata(sl_data_proj[inpts.ravel(),:2],sl_data_proj[inpts.ravel(),2:],**g_dict).squeeze()+sealevel_elev
                    
                    # Calculate fw-sw interface with different functions
                    swi_ghybenherz = ghybenherzberg(head-(sl_mean+sealevel_elev),rho_f/(den_mean-rho_f))
                    
                    # Make shapefiles of inland extent and calculate inland migration as f(slr)
                    outshpdir1 = os.path.join(save_dir,county_name)
                    if not os.path.isdir(outshpdir1):
                        os.makedirs(outshpdir1)
                    if linear_resp_bool:
                        ghz_shp_fname = os.path.join(outshpdir1,'{}_linresponse_{}_{}_GH_swi.shp'.format(county_name,scenario_type,'_'.join(temp_name.split('_')[-2:])))
                    else:
                        ghz_shp_fname = os.path.join(outshpdir1,'{}_{}_{}_GH_swi.shp'.format(county_name,scenario_type,'_'.join(temp_name.split('_')[-2:])))
                    out_ghz = -100*np.ones_like(head)
                    with np.errstate(invalid='ignore'):
                        out_ghz[(swi_ghybenherz <= max_thick)] = 1
                        out_ghz[marine_mask] = -2
                        out_ghz[(~marine_mask) & (out_ghz<-2) & ~np.isnan(head)] = 2 # inside model but not swi footprint
                    ghz_outfield = 'swi_types'
                    in_dict={'XY':None,'Z':out_ghz,'in_proj':active_proj,'gt':profile['transform'].to_gdal(),
                                 'out_shp':ghz_shp_fname,'field_name':ghz_outfield}
                    cfu.raster_to_polygon_gdal(**in_dict)
                    
                    # Cull some unwanted data
                    shp_df = gpd.read_file(ghz_shp_fname)
                    shp_df = shp_df.loc[shp_df[ghz_outfield]>-5] # remove areas outside active model domain
                    # Assign new field
                    shp_df['type'] = ''
                    for itype in list(cell_types.keys()):
                        shp_df.loc[shp_df[ghz_outfield]==itype,'type'] = cell_types[itype]
        #            shp_df['areakm2'] = shp_df.area/1e6 # m2 to km2    
                    shp_df.to_file(ghz_shp_fname,crs_wkt=active_proj)
                    
# Save density and datum elevation data
if not linear_resp_bool:                    
    out_df = pd.DataFrame(county_sl_data)
    out_csv = os.path.join(save_dir,'summary_sl_density_{}.csv'.format(swi_date))
    out_df.to_csv(out_csv)