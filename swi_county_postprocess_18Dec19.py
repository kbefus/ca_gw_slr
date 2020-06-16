#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:31:02 2019

@author: kbefus
"""

import sys,os
import numpy as np
import time
import glob

from shapely.ops import unary_union
from shapely import speedups
speedups.enable()

res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
#code_dir = r'C:\Users\kbefus\OneDrive - University of Wyoming\ca_slr\scripts'
sys.path.insert(1,code_dir)

from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_utils import cgw_raster_utils as cru

import geopandas as gpd
import pandas as pd

#%%
research_dir_main = os.path.join(res_dir,'ca_slr')
data_dir_orig = os.path.join(research_dir_main,'data')
swi_date = '4Nov19'
swi_dir = os.path.join(data_dir_orig,'swi_outputs_{}'.format(swi_date),
                       'county_data_{}'.format(swi_date))

analytical_types = ['ghybenherzberg']
model_types = ['model_lmsl_noghb','model_mhhw_noghb']

id_col = 'Id'
sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m
cell_spacing = 10. # meters
marine_mask_val = -500.
Kh_vals = [0.1,1.,10.]
utm10n = 3717
utm11n = 3718
ncrs =  {'init': 'epsg:{}'.format(utm10n)}
scrs = {'init': 'epsg:{}'.format(utm11n)}
crs_dict = {'norca':ncrs,'paca':ncrs,'sfbay':ncrs,
            'cenca':ncrs,'soca':scrs}
nodata_val = -9999
ncrs =  {'init': 'epsg:{}'.format(utm10n)}
scrs = {'init': 'epsg:{}'.format(utm11n)}
crs_dict = {'norca':ncrs,'paca':ncrs,'sfbay':ncrs,
            'cenca':ncrs,'soca':scrs}
swi_type = 'swi_footprint'
marine_type = 'marine/tidal'

wt_fmt = '{}*wtdepth.tif'
head_fmt = '{}*head.tif'
dirname_fmt = 'slr_{0:3.2f}_m_Kh{1:3.1f}mday'
output_fmt = '{0}_{1}_{2}_Kh{3:3.2f}_slr{4:3.2f}m'

county_fname = os.path.join(data_dir_orig,'gis','CA_coastal_Counties_TIGER2016_UTM10N.shp')
ind_joined_df = gpd.read_file(county_fname)
ind_joined_df.crs = ncrs



#%%

icount=0
icount2=0
sl_area_list = []
for linear_resp_bool in [False,True]:
    for analytical_type in analytical_types:
        if analytical_type =='ghybenherzberg':
            at1 = 'GH'
        
        for model_type in model_types:
            print('---------- {} --------------'.format(model_type))
            datum_type = model_type.split('_')[1].upper()
            scenario_type = '_'.join(model_type.split('_')[1:])
            
            if linear_resp_bool:
                scenario_type = '_'.join(['linresponse',scenario_type])

            for Kh in Kh_vals:
#                if not linear_resp_bool and Kh==1.0:
#                    save_footprint_combo = True
                   
                for sl in sealevel_elevs:
                    if linear_resp_bool and sl==0:
                        continue
                    print('--- SL = {} ----'.format(sl))
        #            if save_footprint_combo:
        #                if sl<=to_sealevel:
        #                    pass
        #                else:
        #                    break
                    all_marinetidal_shps=[]
                    all_swi_shps = []
                    for ind,county_df in ind_joined_df.iterrows():
                        county_name = county_df['NAME'].replace(' ','_')
                        
                        if county_name == 'San_Benito':
                            continue
                        print('---------- {} ----------'.format(county_name))
                        shpdir = os.path.join(swi_dir,county_name)
                        tempname = '{0}_{1}_Kh{2:3.2f}_slr{3:3.2f}m_{4}_swi'.format(county_name,scenario_type,Kh,sl,at1)
                        tempname = tempname.replace('.','p')
                        shp_fname = os.path.join(shpdir,'{}.shp'.format(tempname))
                        
                        idf = gpd.read_file(shp_fname)
                        if idf.crs['init'] != ncrs['init']:
                            # Reproject to UTM10N
                            idf = idf.to_crs(ncrs)
            
                        all_marinetidal_df = idf.loc[idf['type']==marine_type,:].buffer(1e-3,cap_style=3).unary_union#.dissolve(by='type')
                        all_swi_df = idf.loc[idf['type']==swi_type,:].buffer(1e-3,cap_style=3).unary_union
                        area_swi = all_swi_df.area/1e6 # m2 to km2
                        area_marine_tidal = all_marinetidal_df.area/1e6 # m2 to km2
                        
                        sl_area_list.append([county_name,analytical_type,scenario_type,Kh,sl,area_marine_tidal,area_swi])


out_fname = os.path.join(swi_dir,'summary','swi_county_areas_{}.csv'.format(swi_date))
temp_df = gpd.pd.DataFrame(sl_area_list,columns=['County','Type','Datum','Kh_mday','SeaLevel_m','MarineArea_km2','SWIArea_km2']) 
temp_df.to_csv(out_fname,index=False)  