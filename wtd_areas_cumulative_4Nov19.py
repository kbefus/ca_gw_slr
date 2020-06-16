#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 09:55:30 2019

Calculate cumulative growth of wt_depth areas and marine/tidal w/ slr

@author: kbefus
"""

import os,sys
import glob
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree as KDTree
from shapely.ops import unary_union
from shapely import speedups
speedups.enable()
res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
sys.path.insert(1,code_dir)

from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_feature_utils as cfu
from shapely.ops import shared_paths,snap,linemerge

#%%

research_dir_main = os.path.join(res_dir,'ca_slr')
research_dir = r'/mnt/762D83B545968C9F'
data_dir = os.path.join(research_dir,'data')

active_date = '29Oct19'
output_dir = 'outputs_fill_gdal_{}'.format(active_date)
shp_dir = os.path.join(data_dir,output_dir)



model_types = ['model_lmsl_noghb','model_mhhw_noghb']

sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m
Kh_vals = [0.1,1.,10.]
dirname_fmt = 'slr_{0:3.2f}_m_Kh{1:3.1f}mday'

utm10n = 3717
utm11n = 3718
cell_spacing = 10

ncrs =  {'init': 'epsg:{}'.format(utm10n)}
scrs = {'init': 'epsg:{}'.format(utm11n)}
crs_dict = {'norca':ncrs,'paca':ncrs,'sfbay':ncrs,
            'cenca':ncrs,'soca':scrs}

marine_type = 'marine/tidal'
swi_type = 'swi_footprint'
fresh_type = 'fresh_footprint'
feature_types = [fresh_type,swi_type,marine_type]

all_footprints = []
min_area = 1 # smaller than 1e2 takes all


overwrite_bool = False
save_dir = os.path.join(research_dir_main,'results','no_ghb','wt_analysis')
sv_fname = os.path.join(save_dir,'wtdepth_areas_bycounty_6Dec19.csv')


#linear_resp_bool = True
out_list = []

wt_col = 'wtdepth'
for linear_resp_bool in [False, True]:
    for model_type in model_types:
        datum_type = model_type.split('_')[1].upper()
        scenario_type = '_'.join(model_type.split('_')[1:])
        print('---------- {} --------------'.format(scenario_type))
        county_dirs = glob.glob(os.path.join(shp_dir,model_type,'shp','*'))
        county_dirs = [idir for idir in county_dirs if os.path.isdir(idir) and 'Error' not in idir]
        for Kh in Kh_vals:
            print('------------ Kh = {} ---------------'.format(Kh))
            kh_dir = 'Kh{0:3.2f}mday'.format(Kh)
            kh_dir=kh_dir.replace('.','p')
            
            for cdir in county_dirs:
                county_name = os.path.basename(cdir)
                print('----------- {} -------------'.format(county_name))
                county_merged = {}
                for sl in sealevel_elevs:
                    print('--- SL = {} ----'.format(sl))
                    if linear_resp_bool and sl==0:
                        # Load original, not lin, modeled output for sl=0
                        temp_fname = '{0}_{1}_slr{2:3.2f}m_Kh{3:3.2f}mday_emergent'.format(county_name,scenario_type,sl,Kh) 
                        temp_fname = temp_fname.replace('.','p')
                        shp_name = os.path.join(cdir,kh_dir,'{}.shp'.format(temp_fname))
                        shp_df = gpd.read_file(shp_name)
                        out_type = '_'.join([scenario_type,'linresp'])
                    else:
                        temp_fname = '{0}_{1}_slr{2:3.2f}m_Kh{3:3.2f}mday_emergent'.format(county_name,scenario_type,sl,Kh)
                        if linear_resp_bool:
                           kh_dir2 = '_'.join(['linresponse',kh_dir])
                           temp_fname = '{}_lin'.format(temp_fname)
                           out_type = '_'.join([scenario_type,'linresp'])
                        else:
                            kh_dir2 = kh_dir
                            out_type = scenario_type
                            
                        temp_fname = temp_fname.replace('.','p')
                        shp_name = os.path.join(cdir,kh_dir2,'{}.shp'.format(temp_fname))
                        shp_df = gpd.read_file(shp_name)
                        
                        
#                    unique_types = shp_df[wt_col].unique()
                    
                    shp_df['area_km2'] = shp_df.area/1e6
                    gdf = shp_df.groupby(['wtdepth'])['area_km2'].agg('sum')
                    unique_types = gdf.index.values.unique()
                    
                    for temp_type in unique_types:
                        # new way
                        out_list.append([out_type,Kh,county_name,sl,temp_type,gdf[temp_type]])
                        
                        
                        # old way
#                        temp_shp = shp_df[shp_df[wt_col]==temp_type].unary_union
#                        if temp_type not in list(county_merged.keys()):
#                            # Add key and merge features
#                            county_merged.update({temp_type:temp_shp})
#                                   
#                        else:
#                            county_merged[temp_type] = unary_union([county_merged[temp_type],temp_shp])
#                            
#                        out_list.append([out_type,Kh,county_name,sl,temp_type,county_merged[temp_type].area/1e6])
            
            sv_cols = ['Scenario','Kh_mday','County','Sea_level_m','Category','Area_km2']
            sv_df = gpd.pd.DataFrame(out_list,columns=sv_cols)
            if os.path.isfile(sv_fname) and not overwrite_bool:
                sv_df_orig = gpd.pd.read_csv(sv_fname)
                sv_df = gpd.pd.concat([sv_df_orig,sv_df],ignore_index=True)
            
            sv_df.to_csv(sv_fname,index=False)
