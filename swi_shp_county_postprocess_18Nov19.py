#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:01:57 2019

@author: kbefus
"""
import os,sys
import glob
import numpy as np
import geopandas as gpd
from shapely.ops import unary_union
from shapely import speedups
speedups.enable()
res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
sys.path.insert(1,code_dir)

from cgw_model.cgw_utils import cgw_general_utils as cgu

#%%

research_dir_main = os.path.join(res_dir,'ca_slr')
data_dir_orig = os.path.join(research_dir_main,'data')
swi_date = '4Nov19'
swi_dir = os.path.join(data_dir_orig,'swi_outputs_{}'.format(swi_date),
                       'county_data_{}'.format(swi_date))

analytical_types = ['ghybenherzberg']
model_types = ['model_lmsl_noghb','model_mhhw_noghb']

sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m
Kh_vals = [0.1,1.,10.]
dirname_fmt = 'slr_{0:3.2f}_m_Kh{1:3.1f}mday'
cell_spacing=10
utm10n = 3717
utm11n = 3718
ncrs =  {'init': 'epsg:{}'.format(utm10n)}
scrs = {'init': 'epsg:{}'.format(utm11n)}
crs_dict = {'norca':ncrs,'paca':ncrs,'sfbay':ncrs,
            'cenca':ncrs,'soca':scrs}

marine_type = 'marine/tidal'
swi_type = 'swi_footprint'

out_fname = os.path.join(swi_dir,'summary','swi_areas_{}.csv'.format(swi_date))
if os.path.isfile(out_fname):
    out_df = gpd.pd.read_csv(out_fname)

all_footprints = []
to_sealevel=1
save_footprint_combo=False

county_fname = os.path.join(data_dir_orig,'gis','CA_coastal_Counties_TIGER2016_UTM10N.shp')
ind_joined_df = gpd.read_file(county_fname)
ind_joined_df.crs = ncrs

#%%

#linear_resp_bool = False
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
            
            sl_area_list = [] 
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
    
                        all_marinetidal_shps.append(idf.loc[idf['type']==marine_type,:].copy())
                        all_swi_shps.append(idf.loc[idf['type']==swi_type,:].copy())
            
                    all_marinetidal_df = gpd.pd.concat(all_marinetidal_shps).pipe(gpd.GeoDataFrame).buffer(1e-3,cap_style=3).unary_union#.dissolve(by='type')
                    all_swi_df = gpd.pd.concat(all_swi_shps).pipe(gpd.GeoDataFrame).buffer(1e-3,cap_style=3).unary_union
                    area_swi = all_swi_df.area/1e6 # m2 to km2
                    area_marine_tidal = all_marinetidal_df.area/1e6 # m2 to km2
                    sl_area_list.append([analytical_type,scenario_type,Kh,sl,area_marine_tidal,area_swi])
                
                    
                    if save_footprint_combo:
                        if sl<=to_sealevel:
                            all_footprints.extend(all_swi_shps)
        #                    all_footprints.append(unary_union([ipoly.buffer(0).unary_union for ipoly in all_swi_shps]))
                            
            
                if save_footprint_combo:
                    new_footprints = []
                    for iprint in range(len(all_footprints)):
                        new_footprints.extend(all_footprints[iprint].buffer(1e-3,cap_style=3).geometry.values)                       
        
                    #df1 = gpd.GeoDataFrame(['swi_footprint']*len(all_footprints),geometry=np.array(all_footprints))
                    allshps2 = unary_union(new_footprints)
                    all_swi_df = gpd.GeoDataFrame(geometry=[allshps2])
                    all_swi_df['type'] = 'swi_footprint'
                    all_swi_df['model']= 'Kh1mday_1mSLR_{}'.format(scenario_type.upper())
                    all_swi_fname = os.path.join(swi_dir,'EkstromShps','AllCA_swifootprint_gh_neg50mNAVD88_{0}_slr1m_Kh{1:3.0f}_{2:.0f}m.shp'.format(scenario_type,Kh,cell_spacing))
                    all_swi_df.to_file(all_swi_fname,crs_dict=ncrs)

            if len(sl_area_list)>0:
            
                temp_df = gpd.pd.DataFrame(sl_area_list,columns=['Type','Scenario','Kh_mday','SeaLevel_m','MarineArea_km2','SWIArea_km2']) 
                
                if not os.path.isfile(out_fname):            
                    out_df = temp_df.copy()
                else:
                    out_df = gpd.pd.read_csv(out_fname)
                    out_df = gpd.pd.concat([out_df,temp_df],ignore_index=True,sort=True)
                
                try:    
                    out_df.to_csv(out_fname,index=False)         
                except OSError as e:
                    print('{}'.format(e))
                    i=0
                    maxi=10
                    imnotout=True
                    while imnotout:
                        new_out_fname = os.path.join('_'.join([os.path.splitext(out_fname)[0],'v{}.csv'.format(i)]))
                        if os.path.isfile(new_out_fname):
                            i+=1
                            if i>maxi:
                                imnotout=False
                        else:
                            out_df.to_csv(new_out_fname,index=False) 
                            imnotout = False
            