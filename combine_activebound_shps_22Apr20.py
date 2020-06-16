#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 12:55:56 2019

@author: kbefus

Combine all 
"""

import os
import glob
import geopandas as gpd

#%%
research_dir = os.path.join(r'/mnt/data2/CloudStation','ca_slr')
model_dir = os.path.join(r'/mnt/762D83B545968C9F/model_lmsl_noghb')
gis_dir = os.path.join(research_dir,'data','gis')
domain_dirs = glob.glob(os.path.join(model_dir,'*'))
domain_dirs = [d for d in domain_dirs if 'fig' not in d]

utm10n = 3717
utm11n = 3718
ncrs =  {'init': 'epsg:{}'.format(utm10n)}
scrs = {'init': 'epsg:{}'.format(utm11n)}

inactive_val = 0
all_dfs = []
for domain_dir in domain_dirs:
    activebound_dir = os.path.join(domain_dir,'active_bounds')
    activebound_shps = glob.glob(os.path.join(activebound_dir,'*slr0.00m_Kh1.0*.shp'))
    activebound_shps.sort()
    for active_shp in activebound_shps:
        temp_df = gpd.read_file(active_shp)
        if not hasattr(temp_df,'crs'):
            if 'soca' in active_shp:
                temp_df.crs = scrs
            else:
                temp_df.crs = ncrs
        # Transform soca to UTM 10
        if 'soca' in active_shp:
            temp_df = temp_df.to_crs(ncrs)
        # Drop inactive features
        temp_df = temp_df[temp_df['ID']!=0].copy()
        # Save model name as column
        temp_df['model'] = '_'.join(os.path.basename(active_shp).split('_')[:2])
        all_dfs.append(temp_df)

final_df = gpd.pd.concat(all_dfs,axis=0,ignore_index=True)

#%%
outfname = os.path.join(gis_dir,'active_bounds_lsml_slr0.00m.shp')
final_df.to_file(outfname)
#%%
# Save merged by type    
dissolve_df = final_df.dissolve(by='ID')
outfname2 = os.path.join(gis_dir,'dissolved_active_bounds_lsml_slr0.00m.shp')
dissolve_df.to_file(outfname2)