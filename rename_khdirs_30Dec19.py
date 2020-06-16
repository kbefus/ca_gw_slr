#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:28:23 2019

@author: kbefus

rename dirs

"""

import os
import geopandas as gpd
import numpy as np
#%%
res_dir = r'/mnt/data2/CloudStation'
research_dir_orig = os.path.join(res_dir,'ca_slr')
research_dir = r'/mnt/762D83B545968C9F'
data_dir = os.path.join(research_dir,'data')
data_dir_orig = os.path.join(research_dir_orig,'data')
active_date = '27Dec19'
out_dir = os.path.join(data_dir,'masked_wt_depths_gdal_{}'.format(active_date))
output_dir = 'outputs_fill_gdal_{}'.format(active_date)
dem_dir = os.path.join(data_dir_orig,'gw_dems11Feb19')

output_fmt = '{0}_{1}_{2}_Kh{3:3.1f}_slr{4:3.2f}m'

ftypes = ['head','wt','shp']
sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m

model_types = ['model_lmsl_noghb','model_mhhw_noghb']
Kh_vals = [0.1,1.,10.]

county_fname = os.path.join(data_dir_orig,'gis','CA_coastal_Counties_TIGER2016_UTM10N.shp')
ind_joined_df = gpd.read_file(county_fname)
county_dirs = ['_'.join(ival.split()) for ival in ind_joined_df['NAME'].values]

for county_dir in county_dirs:
    for model_type in model_types:
        print('---------- {} --------------'.format(model_type))
        datum_type = model_type.split('_')[1].upper()
        scenario_type = '_'.join(model_type.split('_')[1:])
        for Kh in Kh_vals:
            print('------------ Kh = {} ---------------'.format(Kh))
            kh_dir = 'Kh{0:3.2f}mday'.format(Kh)
            kh_dir_new = 'Kh{0:3.1f}mday'.format(Kh)
            kh_dir=kh_dir.replace('.','p')
            kh_dir_new=kh_dir_new.replace('.','p')
            for sealevel_elev in sealevel_elevs:
                print('--- SL = {} ----'.format(sealevel_elev))
                for ftype in ftypes:

                    out_dirtemp = os.path.join(data_dir,output_dir,model_type,ftype,county_dir,kh_dir)
                    new_outdir = os.path.join(data_dir,output_dir,model_type,ftype,county_dir,kh_dir_new)
                    if os.path.isdir(out_dirtemp):
                        print(new_outdir)
                        os.rename(out_dirtemp,new_outdir)
                    else:
                        print(out_dirtemp)
                    

