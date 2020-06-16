#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:44:43 2018

@author: kbefus
"""

import sys,os
import numpy as np
import glob
import pandas as pd
#import dask.array as da
import rasterio

res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
sys.path.insert(1,code_dir)
#
#from cgw_model.cgw_utils import cgw_general_utils as cgu
#from cgw_model.cgw_utils import cgw_raster_utils as cru
#from cgw_model.cgw_utils import cgw_feature_utils as cfu
#
#import matplotlib as mpl
#mpl.rcParams['pdf.fonttype'] = 42
#import matplotlib.pyplot as plt
#plt.rc('legend',**{'fontsize':9})
#from matplotlib import gridspec
#from mpl_toolkits.axes_grid1 import make_axes_locatable


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

active_date = '4Nov19'

col_fmt = '{0}_lincount_sl{1:3.2f}_Kh{2:3.2f}'
flood_ind_fmt = '{0}_sl{1:3.2f}_Kh{2:3.2f}'
#%% 
dx = 0.05
#hist_bins = np.arange(-2,7.+dx,dx) # for dimensional
hist_bins = np.arange(-5.,1.2+dx,dx) # for non-dimensional

df=0.25
emerg_shoal_depths = np.hstack([-np.inf,marine_value,np.arange(0,5+df,df),np.inf])
shoal_ind_bins = np.arange(1,len(emerg_shoal_depths)+1)
flood_bins = [-np.inf,marine_value,0,1,2,5,np.inf]

col_area_fmt = 'area_km2_{0}mdepth'
flood_cols = ['MarineInundation']
flood_cols.extend([col_area_fmt.format(d1) for d1 in emerg_shoal_depths[2:]])

#%%
#ca_regions = ['norca','sfbay','soca','paca','cenca'] # order to assign cells
#region_fname = os.path.join(data_dir,'masked_wt_depths','CA_domains_14Nov18.tif')
#region_array = xr.open_rasterio(region_fname,chunks=(1e3,1e3),parse_coordinates=False).data
#%%
sumeveryxrows=lambda array,x: array.reshape((int(array.shape[0]/x),x)).sum(1)
#n_orig_cell_list = []
all_floodmats = {}
for model_type in model_types:
    datum_type = model_type.split('_')[1].upper()
    scenario_type = '_'.join(model_type.split('_')[1:])
    
    wt_dir = os.path.join(output_dir,model_type,'wt')
        
    county_dirs = [idir for idir in glob.glob(os.path.join(wt_dir,'*')) if os.path.isdir(idir)]
    f_dfs=[]
    for Kh in Kh_vals:    
#        print('------------ Kh = {} ---------------'.format(Kh))
        kh_dir = 'Kh{0:3.2f}mday'.format(Kh)
        kh_dir=kh_dir.replace('.','p')
        
        out_hist_data = []
        out_hist_cols = []
        flood_data = []
        flood_inds = []
        
        for county_dir in county_dirs:
            county_name = os.path.basename(county_dir)
#            print('------- {} --------'.format(county_name))
            for sl in sealevel_elevs[1:]:
#                print('--- SL = {} ----'.format(sl))

                # ---------- Calculate difference between model and linear response ----------            
                # Load linear response wt
                if scenario_type in ['mhhw']:
                    tempname =file_fmt.format(county_name,'linresponse','wt',Kh,sl)
                else:
                    tempname = file_fmt.format(county_name,'linresponse_wt',scenario_type,Kh,sl)
                tempname = tempname.replace('.','p')
                linwt_fname = os.path.join(county_dir,'linresponse_{}'.format(kh_dir),'{}.tif'.format(tempname))

                with rasterio.open(linwt_fname) as src:
                    shifted_wt = src.read()[0]
                    shifted_wt[shifted_wt==src.nodata] = np.nan
                with np.errstate(invalid='ignore'):
                    shifted_wt[(shifted_wt<0) & (shifted_wt!=marine_value)] = 0. # all wt<0 depth set to 0 = emergent gw
                
                # --------- Calculate area of flooding and emergent gw ---------------
                
                # Calculate area of marine inundation, gw shoaling, and emergence
                binned_wt_depths = np.digitize(shifted_wt,bins=emerg_shoal_depths,
                                               right=True)
                binned_wt_depths = binned_wt_depths[~np.isnan(shifted_wt)]
                bin_count,edges = np.histogram(binned_wt_depths,bins=shoal_ind_bins)    
                flood_data.append(bin_count)
                flood_inds.append(flood_ind_fmt.format(county_name,sl,Kh))
    
        # Save flooding outputs
        flooding_area_km2 = np.array(flood_data)*(cell_spacing**2)/1e6 # count to m**2 to km**2
        flood_df = pd.DataFrame(flooding_area_km2,index=flood_inds,columns=flood_cols)
        flood_df.index.name = 'model'
        flood_df['model']=flood_df.index.values
        
        flood_fname = os.path.join(results_dir,'SLR_flood_area_linresponse_{0}_bycounty_Kh{1:4.2f}mday_{2}.csv'.format(scenario_type,Kh,active_date))
        flood_df.to_csv(flood_fname,index_label='model')
        f_dfs.append(flood_df)

    all_f_indexes = np.array([tempdf.index.values for tempdf in f_dfs]).ravel()
    f_df = pd.concat(f_dfs,axis=0,ignore_index=True)
#    f_df.set_index('model',drop=False,inplace=True)
    f_df.index=all_f_indexes

    nbins = f_df.shape[1]
#%%
    for i,Kh in enumerate(Kh_vals):
        # Collect all sl data for Kh run
        Kh_inds = []
        for j,sl in enumerate(sealevel_elevs[1:]):
            for ireg,county_dir in enumerate(county_dirs):
                ca_county = os.path.basename(county_dir)
                Kh_inds.append(flood_ind_fmt.format(ca_county,sl,Kh))
            
    #    total_model_area =  f_df.loc[Kh_inds].sum(axis=1).values[0]
        flood_mat = np.cumsum(f_df.loc[Kh_inds].values.copy()[:,1:-1][:,::-1],axis=1)[:,::-1]
        
        flood_mat_all = np.array([sumeveryxrows(flood_mat[:,i],len(county_dirs)) for i in range(flood_mat.shape[1])]).T
        #flood_mat[:,0] = flood_mat[:,0] - flood_mat[0,0] # marine inundation from present
        #flood_mat[:,-1] = flood_mat[:,-1] - flood_mat[0,-1] # and for > 5 m
        
        all_floodmats.update({scenario_type:{Kh:flood_mat_all.T}})
        
        
        # find area and % of modern and % of active for <= 2 m
        present_active_area = flood_mat_all[0,0]
        present_lt2m_area = flood_mat_all[0,1]-flood_mat_all[0,3] # cumulative, so difference with next lowest gives area
        sl1m_lt2m_area = flood_mat_all[3,1]-flood_mat_all[3,3] # remove 1 from sea level elev indexes b/c starting at 0.25
        print('-----{}, Kh={} mday-------'.format(scenario_type,Kh))
        print('Active_area = {0:3.2f}; <2m area present-day = {1:3.2f}; SLR1m <2m area = {2:3.2f}; diff in 2m area = {3:3.2f} km2'.format(present_active_area,present_lt2m_area,sl1m_lt2m_area,present_lt2m_area-sl1m_lt2m_area))
        print('<2m 1m SLR: {0:3.2f}% active, {1:3.2f}% present day'.format(1e2*sl1m_lt2m_area/present_active_area,
                                                                          1e2*sl1m_lt2m_area/present_lt2m_area))
        
        
        
        # save flood mat
        f_file = os.path.join(results_dir,'flood_areas_allCA_linresponse_{0}_{1:2.1f}mday_{2}.csv'.format(scenario_type,Kh,active_date))
        out_temp_df = pd.DataFrame(flood_mat_all,columns=f_df.columns.values[1:-1],index=sealevel_elevs[1:])
        out_temp_df.to_csv(f_file,index=True,index_label='sea_level_m')
    
#%%
        

        
