#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:44:43 2018

Compare model and linear response files
Run after CA_lineaWTresponse_County_*.py

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

active_date = '6Nov19'

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
small_neg = -1.
#n_orig_cell_list = []
for model_type in model_types:
    datum_type = model_type.split('_')[1].upper()
    scenario_type = '_'.join(model_type.split('_')[1:])
    
    wt_dir = os.path.join(output_dir,model_type,'wt')
        
    county_dirs = [idir for idir in glob.glob(os.path.join(wt_dir,'*')) if os.path.isdir(idir)]

    for Kh in Kh_vals:    
        print('------------ Kh = {} ---------------'.format(Kh))
        kh_dir = 'Kh{0:3.2f}mday'.format(Kh)
        kh_dir=kh_dir.replace('.','p')
        
        out_hist_data = []
        out_hist_cols = []
        flood_data = []
        flood_inds = []
        
        for county_dir in county_dirs:
            county_name = os.path.basename(county_dir)
            print('------- {} --------'.format(county_name))
            for sl in sealevel_elevs:
                print('--- SL = {} ----'.format(sl))
                # Load water table depth tifs
                tempname = file_fmt.format(county_name,'wt',scenario_type,Kh,sl)
                tempname = tempname.replace('.','p')
                wt_fname = os.path.join(county_dir,kh_dir,'{}.tif'.format(tempname))
    
                if sl==0.0:
                    with rasterio.open(wt_fname) as src:
                        wt_sl0 = src.read()[0]
                        wt_sl0[wt_sl0==src.nodata] - np.nan
                        with np.errstate(invalid='ignore'):
                            wt_sl0[(wt_sl0<0) & (wt_sl0!=marine_value)]=0 # set negative water tables to zero
                        
                    # Populate flood data for sl=present
                    binned_wt_depths = np.digitize(wt_sl0,bins=emerg_shoal_depths,
                                               right=True)
                    binned_wt_depths = binned_wt_depths[~np.isnan(wt_sl0)]
                    bin_count,edges = np.histogram(binned_wt_depths,bins=shoal_ind_bins)    
                        
                    flood_inds.append(flood_ind_fmt.format(county_name,0,Kh))
                    flood_data.append(bin_count)
                        
                    continue
                else:
                    with rasterio.open(wt_fname) as src:
                        wt_other = src.read()[0]
                        wt_other[wt_other==src.nodata] = np.nan
                        with np.errstate(invalid='ignore'):
                            wt_other[(wt_other<0) & (wt_other!=marine_value)] = 0 # set negative water tables to zero
    
                # Assign marine mask
                marine_mask = wt_other == marine_value
                
                # ---------- Calculate difference between model and linear response ----------            
                # Load linear response wt
                tempname = file_fmt.format(county_name,'linresponse_wt',scenario_type,Kh,sl)
                tempname = tempname.replace('.','p')
                linwt_fname = os.path.join(county_dir,'linresponse_{}'.format(kh_dir),'{}.tif'.format(tempname))
                with rasterio.open(linwt_fname) as src:
                    shifted_wt = src.read()[0]
                    shifted_wt[shifted_wt==src.nodata] = np.nan
                with np.errstate(invalid='ignore'):
                    shifted_wt[(shifted_wt<0) & (shifted_wt!=marine_value)] = 0. # all wt<0 depth set to 0 = emergent gw
                
                # Compare model with linear increase
                wt_diff = wt_other-shifted_wt
                wt_diff[marine_mask] = np.nan
    
                # Calculate histograms
        #        xbins,edges=da.histogram(wt_diff[~np.isnan(wt_diff)],bins=hist_bins) # dimensional
                xbins,edges=np.histogram(wt_diff[~np.isnan(wt_diff)]/sl,bins=hist_bins) # nondimensional
                
                if 'bin_left' not in out_hist_cols:
                    left,right = edges[:-1],edges[1:]
                    out_hist_cols.extend(['bin_left','bin_right'])
                    out_hist_data.extend([left,right])
                    
                out_hist_data.append(xbins)
                out_hist_cols.append(col_fmt.format(county_name,sl,Kh))
                # --------- Calculate area of flooding and emergent gw ---------------
                
                # Calculate area of marine inundation, gw shoaling, and emergence
                binned_wt_depths = np.digitize(wt_other,bins=emerg_shoal_depths,
                                               right=True)
                binned_wt_depths = binned_wt_depths[~np.isnan(wt_other)]
                bin_count,edges = np.histogram(binned_wt_depths,bins=shoal_ind_bins)    
                flood_data.append(bin_count)
                flood_inds.append(flood_ind_fmt.format(county_name,sl,Kh))
    
            # Save flooding outputs
            flooding_area_km2 = np.array(flood_data)*(cell_spacing**2)/1e6 # count to m**2 to km**2
            flood_df = pd.DataFrame(flooding_area_km2,index=flood_inds,columns=flood_cols)
            
            flood_fname = os.path.join(results_dir,'SLR_flood_area_{0}_bycounty_Kh{1:4.2f}mday_{2}.csv'.format(scenario_type,Kh,active_date))
            flood_df.to_csv(flood_fname,index_label='model')
                
            # Save linear difference outputs
            lin_df = pd.DataFrame(np.array(out_hist_data).T,columns=out_hist_cols)
            lin_fname = os.path.join(results_dir,'Model_vs_Linear_wt_response_nondim_{0}_bycounty_Kh{1:4.2f}mday_{2}.csv'.format(scenario_type,Kh,active_date))
            lin_df.to_csv(lin_fname,index_label='type')
    
          
