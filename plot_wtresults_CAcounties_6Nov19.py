#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 08:09:22 2018

Plot histograms of r- and topo-limited responses
Run after linear_wt_CAcounties*.py

@author: kbefus
"""

import os
import glob
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ytick.right'] = mpl.rcParams['xtick.top'] = True
from matplotlib import cm,colors
import matplotlib.pyplot as plt
plt.rc('legend',**{'fontsize':9})


#%%

res_dir = r'/mnt/data2/CloudStation'
research_dir_orig = os.path.join(res_dir,'ca_slr')
data_dir_orig = os.path.join(research_dir_orig,'data')
research_dir = r'/mnt/762D83B545968C9F'
output_dir = os.path.join(research_dir,'data','outputs_fill_gdal_29Oct19')

results_dir = os.path.join(research_dir,'results','no_ghb','wt_analysis')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

model_types = ['model_lmsl_noghb','model_mhhw_noghb']

Kh_vals = [0.1,1.,10.]
active_date = '5Aug19'
cell_spacing = 10. # meters

col_fmt = '{0}_lincount_sl{1:3.2f}_Kh{2:3.2f}'
flood_ind_fmt = '{0}_sl{1:3.2f}_Kh{2:3.2f}'

for model_type in model_types:
    datum_type = '_'.join(model_type.split('_')[1:])
    
    wt_dir = os.path.join(output_dir,model_type,'wt')
        
    county_dirs = [idir for idir in glob.glob(os.path.join(wt_dir,'*')) if os.path.isdir(idir)]


    flood_fmt = 'SLR_flood_area_{0}_bycounty_Kh{1:4.2f}mday_{2}.csv'
    mvl_fmt = 'Model_vs_Linear_wt_response_nondim_{0}_bycounty_Kh{1:4.2f}mday_{2}.csv'
    f_dfs = []
    l_dfs = []
    for Kh in Kh_vals:
        flood_fname = os.path.join(results_dir,flood_fmt.format(datum_type,Kh,active_date))
        lin_fname = os.path.join(results_dir,mvl_fmt.format(datum_type,Kh,active_date))
        f_df = pd.read_csv(flood_fname)
        f_df.set_index('model',inplace=True)
        l_df = pd.read_csv(lin_fname)
        l_dfs.append(l_df)
        f_dfs.append(f_df)
    
    all_l_cols = np.array([tempdf.columns for tempdf in l_dfs]).ravel()
    l_df = pd.concat(l_dfs,axis=1,ignore_index=True)
    l_df.columns = all_l_cols
    l_df = l_df.T.drop_duplicates(keep='first').T.copy()
    all_f_indexes = np.array([tempdf.index.values for tempdf in f_dfs]).ravel()
    f_df = pd.concat(f_dfs,axis=0,ignore_index=True)
    f_df.index=all_f_indexes
    
    
    
    #%% Plot histograms of linear vs modeled wt response to slr
    
    # Parse column names
    col_names = l_df.columns
    x_cols = [i for i in col_names if 'bin' in i]
    y_cols = [i for i in col_names if 'lincount' in i]
    sealevels = np.unique([float(i.split('sl')[1][:4]) for i in y_cols])
    if 0. not in sealevels:
        sealevels = np.hstack([0,sealevels])
    Kh_vals = np.unique([float(i.split('Kh')[1]) for i in y_cols])
    plt.close('all')
    
    cvals = plt.cm.viridis(np.arange(sealevels.shape[0])/(sealevels.shape[0]))
    X = np.array([l_df[x_cols[0]],l_df[x_cols[1]]]).T.flatten()
    
    # Figure for whole coast, diff K scenarios
    fig,ax = plt.subplots(2,2,sharey=True)
    ax_whole = ax.ravel()
    
    reg_figs,reg_axs = [],[]
    for ireg in enumerate(county_dirs):
        figtemp,axtemp = plt.subplots(2,2,sharey=True)
        reg_figs.append(figtemp)
        reg_axs.append(axtemp.ravel())
    
    store_all_p = []
    store_all_area = []
    for iK,Kh in enumerate(Kh_vals):
        axwhole_temp = ax_whole[iK]
        all_region_areas = []
        for ind1,sl in enumerate(sealevels):
            all_region_hits = np.zeros_like(X)
            p90_area_all = 0.
            p10_area_all = 0.
            for ireg,county_dir in enumerate(county_dirs):
                ca_county = os.path.basename(county_dir)
                if sl==0.:
                    # Calculate total area of active,non-marine model cells
                    noriginal_area = f_df.loc[flood_ind_fmt.format(ca_county,sl,Kh)].values[1:].sum()
                    all_region_areas.append(noriginal_area)
                else:
                    # Make histogram plot
                    col_temp = col_fmt.format(ca_county,sl,Kh)
                    
                    Y = np.array([l_df[col_temp],l_df[col_temp]]).T.flatten()*(cell_spacing**2)/1e6 # area, km2 per bin
                    all_region_hits += Y
                    axtemp = reg_axs[ireg][iK]
                    axtemp.plot(1e2*X,1e2*Y/noriginal_area,color=cvals[ind1,:],label='{}'.format(sl))
                    reg_figs[ireg].suptitle(ca_county)  
                    
                    # Regional axis options
                    axtemp.set_xlim([1e2*-.4,1e2*1.25])# nondim
                    axtemp.set_xlabel("Overprediction of water table rise by uniform response [%]")
                #    axtemp.set_xlim([-1,5.25])# dim
                #    axtemp.set_xlabel("Overprediction of water table rise by uniform response [m]")
                    axtemp.set_title("Kh = {0:3.2f}".format(Kh))     
                    axtemp.set_ylabel("% of present-day land")
                    axtemp.legend(title='SLR scenario [m]')
                    
                    # Print % area above 90% and below 10%
                    p90_area = l_df[l_df[x_cols[1]]>=.89][col_temp].sum()*(cell_spacing**2)/1e6 # km2
                    p10_area = l_df[l_df[x_cols[1]]<=.11][col_temp].sum()*(cell_spacing**2)/1e6 # km2
                    p90_area_all+= p90_area
                    p10_area_all+= p10_area
                    p90 = p90_area/noriginal_area
                    p10 = p10_area/noriginal_area
                    print("{0} | <10%: {1:4.2f} % | >90%: {2:4.2f} %".format(col_temp,1e2*p10,1e2*p90))
                    store_all_p.append([ca_county,Kh,sl,noriginal_area,p10_area,p10,p90_area,p90])
            
            # Plot whole CA coast results
            total_area = np.sum(all_region_areas)
    #        print(total_area)
            store_all_area.append(all_region_areas)
            axwhole_temp.plot(1e2*X,1e2*all_region_hits/total_area,color=cvals[ind1,:],label='{}'.format(sl))
            
            print("{0} | <10%: {1:4.2f} % | >90%: {2:4.2f} %".format('All_CA_{0:3.2f}'.format(Kh),1e2*p10_area_all/total_area,1e2*p90_area_all/total_area))
            print("---------------------------------------------------------------------")
            if sl>0.:
                store_all_p.append(['All',Kh,sl,total_area,p10_area_all,p10_area_all/total_area,p90_area_all,p90_area_all/total_area])
    
        # Whole coast axis options    
        axwhole_temp.set_xlim([1e2*-.4,1e2*1.25])
        axwhole_temp.set_xlabel("Overprediction of water table rise by uniform response [%]")
    #    axtemp.set_xlim([-1,5.25])# dim
    #    axtemp.set_xlabel("Overprediction of water table rise by uniform response [m]")
        axwhole_temp.set_title("Kh = {0:3.2f}".format(Kh))    
        axwhole_temp.set_ylabel("% of present-day land")
        axwhole_temp.legend(title='SLR scenario [m]')
        
    area_cols = ['name','Kh','sl','total_area','p10_area','p10_percent','p90_area','p90_percent']
    area_df = pd.DataFrame(store_all_p,columns=area_cols)
    
    allCA_area_df = area_df[area_df['name']=='All'].copy()
    allCA_area_df.to_csv(os.path.join(results_dir,'Model_vs_Linear_wt_response_allCA_{}_{}.csv'.format(datum_type,active_date)),
                                      index=False)
    
    #%% Plot areas flooded by marine inundation or w/ emergent gw
    plt.close('all')
    nbins = f_df.shape[1]
    df=0.25
    flood_vals = np.hstack([-1,np.arange(0,5+df,df),6])
    cmap = plt.cm.RdBu(flood_vals/(nbins))
    
    fig,ax = plt.subplots(2,2,sharey=True)
    ax = ax.ravel()
    
    fig2,ax2=plt.subplots()
    #ax2.set_title('Net area, sum($\Delta$WT area)-Marine Inundation')
    #ax3=ax2.twinx()
    
    for i,Kh in enumerate(Kh_vals):
        # Collect all sl data for Kh run
        Kh_inds = []
        for j,sl in enumerate(sealevels):
            for ireg,county_dir in enumerate(county_dirs):
                ca_county = os.path.basename(county_dir)
                Kh_inds.append(flood_ind_fmt.format(ca_county,sl,Kh))
        
        flood_mat = f_df.loc[Kh_inds].values
        flood_mat[:,0] = flood_mat[:,0] - flood_mat[0,0] # marine inundation from present
        flood_mat[:,-1] = flood_mat[:,-1] - flood_mat[0,-1] # and for > 5 m
        
        # Loop through flood vals (warning, this could change...)
        for iflood in range(len(flood_vals)):
            flood_val = flood_vals[iflood]
            icolor = cmap[iflood,:]
            
            if flood_val == -1:
                label_text = 'Marine Inundation'
                icolor = 'b'
            elif flood_val == 0:
                label_text = 'WT at surface'
            elif flood_val == 6:
                label_text = 'WT > 5 m'
            else:
                label_text = '{} < WT <= {} m'.format(flood_vals[iflood-1],flood_val)
                
            # for all regions summed
            ax[i].plot(sealevels,flood_mat[:,iflood].reshape((len(county_dirs),-1)).sum(axis=0),'.-',color=icolor,
                    label=label_text)
            
        ax[i].set_ylabel('Area [km$^2$]')
        ax[i].set_xlabel('Sea level above MHHW$_{present}$ [m]')    
        ax[i].set_title("Kh = {0:3.2f}".format(Kh))
        if i==3:
            ax[i].legend(loc='center left',bbox_to_anchor=[1.,0.5])
    
        # Plot difference bewteen marine inundation-gw shoal area
        delta_wt = np.diff(np.sum(flood_mat[:,1:-1],axis=1).reshape((len(county_dirs),-1)).sum(axis=0)) # don't include WTD > 5 m
        total_wt_area_from_present = np.sum(flood_mat[:,1:],axis=1).reshape((len(county_dirs),-1)).sum(axis=0)-\
                                    np.sum(flood_mat[0,1:])
        delta_mi = np.diff(flood_mat[:,0].reshape((len(county_dirs),-1)).sum(axis=0))
        ax2.plot(sealevels[1:],delta_wt/delta_mi,'--',label="{0} Kh = {1:3.2f}".format('$\Delta$GW$_{area}$/$\Delta$MI$_{area}$',Kh),color=cmap[i,:])
    #    ax2.plot(np.nan,1,'-.',color=cmap[i,:],label="Net GW-impacted area Kh = {0:3.2f}".format(Kh))
        ax2.legend(loc=1)
        ax2.set_xlabel('Sea level above MHHW$_{present}$ [m]')
        ax2.set_ylabel('$\Delta$GW$_{area}$/$\Delta$MI$_{area}$')
    #    ax3.plot(sealevels,total_wt_area_from_present,'.--',color=cmap[i,:],label="Net GW-impacted area Kh = {0:3.2f}".format(Kh))
        
    
    
    #%% New plot with filled areas
    plt.close('all')
    fig,ax = plt.subplots(2,2,sharey=True)
    ax = ax.ravel()
    cmap = plt.cm.RdBu(np.arange(nbins)/(nbins))
    #fig2,ax2=plt.subplots()
    #ax2.set_title('Net area, sum($\Delta$WT area)-Marine Inundation')
    #ax3=ax2.twinx()
    sumeveryxrows=lambda array,x: array.reshape((int(array.shape[0]/x),x)).sum(1)
    minz=5e3
    for i,Kh in enumerate(Kh_vals):
        # Collect all sl data for Kh run
        Kh_inds = []
        for j,sl in enumerate(sealevels):
            for ireg,county_dir in enumerate(county_dirs):
                ca_county = os.path.basename(county_dir)
                Kh_inds.append(flood_ind_fmt.format(ca_county,sl,Kh))
            
    #    total_model_area =  f_df.loc[Kh_inds].sum(axis=1).values[0]
        flood_mat = np.cumsum(f_df.loc[Kh_inds].values.copy()[:,1:][:,::-1],axis=1)[:,::-1]
        
        flood_mat_all = np.array([sumeveryxrows(flood_mat[:,i],len(county_dirs)) for i in range(flood_mat.shape[1])]).T
        #flood_mat[:,0] = flood_mat[:,0] - flood_mat[0,0] # marine inundation from present
        #flood_mat[:,-1] = flood_mat[:,-1] - flood_mat[0,-1] # and for > 5 m
        
        # save flood mat
        f_file = os.path.join(results_dir,'flood_areas_allCA_{0}_{1:2.1f}mday_{2}.csv'.format(datum_type,Kh,active_date))
        out_temp_df = pd.DataFrame(flood_mat_all,columns=f_df.columns.values[1:],index=sealevels)
        out_temp_df.to_csv(f_file,index=True,index_label='sea_level_m')
        
        
        # Loop through flood vals (warning, this could change...)
        for iflood in range(flood_mat_all.shape[1]-1):
            flood_val = flood_vals[iflood+1]
            icolor = cmap[iflood+1,:]
            
            if flood_val == -1:
                label_text = 'Marine Inundation'
                icolor = 'b'
            elif flood_val == 0:
                label_text = 'WT at surface'
    #            icolor = 'r'
            elif flood_val == 6:
                label_text = 'WT > 5 m'
            else:
                label_text = '{} < WT <= {} m'.format(flood_vals[iflood],flood_val)
                
            ax[i].set_title("Kh = {0:3.2f}".format(Kh))
            ax[i].fill_between(sealevels,flood_mat_all[:,iflood],y2=minz*np.ones_like(sealevels),facecolor=icolor,
                    label=label_text)
            ax[i].set_ylabel('Area [km$^2$]')
            ax[i].set_xlabel('Sea level above MHHW$_{present}$ [m]')
    
            if i==3:
                ax[i].legend(loc='center left',bbox_to_anchor=[1.,0.5])
            
    #    ax[i].set_yscale('log')
        ax[i].set_ylim([minz,9.5e3])
        ax[i].set_xlim([0,5])
    
    # make colormap
    N = colors.Normalize(vmin=flood_vals[1],vmax=flood_vals[-2])
    sm = cm.ScalarMappable(cmap=cm.RdBu,norm=N)
    sm.set_array([])
    fig.colorbar(sm,ax=ax.ravel()[1::2].tolist())
    plt.show()





