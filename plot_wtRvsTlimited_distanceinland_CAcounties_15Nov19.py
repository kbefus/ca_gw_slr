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
out_fig_dir = os.path.join(results_dir,'figs2')
for idir in [results_dir,out_fig_dir]:
    if not os.path.isdir(idir):
        os.makedirs(idir)

model_types = ['model_lmsl_noghb','model_mhhw_noghb']

Kh_vals = [0.1,1.,10.]
lowval = -5
active_date = '6Nov19'
cell_spacing = 10. # meters
R_limit = 0.06
T_limit = 0.09
x_dims = np.array([-.25,1.25])*100

col_fmt = '{0}_lincount_sl{1:3.2f}_Kh{2:3.2f}_inland{3}m'
flood_ind_fmt = '{0}_sl{1:3.2f}_Kh{2:3.2f}'

for model_type in model_types[:1]:
    datum_type = '_'.join(model_type.split('_')[1:])
    
    wt_dir = os.path.join(output_dir,model_type,'wt')
        
    county_dirs = [idir for idir in glob.glob(os.path.join(wt_dir,'*')) if os.path.isdir(idir)]


    flood_fmt = 'SLR_flood_area_{0}_bycounty_Kh{1:4.2f}mday_{2}.csv'
    mvl_fmt = 'Model_vs_Linear_wt_response_inlanddist_nondim_{0}_bycounty_Kh{1:4.2f}mday_{2}.csv'
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
    
    # Plot histograms of linear vs modeled wt response to slr
    
    # Parse column names
    col_names = l_df.columns
    x_cols = [i for i in col_names if 'bin' in i]
    y_cols = [i for i in col_names if 'lincount' in i]
    sealevels = np.unique([float(i.split('sl')[1][:4]) for i in y_cols])
    if 0. not in sealevels:
        sealevels = np.hstack([0,sealevels])
    Kh_vals = np.unique([float(i.split('Kh')[1].split('_')[0]) for i in y_cols])
    inland_dists = np.unique([float(i.split('inland')[1][:-1]) for i in y_cols])
    plt.close('all')
    
    cvals = plt.cm.viridis(np.arange(sealevels.shape[0])/(sealevels.shape[0]))
    X = np.array([l_df[x_cols[0]],l_df[x_cols[1]]]).T.flatten()
    X[np.isinf(X)] = -10
    
    store_all_p = []
    store_all_area = []
    #%%
    # Figure for whole coast, diff K scenarios
    fig,ax_whole = plt.subplots(len(inland_dists),3,sharey=True,figsize=(11,8.5))
    
    reg_figs,reg_axs = [],[]
    for ireg in enumerate(county_dirs):
        figtemp,axtemp = plt.subplots(len(inland_dists),3,sharey=True,figsize=(11,8.5))
        reg_figs.append(figtemp)
        reg_axs.append(axtemp)
    
    for idist,inland_dist in enumerate(inland_dists): 

        for iK,Kh in enumerate(Kh_vals):
            axwhole_temp = ax_whole[idist,iK]
            all_region_areas = []
            sl_plots = []
            for ind1,sl in enumerate(sealevels):
                all_region_hits = np.zeros_like(X)
                Tlim_area_all = 0.
                Rlim_area_all = 0.
                emerg_area_all = 0.
                ca_areas_analyzed = 0.
                reg_plots = []
                for ireg,county_dir in enumerate(county_dirs):
                    ca_county = os.path.basename(county_dir)
                    if sl==0.:
                        # Calculate total area of active,non-marine model cells
                        noriginal_area = f_df.loc[flood_ind_fmt.format(ca_county,sl,Kh)].values[1:].sum()
                        all_region_areas.append(noriginal_area)
                    else:
                        # Make histogram plot
                        col_temp = col_fmt.format(ca_county,sl,Kh,int(inland_dist))
                        
                        Y = np.array([l_df[col_temp],l_df[col_temp]]).T.flatten()*(cell_spacing**2)/1e6 # area, km2 per bin
                        total_area_analyzed = l_df[col_temp].sum()*(cell_spacing**2)/1e6
                        all_region_hits += Y
                        axtemp = reg_axs[ireg][idist,iK]
                        iplot = axtemp.plot(1e2*X,1e2*Y/total_area_analyzed,color=cvals[ind1,:],label='{}'.format(sl))
                          
                        
                        # Regional axis options
#                        axtemp.set_xlim([1e2*-.4,1e2*1.25])# nondim
                        axtemp.set_xlabel("Overprediction of water table rise by uniform response [%]")
                        axtemp.set_xlim(x_dims)# dim
                    #    axtemp.set_xlabel("Overprediction of water table rise by uniform response [m]")
                        axtemp.set_title("Kh = {0:3.2f}; Dinland = {1} m".format(Kh,inland_dist))     
                        axtemp.set_ylabel("% of present-day land")
                        axtemp.grid()
                        
                        if Kh == Kh_vals[-1] and inland_dist == inland_dists[-1]:
                            axtemp.legend(title='SLR scenario [m]')
                        
#                        reg_plots.extend(iplot)
                        
                        # Print % area above T_limit and below R_limit
                        Tlim_area = l_df[l_df[x_cols[1]]>=T_limit][col_temp].sum()*(cell_spacing**2)/1e6 # km2
                        Rlim_area = l_df[(l_df[x_cols[1]]<=R_limit) & (l_df[x_cols[0]]>lowval)][col_temp].sum()*(cell_spacing**2)/1e6 # km2
                        emerg_area = l_df[(l_df[x_cols[1]]<=lowval)][col_temp].sum()*(cell_spacing**2)/1e6 # km2
                        
                        emerg_area_all+=emerg_area
                        Tlim_area_all+= Tlim_area
                        Rlim_area_all+= Rlim_area
                        Tlim = Tlim_area/total_area_analyzed
                        Rlim = Rlim_area/total_area_analyzed
                        emerg = emerg_area/total_area_analyzed
                        print("{0} | <{1}%: {2:4.2f}% | >{3}%: {4:4.2f}% | emerg: {5:4.2f}%".format(col_temp,R_limit,1e2*Rlim,T_limit,1e2*Tlim,1e2*emerg))
                        store_all_p.append([ca_county,datum_type,Kh,sl,inland_dist,total_area_analyzed,Rlim_area,Rlim,Tlim_area,Tlim,emerg_area,emerg])
                        ca_areas_analyzed +=total_area_analyzed
                        
                
                # Plot whole CA coast results
#                total_area = np.sum(all_region_areas)
        #        print(total_area)
                if sl>0.:
#                    sl_plots.append(np.array(reg_plots))
                    
                    store_all_area.append(ca_areas_analyzed)
                    axwhole_temp.plot(1e2*X,1e2*all_region_hits/ca_areas_analyzed,color=cvals[ind1,:],label='{}'.format(sl))
                    axwhole_temp.set_xlabel("Overprediction of water table rise by uniform response [%]")
                    axwhole_temp.set_xlim(x_dims)# dim
                    axwhole_temp.set_ylim((0,40))
                #    axtemp.set_xlabel("Overprediction of water table rise by uniform response [m]")
                    axwhole_temp.set_title("Kh = {0:3.2f}".format(Kh))    
                    axwhole_temp.set_ylabel("% of present-day land")
                    axwhole_temp.grid()
                    print("{0} | <{1}%: {2:4.2f}% | >{3}%: {4:4.2f}% | emerg: {5:4.2f}%".format('All_CA_{0:3.2f}'.format(Kh),R_limit,1e2*Rlim_area_all/ca_areas_analyzed,T_limit,1e2*Tlim_area_all/ca_areas_analyzed,1e2*emerg_area_all/ca_areas_analyzed))
                    print("---------------------------------------------------------------------")
                    store_all_p.append(['All',datum_type,Kh,sl,inland_dist,ca_areas_analyzed,Rlim_area_all,Rlim_area_all/ca_areas_analyzed,Tlim_area_all,Tlim_area_all/ca_areas_analyzed,emerg_area_all,emerg_area_all/ca_areas_analyzed])
        
        
    # Save and close county-based figures
    for ireg,county_dir in enumerate(county_dirs):
        ca_county = os.path.basename(county_dir)
        col_temp = '{0}_{1}_inland{2}m'.format(ca_county,Kh,datum_type,int(inland_dist))
        out_fig = os.path.join(out_fig_dir,'{}.png'.format(col_temp))
        
#            axtemp = reg_axs[ireg][iK+1]
#            axtemp.axis('off')
#            
        
        reg_figs[ireg].suptitle(ca_county)
        reg_figs[ireg].savefig(out_fig,dpi=300,format='png',papertype='letter',orientation='landscape')
        plt.close(reg_figs[ireg])
    
    # Whole coast axis options    
#            axwhole_temp.set_xlim([1e2*-.4,1e2*1.25])
    
#        axw2 = ax_whole[iK+1]
#        axw2.axis('off')
    axwhole_temp.legend(title='SLR scenario [m]')
    out_fig = os.path.join(out_fig_dir,'{}.pdf'.format('AllCA_{0}_inland{1}m'.format(datum_type,int(inland_dist))))
    fig.savefig(out_fig,format='pdf',papertype='letter',orientation='landscape')
        
    area_cols = ['name','sldatum','Kh','sl','D_inland_m','area_analyzed_km2','p{}_area'.format(int(R_limit*1e2)-1),'p{}_percent'.format(int(R_limit*1e2)-1),'p{}_area'.format(int(T_limit*1e2)+1),'p{}_percent'.format(int(T_limit*1e2)+1),'emerg_area','emerg_percent']
    area_df = pd.DataFrame(store_all_p,columns=area_cols)
    
#    allCA_area_df = area_df[area_df['name']=='All'].copy()
    area_df.to_csv(os.path.join(results_dir,'Model_vs_Linear_wt_response_allCA_{}_{}.csv'.format(datum_type,active_date)),
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





