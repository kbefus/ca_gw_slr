# -*- coding: utf-8 -*-
"""
Created on Mon May 23 08:32:21 2016

@author: kbefus
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
   
from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_utils import cgw_raster_utils as cru

# Dictionaries
mpend_col_names = ['ParticleID','ParticleGroup','Status',
                   'InitialTime','FinalTime','InitialGrid','InitialLayer',
                   'InitialRow','InitialColumn','InitialCellFace',
                   'InitialZone','InitialLocalX','InitialLocalY',
                   'InitialLocalZ','InitialGlobalX','InitialGlobalY','InitialGlobalZ',
                   'FinalGrid','FinalLayer','FinalRow','FinalColumn',
                   'FinalCellFace','FinalZone','FinalLocalX','FinalLocalY','FinalLocalZ',
                   'FinalGlobalX','FinalGlobalY','FinalGlobalZ','Label']

mpend_col_dtypes = {'ParticleID':np.int,'ParticleGroup':np.int,'Status':np.int,
                   'InitialTime':np.float,'FinalTime':np.float,'InitialGrid':np.int,'InitialLayer':np.int,
                   'InitialRow':np.int,'InitialColumn':np.int,'InitialCellFace':np.int,
                   'InitialZone':np.int,'InitialLocalX':np.float,'InitialLocalY':np.float,
                   'InitialLocalZ':np.float,'InitialGlobalX':np.float,'InitialGlobalY':np.float,'InitialGlobalZ':np.float,
                   'FinalGrid':np.int,'FinalLayer':np.int,'FinalRow':np.int,'FinalColumn':np.int,
                   'FinalCellFace':np.int,'FinalZone':np.int,'FinalLocalX':np.float,'FinalLocalY':np.float,'FinalLocalZ':np.float,
                   'FinalGlobalX':np.float,'FinalGlobalY':np.float,'FinalGlobalZ':np.float,'Label':str}
                   
mppth_col_names = ['ParticleID','ParticleGroup','TimePointIndex',
                   'CumulativeTimeStep','TrackingTime',
                   'GlobalX','GlobalY','GlobalZ',
                   'Layer','Row','Column','Grid',
                   'LocalX','LocalY','LocalZ','LineSegmentIndex']

mppth_col_dtypes = {'ParticleID':np.int,'ParticleGroup':np.int,'TimePointIndex':np.int,
                   'CumulativeTimeStep':np.int,'TrackingTime':np.float,
                   'GlobalX':np.float,'GlobalY':np.float,'GlobalZ':np.float,
                   'Layer':np.int,'Row':np.int,'Column':np.int,'Grid':np.int,
                   'LocalX':np.float,'LocalY':np.float,'LocalZ':np.float,'LineSegmentIndex':np.int}
                   
tracking_dict = {'Forward':1,'Backward':2}
weaksink_dict = {'Allow':1,'Stop':2}
weaksource_dict = {'Allow':1,'Stop':2}
zonearray_dict = {'NoZoneArray':1,'UseZoneArray':2}

# ========== General Utility functions ==========
def to_py_ind(col_names,fix_names=['Layer','Row','Column']):
    return [colname for colname in col_names if np.any([f1 in colname for f1 in fix_names])]


def save_mp_nc(out_fname=None,model_name=None, XY=None, grid=None,grid_name=None,
               units='hits',var_desc='Number of pathlines crossing cell'):
    
    # Prepare output dictionary
    save_grid_dict = {'out_desc':'Grid data for cgw model {}'.format(model_name),
                      'dims':{'dim_order':['nr','nc'],
                              'nr':{'attr':{'var_desc':'Number of Rows',
                                            'long_name':'n_rows',
                                            'units':'count'},
                                  'data':np.arange(XY[0].shape[0])},
                              'nc':{'attr':{'var_desc':'Number of Columns',
                                            'long_name':'n_cols',
                                            'units':'count'},
                                  'data':np.arange(XY[1].shape[1])}},
                      'X':{'attr':{'var_desc':'X coordinate of cell center in model reference frame',
                                   'long_name':'X','units':'m'},
                                   'dims':('nr','nc'),
                                   'data':XY[0]},
                      'Y':{'attr':{'var_desc':'Y coordinate of cell center in model reference frame',
                                   'long_name':'Y','units':'m'},
                                   'dims':('nr','nc'),
                                   'data':XY[1]},
                      'vars':['X','Y']}            
    

    # Add variable data
    save_grid_dict.update({grid_name:{'attr':{'var_desc':var_desc,
                                           'long_name':'elevation','units':units},
                                           'dims':('nr','nc'),
                                           'data':grid}})
    save_grid_dict['vars'].append(grid_name)    
    
    # Save nc
    cru.save_nc(fname=out_fname,out_data_dict=save_grid_dict) 

      
            
# ========== End General Utility functions ==========

# ========== Modpath Prep functions ==========

def mp_zone_from_zb(zones=None,zone_mapping=None,
                    track_dir=None,release_time=None,
                    track_active = False,land_active=True):

    zone2caf_dictkeys = zone_mapping.keys()
    uniq_zones = np.unique(zones)
    # Use groups if tracking is backwards, zones if tracking is forwards
    if track_dir.lower() in ['backward','back','backwards']:
        if release_time is None:
            release_time=1
        # use unique zones from zone budget zone array for mask
        
        zone_flag = 'NoZoneArray'
        mask_array = []
        group_names = []
        zone_array = 1
        stop_zones = 0 # do not stop particle based on a zone
        
        for unq_zone in uniq_zones:
            if unq_zone in zone2caf_dictkeys:
                if zone_mapping[unq_zone][0] in ['W','w']: #use only waterbodies as starting locations
                    mask_temp = np.zeros_like(zones)
                    mask_temp[zones==unq_zone] = 1
                    mask_array.append(mask_temp)
                    group_names.append(zone_mapping[unq_zone])
                
            if unq_zone==1 and track_active:
                # Active_cells
                mask_temp = np.zeros_like(zones)
                mask_temp[zones==unq_zone] = 1
                group_names.append('ActiveCells')
                mask_array.append(mask_temp)
        ngroups=len(mask_array) 
         
    elif track_dir.lower() in ['forward','forwards']:
        if release_time is None:
            release_time = 0

        zone_flag = 'UseZoneArray'
        
        if land_active:
            # Use all active land cells as one zone
            ngroups=1
            group_names = [] 
            mask_array_input = np.zeros_like(zones)
            
            if not track_active:
                group_names.append('LandCells')
                for unq_zone in uniq_zones:
                    if unq_zone in zone2caf_dictkeys:
                        if zone_mapping[unq_zone][0] in ['L','l']: #use only land as starting locations
                            mask_array_input[zones==unq_zone] = 1
                    
            else:
                # Use all active_cells
                mask_array_input[zones==unq_zone] = 1
                group_names.append('ActiveCells')

            mask_array = [mask_array_input]    
        else:
            mask_array = []
            group_names = []
            
            # Use land data as unique zones for starting locations
            for unq_zone in uniq_zones:
                if unq_zone in zone2caf_dictkeys and not track_active:
                    if zone_mapping[unq_zone][0] in ['L','l']: #use only land as starting locations
                        mask_temp = np.zeros_like(zones)
                        mask_temp[zones==unq_zone] = 1
                        mask_array.append(mask_temp)
                        group_names.append(zone_mapping[unq_zone])
                    
                if unq_zone==1 and track_active:
                    # Active_cells
                    mask_temp = np.zeros_like(zones)
                    mask_temp[zones==unq_zone] = 1
                    group_names.append('ActiveCells')
                    mask_array.append(mask_temp)
            ngroups=len(mask_array) 

        #=========== only if using all active indescriminately ===========
#        # Use all active cells as particle starting locations
#        mask_array_input = np.zeros_like(zones)
#        if track_active:
#            mask_array_input[zones==1] = 1
#        else:
#            mask_array_input[zones>1] = 1
#        mask_array = [mask_array_input for i in range(ngroups)]
        #===========#===========#===========#===========#===========#===========
        
        # Assign active stop zones with unique water zones from zone budget zone array
        zone_array = np.zeros_like(zones)
        stop_zones = 1
        for unq_zone in uniq_zones:
            if unq_zone in zone2caf_dictkeys:
                if zone_mapping[unq_zone][0] in ['W','w']: # only waterbodies stop flowpath
                    zone_array[zones==unq_zone] = stop_zones
    

    output_data = {'zone_array':zone_array,'zone_flag':zone_flag,
                   'mask_array':mask_array, 'release_time':release_time,
                   'ngroups':ngroups,'group_names':group_names,
                   'stop_zones':stop_zones}
           
    return output_data
        
def mp_sim_order_options(mpSim_opt_dict=None):
    '''Order Modpath Simulation options into list.'''
    mpSim_opt_vect = [mpSim_opt_dict['SimulationType'],mpSim_opt_dict['TrackingDirection'],
                  mpSim_opt_dict['WeakSinkOption'],mpSim_opt_dict['WeakSourceOption'],
                  mpSim_opt_dict['ReferenceTimeOption'],mpSim_opt_dict['StopOption'],
                  mpSim_opt_dict['ParticleGenerationOption'],mpSim_opt_dict['TimePointOption'],
                  mpSim_opt_dict['BudgetOutputOption'],mpSim_opt_dict['ZoneArrayOption'],
                  mpSim_opt_dict['RetardationOption'],mpSim_opt_dict['AdvectiveObservationsOption']] 

    return mpSim_opt_vect
            
def mp_order_groups(group_placement_dict=None,ngroups=1):
    '''Order Modpath group placement options into list.'''        
    group_placement = [group_placement_dict['Grid'], group_placement_dict['GridCellRegionOption'],
                   group_placement_dict['PlacementOption'], group_placement_dict['ReleaseStartTime'],
                   group_placement_dict['ReleaseOption'], group_placement_dict['CHeadOption']]
                   
    group_placement_array = [group_placement for i in range(ngroups)]
    
    return group_placement_array
    
# ========== End Modpath Prep functions ==========


# ========== Modpath Run functions ==========

def run_mp_process(mp_model=None):
    '''Run modpath without any i/o or error catching.'''
    
    from subprocess import call
    # Run command from model directory
    call([mp_model.exe_name,mp_model.sim.file_name[0]],cwd=mp_model.model_ws)
#    from subprocess import Popen, PIPE
#    from tempfile import TemporaryFile
            
    
#    with TemporaryFile() as tempFile:
#
#        p = Popen([mp_model.exe_name,mp_model.sim.file_name[0]],stdin=PIPE,stdout=tempFile,stderr=tempFile,
#                  cwd=mp_model.model_ws)
#        
#        [std_out,std_err]=p.communicate()

# ========== End Modpath Run functions ==========


# ========== Load Modpath output functions ==========

def find_ngroups(fpath):
    group_names=[]
    with open(fpath) as f:
        icount=0
        for line in f:
            icount+=1
            if 'END HEADER' in line:
                break
            elif icount>3:
                group_names.append(line.split('\n')[0].strip())
    return icount-5,group_names

def load_mp_endpt(fpath,group_ct=None,nrow_ncol_nlay_nper=None,
                  usecols=['ParticleID','ParticleGroup','InitialLayer',
                           'InitialRow','InitialColumn',
                           'FinalLayer','FinalRow','FinalColumn','Label']):
    '''
    Load Modpath endpoint file to pandas dataframe. Converts lay,row,col index
    to python zero-indexing
    '''
    
    if group_ct is None:
        group_ct,group_names = find_ngroups(fpath)
    
    header_nrows = group_ct+5
    
    mp_endpt_df = pd.read_csv(fpath,sep=' ',skipinitialspace=True,header=None,
                    names=mpend_col_names,skiprows=header_nrows,dtype=mpend_col_dtypes,
                    usecols=usecols)
                    
    # convert lay,row,col to python 0-based indexing
    fix_cols = to_py_ind(mp_endpt_df.columns.values)
    mp_endpt_df[fix_cols] = mp_endpt_df[fix_cols]-1

    return mp_endpt_df

def load_mp_pthln(fpath,usecols=['ParticleID','ParticleGroup','GlobalX','GlobalY','Layer','Row','Column']):
    '''
    Load Modpath pathline file to pandas dataframe. Converts lay,row,col index
    to python zero-indexing
    '''
    mp_path_df = pd.read_csv(fpath,sep=' ',skipinitialspace=True,header=None,
                    names=mppth_col_names,skiprows=3,dtype=mppth_col_dtypes,
                    usecols=usecols)
                    
    # convert lay,row,col to python 0-based indexing
    fix_cols = to_py_ind(mp_path_df.columns.values)
    mp_path_df[fix_cols] = mp_path_df[fix_cols]-1
    
    return mp_path_df

def iter_load_mp_pthln(fpath,chunksize=1e6,
                       usecols=['ParticleID','ParticleGroup','GlobalX','GlobalY','Layer','Row','Column']):
    
    mp_path_iter = pd.read_csv(fpath,sep=' ',skipinitialspace=True,header=None,
                    names=mppth_col_names,skiprows=3,chunksize=chunksize,
                    iterator=True,dtype=mppth_col_dtypes,usecols=usecols)
    
    return mp_path_iter


# ========== End (Load Modpath output functions) ==========


# ========== Post-Modpath functions ==========
def mp_pt_array(mp_endpt_df=None, array_shp=None, loc_name='start',
                      group_col='ParticleGroup'):
    '''Create grid showing particle start or end locations.'''
    
    if loc_name.lower() in ['start','initial','begin']:
        ind_cols=['InitialLayer','InitialRow','InitialColumn']
    elif loc_name.lower() in ['end','final','stop']:
        ind_cols=['FinalLayer','FinalRow','FinalColumn']
                      
    loc_array = np.nan*np.zeros(array_shp)
    
    if len(loc_array.shape)==2:
        loc_array[mp_endpt_df[ind_cols[1]],
                  mp_endpt_df[ind_cols[2]]] = mp_endpt_df[group_col]
    else:
        loc_array[mp_endpt_df[ind_cols[0]],
                  mp_endpt_df[ind_cols[1]],
                  mp_endpt_df[ind_cols[2]]] = mp_endpt_df[group_col]
                  
    return loc_array

def select_particles_by_mask(mp_endpt_df=None,mask=None,loc_name=None):
    
    if loc_name.lower() in ['start','initial','begin']:
        ind_cols=['InitialLayer','InitialRow','InitialColumn']
    elif loc_name.lower() in ['end','final','stop']:
        ind_cols=['FinalLayer','FinalRow','FinalColumn']
    
    mask = mask.copy()
    if mask.dtype is not bool:
        mask[np.isnan(mask)] = False
        mask = mask.astype(bool)
    
    if len(mask.shape)==2:
        active_rc = np.array(np.where(mask)).T
        ind_cols = ind_cols[1:]
    else:
        active_rc = np.array(np.where(mask)).T
        
    rc_df = pd.DataFrame(data=active_rc,columns=ind_cols)

    select_df = pd.merge(mp_endpt_df,rc_df,left_on=ind_cols,right_on=ind_cols)
    return select_df['ParticleID'].unique()

def assign_zbzones(pt_locs=None,zones=None,zb_dict=None):
    
    pt_loc_zones = pt_locs*zones.squeeze()
    unique_zones = np.unique(pt_loc_zones[~np.isnan(pt_loc_zones)]).astype(np.int)
    out_names=[]
    out_pt_locs = np.nan*np.zeros_like(pt_locs)
    
    for icount,unq_zone in enumerate(unique_zones):
        out_names.append(zb_dict[unq_zone])
        out_pt_locs[pt_loc_zones==unq_zone] = np.int(icount+1)

    return out_pt_locs,out_names
    
    
def save_endpt_shp(XY=None,Z=None,proj_kwargs=None,out_fname=None,group_names=None):
    
    # Make polygon from grid
    cc_dict = {'XY':XY}
    [_,_,_,cellspacing]= return_cc(**cc_dict)
    
    # reset origin in XY grids
    XY[0] = XY[0]-XY[0][0,0]
    XY[1] = XY[1]-XY[1][-1,0]
    
    extent_outline,unq_vals,n_vals = cfu.raster_to_polygon(XY=XY,Z=Z,
                                                           cell_spacing=cellspacing,
                                                           select_largest=False,
                                                           slow_method=True)
    
    # Project to output coordinate system from model coordinate system
    proj_dict = {'polys':extent_outline,'proj_kwargs':proj_kwargs}
    polys_proj = cfu.proj_polys(**proj_dict)
    
    # Covert str group name to catchment FID
    group_fids = [tempname.split('_')[0][1:] for tempname in group_names] # format is L6581_lay0
    
    # Assign group number to name
    out_fids = np.array(group_fids)[np.array(unq_vals).astype(np.int)-1] # convert group number to python index

    # Write shapefile 
    out_data = np.array(zip([[int(os.path.basename(out_fname).split('_')[1])]*len(unq_vals),
                              out_fids,
                              unq_vals,
                              n_vals])).squeeze().T
    col_name_order = ['CAFmodel','CAF_FID','MPgroup','Ncells']
    field_dict = cfu.df_field_dict(None,col_names=col_name_order,col_types=['int','int','int','int'])
#    inproj = proj_kwargs['proj_in']
    inproj=None
    if 'proj_out' in proj_kwargs.keys():
        
        if proj_kwargs['proj_out'] is not None:
            inproj = proj_kwargs['proj_out']
        
    
    shp_dict = {'polys':polys_proj,'data':out_data,'out_fname':out_fname,
               'field_dict':field_dict,'col_name_order':col_name_order,
               'write_prj_file':True,'inproj':inproj}
                               
    cfu.write_shp(**shp_dict)
    return extent_outline,polys_proj
    
def collect_zonebytype(zb_dict=None,zones=None,search_terms=['w','W']):
    
    active_zone_vals = np.array([zone for zone in zb_dict.keys()\
                        if zb_dict[zone][0] in search_terms and isinstance(zone,int)])
    
    zones = zones.squeeze().copy()
    out_bool_array = np.zeros_like(zones,dtype=bool)
    for temp_zone in active_zone_vals:
        out_bool_array[zones==temp_zone] = True
    
    
    return out_bool_array,np.array(out_bool_array.nonzero()).T

def select_pthlines(rc_array=None,mp_path_df=None):
    '''Select only particles with pathlines reaching rc_array indexes.'''

    if rc_array.shape[1]==2:
        
        bool_test = mp_path_df['Row'].isin(rc_array[:,0]) & mp_path_df['Column'].isin(rc_array[:,1])
    else:
        bool_test = mp_path_df['Layer'].isin(rc_array[:,0]) &\
                    mp_path_df['Row'].isin(rc_array[:,1]) &\
                    mp_path_df['Column'].isin(rc_array[:,2])
    
    particle_numbers = mp_path_df.ix[bool_test,'ParticleID'].unique()
    return mp_path_df[mp_path_df['ParticleID'].isin(particle_numbers)]
    
def return_cc(XY=None, n_decimate=1, cell_centeredXY=True):
    
    if not cell_centeredXY:
        # Use grid nodes in model domain coordinate system
        # to caclulate flowpath intersections
        x_nodes = XY[0][0,::n_decimate].ravel()
        y_nodes = XY[1][::n_decimate,0].ravel() 
        xcenters = x_nodes[:-1] + 0.5 * (x_nodes[1:] - x_nodes[:-1])
        ycenters = y_nodes[:-1] + 0.5 * (y_nodes[1:] - y_nodes[:-1])
        X,Y = np.meshgrid(xcenters,ycenters)
    else:
        X,Y = XY
        xcenters = X[0,::n_decimate].ravel()
        ycenters = Y[::n_decimate,0].ravel()
        x_nodes = np.hstack([xcenters[:-1]-.5 * (xcenters[1:]-xcenters[:-1]),
                             xcenters[-1]-(xcenters[-1]-xcenters[-2])/2.,
                             xcenters[-1]+(xcenters[-1]-xcenters[-2])/2.])
        y_nodes = np.hstack([ycenters[:-1]-.5 * (ycenters[1:]-ycenters[:-1]),
                             ycenters[-1]-(ycenters[-1]-ycenters[-2])/2.,
                             ycenters[-1]+(ycenters[-1]-ycenters[-2])/2.])
    
    cellspacing = [np.abs(xcenters[1]-xcenters[0]),np.abs(ycenters[1]-ycenters[0])]

    return [X,Y],[xcenters,ycenters],[x_nodes,y_nodes],cellspacing

def iter_extent_pathlines(in_iter_df=None,extent_kwargs=None):
    
    last_df = None
    all_hits = None
    all_polys = []
    out_groups = []
    # Loop through chunks of dataframe
    for iter_df in in_iter_df:
        hit_array,temppolys,last_df,active_group_list = iter_calc_max_extent(mp_path_df=iter_df,
                                                           old_df=last_df,
                                                           **extent_kwargs)
        
        if len(temppolys)>0:
            # Sum hits
            if all_hits is None and len(hit_array)>0:
                all_hits = hit_array.copy()
            elif len(hit_array)>0:
                all_hits += hit_array
            
            all_polys.extend(temppolys)
            out_groups.extend(active_group_list)
#        elif active_group_list is not None:
#            all_polys.extend([0]) # add space filler only if not passing full dataframe
            
    in_iter_df.close()
    
    # Run calc_max_extent on the last last_df
    calcextent_dict = {}
    calcextent_dict.update(extent_kwargs)
    calcextent_dict.update({'mp_path_df':last_df,
                            'active_groups':[active_group_list[-1]+1],
                           'XY':extent_kwargs['XY']})
                           
    last_hit_array,last_extentpolys,lastgroup = calc_max_extent(**calcextent_dict)
    
    if len(last_extentpolys)>0:
        # Sum hits
        if all_hits is None and len(last_hit_array)>0:
            all_hits = last_hit_array.copy()
        elif len(last_hit_array)>0:
            all_hits += last_hit_array
        
        out_groups.extend(lastgroup)
        all_polys.extend(last_extentpolys)
#    else:
#        all_polys.extend([0]) # add space filler only if not passing full dataframe
            

    return all_hits, all_polys, out_groups
        
    
    
def iter_calc_max_extent(mp_path_df=None,active_groups=None,old_df=None,
                         dxdy=None,XY=None, n_decimate=1, cell_centeredXY=True,
                         zone_kwargs=None):
    '''Iteration envelope function for calculating pathline extents.
    
    mp_path_df: pandas dataframe
                A dataframe containing at least 'ParticleGroup', 'GlobalX', and
                'GlobalY' columns.
    
    active_groups: list or np.ndarray
                   list or array of integers of the particle groups over which
                   to conduct analysis. Not names of the groups.
                   
                   '''
    
    
    # need to convert lay,row,col to python 0-based indexing
    fix_cols = to_py_ind(mp_path_df.columns.values)
    mp_path_df[fix_cols] = mp_path_df[fix_cols]-1
    
    if old_df is not None:
        # Join previous "last_group" data with current dataframe
        mp_path_df = pd.concat([old_df,mp_path_df],axis=0,ignore_index=True)

    if active_groups is None:
        active_groups = mp_path_df['ParticleGroup'].unique()
    
    # Run normal calc_max_extent for all but last group
    last_group = mp_path_df.iloc[-1]['ParticleGroup']

    if isinstance(active_groups,np.ndarray):
        active_group_list = active_groups.tolist()
        active_group_list.remove(last_group)
    elif isinstance(active_groups,list):
        active_group_list = active_groups
        active_group_list.remove(last_group)
    else: # Unique returned a single number, so return only the dataframe to include in next iteration
        active_group_list = None
    
    hit_array = []
    extentpolys=[]
    actgroups = []

    if active_group_list is not None:
        if len(active_group_list)>0:
           calcextent_dict = {'mp_path_df':mp_path_df,'active_groups':active_group_list,
                               'dxdy':dxdy,'XY':XY,'n_decimate':n_decimate,
                               'cell_centeredXY':cell_centeredXY,'zone_kwargs':zone_kwargs}
                           
           hit_array,extentpolys,actgroups = calc_max_extent(**calcextent_dict)
    
    return hit_array, extentpolys, mp_path_df[mp_path_df['ParticleGroup']==last_group].copy(),actgroups
    
    
def calc_max_extent(mp_path_df=None,active_groups=None,dxdy=None,XY=None,
                    n_decimate=1, cell_centeredXY=True, coastal_bool=True,
                    zone_kwargs=None,):
    '''Calculate extent of pathlines.'''  
    
    cc_dict = {'cell_centeredXY':cell_centeredXY,'XY':XY,'n_decimate':n_decimate}
    [[X,Y],[xcenters,ycenters],[x_nodes,y_nodes],cellspacing]=return_cc(**cc_dict)
    
    if coastal_bool:
        # only run analysis on pathlines reaching coastal waters
        bool_array,rc_inds = collect_zonebytype(**zone_kwargs)
        mp_path_df = select_pthlines(rc_array=rc_inds,mp_path_df=mp_path_df)
    
    if active_groups is None:
        active_groups = mp_path_df['ParticleGroup'].unique()
                   
    xyrange = [[XY[0].min(),XY[0].max()],[XY[1].min(),XY[1].max()]]
    all_hits = np.zeros_like(X)
    all_polys = []
    no_hit_group = []
    for active_group in active_groups:
        temp_df = mp_path_df[mp_path_df['ParticleGroup']==active_group].copy()
        
        # 2D histogram of pathline-grid cell intersections
        hitmap,xedge,yedge = np.histogram2d(temp_df['GlobalX'],temp_df['GlobalY'],
                                            bins=[x_nodes,y_nodes[::-1]],range=xyrange)
        
        # Stack pathline hits                                    
        all_hits += hitmap.T
        
        # Find outer border of pathlines in group        
        hitmap[hitmap==0] = np.nan
        #extent_outline_pts = cru.raster_outline(X,Y,hitmap[:,::-1])
        if np.nansum(hitmap)!= 0.:
            extent_outlines,Zval,ncells = cfu.raster_to_polygon(XY=[X,Y],Z=hitmap.T[::-1,:],
                                                       cell_spacing=cellspacing,unq_Z_vals=False)
            all_polys.extend(extent_outlines) 
        else:
            no_hit_group.append(active_group)
#        extent_pts_new = np.array(cfu.sort_xy_cw(extent_outline_pts[:,0],extent_outline_pts[:,1])).T
        # Create polygon of outer border       
           
    for ibad in no_hit_group:
        active_groups.remove(ibad)
        
    return all_hits[::-1,:],all_polys,active_groups

def mp_plot_paths(mp_path_df=None,active_group=None,plot_array=None,
                  XY=None,dxdy=[None,None],ax=None):
    
#    tcol = 'TrackingTime'
    temp_df = mp_path_df[mp_path_df['ParticleGroup']==active_group].copy()
    uniq_particles = temp_df['ParticleID'].unique()
#    maxtime = temp_df[tcol].max()
#    mintime = temp_df[tcol].min()
    Particle_df = temp_df.groupby('ParticleID')
    cmap = plt.cm.rainbow(np.arange(len(uniq_particles))/float(len(uniq_particles)))
    
    if ax is None:    
        fig,ax=plt.subplots()
    
        ax.pcolormesh(XY[0],XY[1],np.ma.masked_invalid(plot_array),cmap= plt.cm.gray,
                      edgecolors='k')
      
    for ipart,uniq_part in enumerate(uniq_particles):
        temp_df2 = Particle_df.get_group(uniq_part)
#        i1=ax.scatter(temp_df2[xcol],temp_df2[ycol],c=cmap[ipart,:],
#                   edgecolors='none',linewidths=1).
#        xtemp = XY[0][temp_df2[ycol],temp_df2[xcol]]-dxdy[0]/2.+dxdy[0]*temp_df2[xcolloc]
#        xpos_dict = {'global_pos':temp_df2['GlobalX'],'local_pos': temp_df2['LocalX'],'dloc':dxdy[0]}
#        ypos_dict = {'global_pos':temp_df2['GlobalY'],'local_pos': temp_df2['LocalY'],'dloc':dxdy[0]}        
#        xtemp = calc_abs_pos(**xpos_dict)
#        ytemp = calc_abs_pos(**ypos_dict)
#        ytemp = XY[1][temp_df2[ycol],temp_df2[xcol]]-dxdy[1]/2.+dxdy[1]*temp_df2[ycolloc]        
        xtemp  =  temp_df2['GlobalX']
        ytemp = temp_df2['GlobalY']
        if ipart==0:
            xmin,xmax,ymin,ymax = xtemp.min(),xtemp.max(),ytemp.min(),ytemp.max()
        else:
            xmin,xmax = np.minimum(xmin,xtemp.min()),np.maximum(xmax,xtemp.max())
            ymin,ymax = np.minimum(ymin,ytemp.min()),np.maximum(ymax,ytemp.max())
        ax.plot(xtemp,ytemp,'-.',c=cmap[ipart,:])
        
#    plt.colorbar(i1,ax=ax)

    ax.axis([xmin,xmax,ymin,ymax])
    
def calc_abs_pos(global_pos=None,local_pos=None,dloc=None):
    '''
    Calculate the absolute position from modpath outputs. Only calculate one dimension,
    so must be called for x, y, and z separately
    
    WRONG! USE GLOBALXYZ directly!
    
    Parameters
    ----------
    
    global_pos: array of 'GlobalX','GlobalY', or 'GlobalZ'.
    
    local_pos: corresponding array of 'LocalX','LocalY', or 'LocalZ'.
    
    dloc: dx, dy, or dz value.
    
    Returns
    ----------
    
    absolute_pos: absolute position in model coordinates.
    
    '''
    return global_pos + dloc * local_pos

# ========== End Post-Modpath functions ==========


