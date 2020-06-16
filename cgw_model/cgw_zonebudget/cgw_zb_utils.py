# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:31:15 2016

@author: kbefus
"""

import os
import numpy as np
from subprocess import Popen, PIPE, STDOUT
from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_utils import cgw_general_utils as cgu
import pandas as pd
from scipy.spatial.distance import cdist

# --------- Preparing data for Zone Budget -----------
def assign_zone(zone_polys,XY,n_other_zones=0):
    grid_inds = cfu.gridpts_in_shp(zone_polys,XY)
    uniq_zone_nums,zone_unq_ind,zone_unq_inv = np.unique(grid_inds[2],return_index=True,return_inverse=True)
    zone_mat = np.zeros_like(XY[0])
    zone_mat[grid_inds[0],grid_inds[1]] = zone_unq_inv+n_other_zones+1
    polyind_zoneind = np.vstack([uniq_zone_nums,np.arange(uniq_zone_nums.shape[0])+n_other_zones+1]).T # original index, zonebudget zone number
    return zone_mat,polyind_zoneind

def merge_attr(attr_list):
    '''Merge features with same attribute.'''
    shps,in_index,id_num = attr_list
    # Find unique id's
    unq_ids,unq_inv = np.unique(id_num,return_inverse=True)
    dup_ids = [unq_ids[id_indval] for id_indval in np.unique(unq_inv) if len((unq_inv==id_indval).nonzero()[0])>1]
    # If duplicates are found, merge shapes, keep larger of input indexes
    save_bool = np.ones(id_num.shape,dtype=bool)
    for dup_id in dup_ids:
        matches_found = (id_num==dup_id).nonzero()[0]
        shps_found = np.array(shps)[matches_found]
        areas_found = [shp.area for shp in shps_found]
        max_area_ind = np.argmax(areas_found)
        all_inds = np.arange(len(matches_found))
        shps[matches_found[max_area_ind]] = cfu.unary_union(shps_found) # overwrite larger with merged shps
        save_bool[matches_found[(all_inds!=max_area_ind)]] = False # don't save smaller shp
    
    # overwrite items
    shps=np.array(shps)[save_bool]
    in_index = np.array(in_index)[save_bool]
    id_num = id_num[save_bool]
    
    attr_list = shps.tolist(),in_index.tolist(),id_num
    return attr_list
        
        

def add_zb_layer(zone_array,num_add=None,new_zones=True):
    if num_add is None:
        num_add = np.int(np.nanmax(zone_array)+1)
        
    if new_zones:
        new_zone_array = np.zeros_like(zone_array)
        new_zone_array[zone_array!=0] = zone_array[zone_array!=0]+num_add
    else:
        new_zone_array = zone_array
        
    return new_zone_array,num_add
    
def update_zb_assignment(assign_array=None,num_add=None,ilay=None):
    temp_assign = assign_array.copy()[assign_array[:,-1]==0,:]
    temp_assign[:,1] = temp_assign[:,1]+num_add
    temp_assign[:,2] = ilay
    out_array = np.vstack([assign_array,temp_assign])
    return out_array

def fill_closest_zone(zone_array,zone_range=None,fill_active=False,fill_wb=False,
                      row_col_buff = 50,
                      active_zone=cgu.grid_type_dict['active'],
                      wb_zone=np.abs(cgu.grid_type_dict['nearshore'])):
    '''Fill unassigned cells with closest zone.
    
        Inputs
        ----------
        
        zone_array: np.ndarray
            2D integer array with cells assigned to specific zones by unique integers
    '''
    if len(zone_array.shape)==3:
        zones_in = zone_array[0].copy()
    else:
        zones_in = zone_array.copy()
    
    if fill_active:
        zone_num = active_zone
    elif fill_wb:
        zone_num = wb_zone
    else:
        zone_num = 0
        
    # Cells with missing zone, [[row,col]]
    cells_no_zone =  np.array((zones_in==zone_num).nonzero()).T
    
    # Zones to match missing assignments to
    match_zones = np.array(((zones_in<=np.max(zone_range)) & (zones_in>=np.min(zone_range)) \
                  & (zones_in!=0) & ~np.isnan(zones_in)).nonzero()).T
    
    zones_out = zones_in.copy()
    # loop through pairs that need a match individually to save memory and apply max buffer distance
    for no_zone_pair in cells_no_zone:
        subset_bool = (match_zones[:,0]<=no_zone_pair[0]+row_col_buff) &\
                      (match_zones[:,0]>=no_zone_pair[0]-row_col_buff) &\
                      (match_zones[:,1]<=no_zone_pair[1]+row_col_buff) &\
                      (match_zones[:,1]>=no_zone_pair[1]-row_col_buff)
        match_zone_subset = match_zones[subset_bool]
        if len(match_zone_subset)>0:
            match_mindist_ind = np.argmin(cdist([no_zone_pair],match_zone_subset),axis=1)
            match_ind = match_zone_subset[match_mindist_ind[0],:]
            zones_out[no_zone_pair[0],no_zone_pair[1]] = zones_in[match_ind[0],match_ind[1]]
            
    return zones_out
    
    
# ---------- Running Zone Budget ---------------
def ModflowZoneBudget(modelname,workspace,zone_array,nrow_ncol_nlay_nper,zone_layer = None,
                      composite_dict=None,exe_name = 'zonbud.exe',
                      input_list = None,zeta_array=None):
    '''
    Create ZoneBudget .zon input file and run zonbud.exe
    
    Parameters
    ----------
    
    modelname:
        Name of modflow model
    
    workspace:
        Workspace of modflow model
    
    nrow_ncol_nlay_nper:
        List of number of rows, columns, layers, and periods
    
    zone_array:
        numpy array of zones specified by unique integers
        
    zone_layer:
        1-based index of maximum layer to apply zones moving from surface. Default is all layers
    
    composite_dict:
        dictionary of composite zones of the form {'composite name': [zones in group]}
        
    '''

    nrow,ncol,nlay,nper =nrow_ncol_nlay_nper
    
    if zone_layer is None:
        zone_layer = np.arange(nlay) # apply zones to all layers
    elif isinstance(zone_layer,(int,float)):
        zone_layer = np.arange(zone_layer+1,dtype=np.int) # all layers down to zone_layer
    

    zone_file = os.path.join(workspace,'{}.zon'.format(modelname))

    npl = ncol
    column_length = ncol
    
    if zone_array.dtype == np.int:
        zone_array = zone_array.astype(np.int)
    
    
    fmt_fortran = '(' + str(npl) + 'I10) ' # I10
    output_fmt = '{0:10d}'
    # Write input file
    write_ZoneBudget_input(zone_file,zone_array,zone_layer,[nlay,nrow,ncol],
                           fmt_fortran=fmt_fortran,output_fmt=output_fmt,
                           column_length=column_length,
                           composite_dict=composite_dict)
    
    # Run ZoneBudget
    run_ZoneBudget(input_list,modelname,workspace,exe_name,path_to_exe=None)
    return zone_file
            
def write_ZoneBudget_input(zone_file,zone_array,zone_layer,dims,
                           fmt_fortran=None,output_fmt=None,column_length=None,
                           composite_dict=None):
                   
    nlay,nrow,ncol = dims
    array_shape = np.array([nlay,nrow,ncol])    
    zone_number = np.max(zone_array) # create new constant zones for unassigned layers
    active_layer = -1 # allows discontinuous layers to be assigned zones by layer
    
    if ncol%column_length == 0:
        lineReturnFlag = False
    else:
        lineReturnFlag = True    
    
    threeDarray = False
    if len(zone_array.shape)==3:
        threeDarray = True    
    
    with open(zone_file, 'w') as file_out:
        # print model shape, nlay nrow ncol
        file_out.write(u'\t'.join(array_shape.astype('|S10').tolist())+'\n')
        
        for k in range(nlay):
            # If layer is part of zoned layers
            if k in zone_layer:
                active_layer += 1
                file_out.write('{}            {}\n'.format("INTERNAL",fmt_fortran))
                # write the array
                for i in range(nrow):
                    #icol = 0
                    for j in range(ncol):
                        if threeDarray:
                            try:
                                file_out.write(output_fmt.format(zone_array[active_layer,i,j]))
                            except:
                                print('Value {0} at lay,row,col [{1},{2},{3}] can not be written'\
                                    .format(zone_array[active_layer, i, j], k, i, j))
                                raise Exception
                            if (j + 1) % column_length == 0.0 and j != 0:
                                file_out.write('\n')
                        else:
                            try:
                                file_out.write(output_fmt.format(zone_array[i,j]))
                            except:
                                print('Value {0} at row,col [{1},{2}] can not be written'\
                                    .format(zone_array[i, j], i, j))
                                raise Exception
                            if (j + 1) % column_length == 0.0 and j != 0:
                                file_out.write('\n')
                    if lineReturnFlag == True:
                        file_out.write('\n')
            else:
                # If layer is not in zoned layers, assign constant value
                zone_number += 1
                file_out.write('{}           {}\n'.format("CONSTANT",zone_number))
                
        if composite_dict is not None:
            for ic,composite_zone in enumerate(composite_dict):
                file_out.write('{:10s}'.format(composite_zone.upper()))
                for composite_piece in composite_dict[composite_zone]:
                    file_out.write(output_fmt.format(composite_piece))
                file_out.write('\n')


def run_ZoneBudget(input_list,modelname,workspace,exe_name,path_to_exe=None):
    # Run ZoneBudget
    if input_list is None:
        input_list = ['{}_ZONBUD ZBLST CSV'.format(modelname),
                      '{}.cbc'.format(modelname), modelname,
                      '{}.zon'.format(modelname),"A"]
    
    # Set up way to not print out zb cmd info in python run
    try:
        from subprocess import DEVNULL # py3k
    except ImportError:
        DEVNULL = open(os.devnull, 'wb')    
    
    if len(input_list)==5:
        if (path_to_exe is not None):
            p = Popen(os.path.join(path_to_exe,exe_name), stdin=PIPE,
                      stdout=DEVNULL, stderr=STDOUT,cwd=workspace) #NOTE: no shell=True here
        else:
            p = Popen(exe_name, stdin=PIPE,stdout=DEVNULL, stderr=STDOUT,cwd=workspace) # assumes exe_name directory is in path
        p.communicate(os.linesep.join(input_list))
    else:
        raise Exception('len(input_list) must equal 5, currently equals {0:d})'.format(len(input_list)))

# ---------- Post processing functions --------------
#%%
def load_ZONBUD(fname,last_time_only=True):
    '''Load ZoneBudget output file to pandas dataframes.
    '''
    df_all = pd.read_csv(fname,delimiter=',',skip_blank_lines=True,index_col=False,skiprows=1)
    df_all.dropna(axis=1,how='all',inplace=True) # Remove hanging comma column
    # fix column names
    column_names = df_all.columns.values
    zone_col = 'Zone info'
    column_names[0] = zone_col
    cols_orig = ['_'.join(col.strip().split()) for col in column_names]
    df_all.columns = cols_orig
    zone_col = df_all.columns.values[0]
    
    row_names = np.char.join('_',np.char.split(np.char.strip(np.char.array(df_all[zone_col].values))))
    empty_row_inds = [i for i,row_name in enumerate(row_names) if len(row_name)==0]
    df_all[zone_col] = row_names
    
    if len(empty_row_inds) != 2:  # more than one time period
        if last_time_only:        
            empty_row_inds=empty_row_inds[-2:] # use final tables if more than one output periods
        else:
            # Remove internal headers about stress/time period
            bad_rows = empty_row_inds[2::3]        
            for bad_row in bad_rows: empty_row_inds.pop(empty_row_inds.index(bad_row))
    
    start_inds = empty_row_inds
    end_inds = np.roll(empty_row_inds,-1)
    end_inds[-1] = df_all.shape[0]
    flux_directions = ['TO','FROM']
    flux_dirs = np.tile(flux_directions,end_inds.shape[0]/2)
    flux_dfs = [read_flux_to_df(df_all,cols_orig,start_ind=s_ind,end_ind=e_ind,flux_dir=f_dir) for s_ind,e_ind,f_dir in zip(start_inds,end_inds,flux_dirs)]
    flux_in_dfs = flux_dfs[0::2]
    flux_out_dfs = flux_dfs[1::2]
    
    flux_budget_dfs = [zb_zone_summary(df_in,df_out,cols_orig) for df_in,df_out in zip(flux_in_dfs,flux_out_dfs)]
    
    return flux_budget_dfs
    
def read_flux_to_df(df_all,cols_orig,start_ind=0,end_ind=None,flux_dir=''):
    '''Extract ZoneBudget flux data from ZoneBudget table.
    '''
    df_flux = pd.DataFrame(df_all.iloc[start_ind+1:end_ind,:].values,columns=cols_orig)
    zone_in_col0=df_all.columns.values[0]
    # Strip FROM from zone column            
    new_rows = np.char.join('',np.char.split(np.char.array(df_flux[zone_in_col0]),'{}_'.format(flux_dir)))            
    df_flux[zone_in_col0] = new_rows
    column_names2 = df_flux.columns.values
    column_names2[0] = '{0:5s}'.format(flux_dir)+column_names2[0]
    cols2 = ['_'.join(col.strip().split()) for col in column_names2]
    df_flux.columns = cols2
    zone_in_col = df_flux.columns.values[0]
    df_flux.set_index(zone_in_col,inplace=True)
    df_flux.dropna(axis=0,how='all',inplace=True) # remove all nan rows 
    return df_flux
    
def zb_zone_summary(df_fin,df_fout,cols_orig,ntail=2):
    '''Extract ZoneBudget summary data from flux dataframes.
    '''
    budget_rows = df_fout.tail(ntail)
    df_fout.drop(budget_rows.index.values,inplace=True) # remove budget rows from df_fout        
    
    summary_inds = np.hstack([df_fin.index.values[-1],df_fout.index.values[-1],budget_rows.index.values])
    summary_data = np.hstack([df_fin.ix[-1,:].values.reshape((-1,1)),df_fout.ix[-1,:].values.reshape((-1,1)),budget_rows.values.T])
    df_budget_summary = pd.DataFrame(summary_data,columns = summary_inds,index=cols_orig[1:])
    
    # Drop budget rows from each matrix,last row of each
    df_fout.drop(df_fout.index.values[-1],inplace=True) 
    df_fin.drop(df_fin.index.values[-1],inplace=True)
    
    # Convert dataframe entries to float
    df_budget_summary = df_budget_summary.apply(pd.to_numeric, errors='coerce')
    df_fout = df_fout.apply(pd.to_numeric, errors='coerce')
    df_fin = df_fin.apply(pd.to_numeric, errors='coerce')
    return df_fin,df_fout,df_budget_summary

zone_type_dict = {'ZONE_1':'ActiveAll','ZONE_2':'GHBAll'}
    
def rename_df_entry(df_in,rename_dict=zone_type_dict,rename_index=False,
                    return_output=False):
    '''
    Rename ZoneBudget zones in a pandas dataframe 
    using a dictionary of the form 'zb_name':'new_name'.
    '''
    
    if rename_index is True:
        # Only rename index    
        df_in.index = rename_df_func(in_list=df_in.index.values,
                                     rename_dict=rename_dict)
    elif rename_index in ['Both','both']:
        # Rename columns and index
        df_in.index = rename_df_func(in_list=df_in.index.values,
                                     rename_dict=rename_dict)
                                           
        df_in.columns = rename_df_func(in_list=df_in.columns.values,
                                     rename_dict=rename_dict)
    else:
        # Rename columns only
        df_in.columns = rename_df_func(in_list=df_in.columns.values,
                                     rename_dict=rename_dict)
    if return_output: 
        return df_in
        
def rename_df_func(in_list=None,inds_to_change=None,rename_dict=None):
    '''
    Find and rename dataframe entries specified by inds_to_change
    '''
    out_list = in_list.copy()
    active_keys = rename_dict.keys()
    active_key_nums = [int(ak.split('ZONE')[-1].split('_')[-1]) for ak in active_keys if not isinstance(ak,(int,np.int))]
    active_key_names = [ak for ak in active_keys if not isinstance(ak,(int,np.int))]
    
    if (inds_to_change is None):
        inds_to_change = range(len(in_list))
            
    for name_find_ind in inds_to_change:
        try:
            find_key_ind = active_key_nums.index(int(in_list[name_find_ind].split('ZONE')[-1].split('_')[-1]))
            
            out_list[name_find_ind] = rename_dict[active_key_names[find_key_ind]]
        except:
            pass
    return out_list
    
    
def zones_to_caf(cafind_to_zone=None,caf_var=None,
                 input_zone_names=None,output_fmt = '{}'):
    '''
    Return CAF HUCs given active CAF index mapping to zones (cafind_to_zone)
    
    input_zone_names: incoming names of the zones as reported by zonebudget
    caf_var: input array of new values to assign, must have len == len(input_zone_names)
    '''
    if (input_zone_names is not None):
        if isinstance(input_zone_names,list):
            input_zone_names = np.array(input_zone_names)
            output_zone_names = input_zone_names.copy()
    
    output_zone_dict = {}    
    zones_found = []
    new_names = []
    if cafind_to_zone.shape[1]==2:
        # Assign to layer 0
        cafind_to_zone = np.hstack([cafind_to_zone,np.zeros((cafind_to_zone.shape[0],1))])
        
    for caf_var_temp,(caf_ind,zone_num,lay_num) in zip(caf_var[cafind_to_zone[:,0].astype(int)],cafind_to_zone.astype(np.int)):
        tempzone = 'ZONE_{0:.0f}'.format(zone_num)
        zones_found.append(tempzone)
        out_name = output_fmt.format(caf_var_temp,lay_num) # can add option to use caf_var[caf_ind]
        new_names.append(out_name)
        if (input_zone_names is not None):
            output_zone_names[input_zone_names==tempzone] = out_name
    
    output_zone_dict.update(zip(zones_found,new_names))
    output_zone_dict.update(zip(cafind_to_zone.astype(np.int)[:,1],new_names))
    
    if (input_zone_names is not None):
        return output_zone_dict,output_zone_names
    else:
        return output_zone_dict,[]

def test_int(text1):
    try:
        int(text1)
        return True
    except:
        return False

def calc_net_discharge(in_df,nzones):
    '''
    Calculate net discharge from lower, constant zone layer to overlying layer of model.
    
    
    '''
    zone_name = 'ZONE_{0:.0f}'.format(nzones+1)
    
    # Calculate discharge from lower layer to upper zones
    lay2_discharge = in_df.ix['FROM_'.format(zone_name)].T.groupby(level=0).sum()
    
    # Calculate recharge into lower layer from upper zones
    lay2_recharge = in_df[zone_name].groupby(level=0).sum()
    
    net_discharge = lay2_discharge-lay2_recharge
    
    return net_discharge


def save_RchgDrain_flux(in_df,out_fname=None, out_cols=['RECHARGE','DRAINS']):
    
    # Calculate cgw contribution
    Cgwdf=in_df.filter(regex="W").T.filter(regex='L').T.sum(axis=1)    
    Cgwdf.columns = ['CGW']
    # Calculate Recharge and drain fluxes
    RDdf = in_df.filter(regex="L").T[out_cols]
    RDdf = pd.concat([RDdf,Cgwdf],axis=1)
    RDdf['fid'] = [tempind.split('_')[0][1:] for tempind in RDdf.index.values]
    RDdf.to_csv(out_fname,index=False)
    
def calc_cgw_flux(in_df,active_layer=None,q_row_name='ActiveAll',active_only=False):
    temp_df = in_df.copy()
    temp_df = temp_df.filter(regex="W") # only select water zones (columns)
    if active_only: # if no land zones are assigned
        active_cols = [col for col in temp_df if active_layer in col]
        total_land2water_flux = temp_df.ix[q_row_name,active_cols].copy()
        feature_id_array = total_land2water_flux.columns.values
        data_array = total_land2water_flux.values
        out_data= [feature_id_array,['Qm3day']*data_array.shape[0],data_array]
        wb_influx_dict = total_land2water_flux.to_dict()
    else:
        temp_df[temp_df==0] = np.nan
        
        if active_layer is None:
            input_rows = [irow for irow in temp_df.index.values if 'L' in irow]
        else:
            if isinstance(active_layer,(float,int)):
                active_layer = 'lay{0:.0f}'.format(active_layer)
            input_rows = [irow for irow in temp_df.index.values if 'L' in irow and active_layer in irow]
        wb_influx_dict = {}
        for col in temp_df:
            if active_layer is not None:
                # Only update dict if col is for active_layer
                if active_layer not in col:
                    continue
            temp_dict = {col:temp_df.ix[input_rows,col].dropna().to_dict()}
            wb_influx_dict.update(temp_dict)
        
        wb_df = pd.DataFrame(wb_influx_dict)
        max_count = wb_df.describe().ix['count',:].max()
        
        # Organize into list
        out_data = {}
        for wb_key in wb_influx_dict.keys():
            temp_wb_dict = wb_influx_dict[wb_key]
            data_array = np.array(temp_wb_dict.values())
            nmatches = data_array.shape[0]
            data_array = np.hstack([np.sum(data_array),nmatches,data_array,np.zeros(int(max_count-nmatches))])
            feature_id_array = np.array(temp_wb_dict.keys())
            feature_id_array = np.hstack(['Qm3day','N_ws',feature_id_array,['-9999']*int(max_count-nmatches)])
            out_data.update({wb_key:{'attr':feature_id_array,'value':data_array}})
        
    return wb_influx_dict,out_data

def save_cgw_flux(out_data,wb_data,fname=None,save_shp=False,ndecimal=4,
                  id_col_name='CAF_FID',n_ws=None):
  
    key_ids = [int(ikey.split('_')[0][1:]) for ikey in out_data.keys()]
    polys,inds,ids = wb_data
    output_data = []
    out_polys = []
    col_headers=['Qm3day','N_ws']
    
    if n_ws is None:
        n_ws = len(out_data[out_data.keys()[0]]['value'])-len(col_headers)
        
    # Collect data
    for data_entry,key_id in zip(out_data.keys(),key_ids):
        temp_dict = out_data[data_entry]
        
        list_value_pairs = [None]*(len(temp_dict['attr'])+len(temp_dict['value']))
        # Update catchment values
        attr_temp = temp_dict['attr'].copy()
        new_vals = []
        for val in attr_temp:
            if isinstance(val,str):
                if 'lay' in val:
                    new_vals.append(int(val.split('_')[0][1:]))
                elif '-9999' in [val]:
                    new_vals.append(int(val))
                else:
                    new_vals.append(val)
            else:
                new_vals.append(val)
        list_value_pairs[::2] = new_vals
        list_value_pairs[1::2] = temp_dict['value']
        for iattr in col_headers:
            if iattr in list_value_pairs:
                list_value_pairs.remove(iattr)
                
        list_value_pairs.insert(0,key_id) # add wb id as column        
        output_data.append(list_value_pairs)
        
        if save_shp:
            # find polygon for matching df_id
            i_match = (ids==key_id).nonzero()[0]
            
            if len(i_match)>1:
                # make union of shapes
                out_poly = cfu.unary_union([polys[i] for i in i_match])
            else:
                out_poly = polys[i_match[0]]
                
            out_polys.append(out_poly)
            
    if id_col_name not in col_headers:    
        col_headers.insert(0,id_col_name)
    # Make col_headers for ws's
    for i in range(n_ws):
        col_headers.extend(['fid_{}'.format(i),'ws_q{}'.format(i)])
    
    if save_shp:
        field_dict = {}
        
        for col_header in col_headers:
            if 'g_id' in col_header.lower():
                field_dict.update({col_header:{'fieldType':"N",'size':19,'decimal':0}})
            elif col_header.lower() in ['n_ws']:
                field_dict.update({col_header:{'fieldType':"N",'size':19,'decimal':0}})
            else:
                field_dict.update({col_header:{'fieldType':"N",'size':19,'decimal':ndecimal}})
                   
        write_shp_dict = {'out_fname':'{}.shp'.format(fname),'field_dict':field_dict,
                          'col_name_order':col_headers,'data':output_data,
                          'polys':out_polys,'write_prj_file':True}    
        cfu.write_shp(**write_shp_dict)
        return col_headers, output_data,field_dict,out_polys
    else:
        out_name = '{}.csv'.format(fname)
        
        with open(out_name,'w') as fout:
            fout.write('{}\n'.format(','.join(col_headers)))
            for rowi in np.array(output_data).astype(str):
                fout.write('{}\n'.format(','.join(rowi)))
            fout.close()
        return col_headers, output_data,[],[]
    

def caf_discharge_save(in_df,all_obj_list,fname=None,q_col_name='ActiveAll',
                       save_shp=False,ndecimal=4,out_layer=0):
    '''Waterbody-centric discharge from all active cells. ***Depricated***
    '''
    df_caf_ids = in_df.columns.values
    # strip first letter (W or L for water or land) and layer info
    df_caf_ids = [[temp,int(temp.split('_')[0][1:])] for temp in df_caf_ids if test_int(temp.split('_')[0][1:]) and temp.split('_lay')[-1] in (str(out_layer))]
    
    polys,inds,ids = all_obj_list
    out_polys = []
    out_data = []
    for df_id,caf_id in df_caf_ids:
        if save_shp:
            # find polygon for matching df_id
            i_match = (ids==caf_id).nonzero()[0]
            
            if len(i_match)>1:
                # make union of shapes
                out_poly = cfu.unary_union([polys[i] for i in i_match])
            else:
                out_poly = polys[i_match[0]]
                
            out_polys.append(out_poly)    
        
        # calculate discharge from active cells to zone
        q_to_caf = np.around(in_df.ix[q_col_name,df_id].sum(),decimals=ndecimal)
    
        
        out_data.append([caf_id,q_to_caf])
    
    if save_shp:    
        field_dict = {'UNIQUE':{'fieldType':"N",'size':19,'decimal':0},
                    'Q_m3day':{'fieldType':"N",'size':19,'decimal':ndecimal}}
        
        write_shp_dict = {'out_fname':'{}.shp'.format(fname),'field_dict':field_dict,
                          'col_name_order':['UNIQUE','Q_m3day']}    
        cfu.write_shp(out_polys,out_data,**write_shp_dict)
    else:
        out_name = '{}.csv'.format(fname)
        
        with open(out_name,'w') as fout:
            fout.write('UNIQUE,Q_m3day\n')
            for rowi in out_data:
                fout.write('{},{}\n'.format(rowi[0],rowi[1]))
            fout.close()
            
        
    

    
    
    
    
    
    
    
    
    
    
    