# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:58:59 2016

@author: kbefus
"""
from __future__ import print_function
import os,warnings
import numpy as np

import matplotlib.pyplot as plt
from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_raster_utils as cru
import flopy.plot as fplt
import flopy.utils as fu
import flopy.modflow as mf

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

def make_zbot(ztop,grid_dis,delv,zthick_elev_min=None):
    '''Make cell bottom elevation array.
    '''
    nlay,nrow,ncol = grid_dis
    if ~isinstance(delv,(list,np.ndarray)):
        delv_array = delv*np.ones(nlay) # m
        delv_cumulative=np.cumsum(delv_array)    
    else:
        delv_cumulative=np.cumsum(delv)

    zbot = [] # for multiple layers
    for ilay in np.arange(nlay):
        zbot.append(ztop-delv_cumulative[ilay])
    
    zbot = np.array(zbot)
    zbot = zbot.reshape(np.hstack([nlay,ztop.shape]))# mainly for nlay=1
    
    if zthick_elev_min is not None:
        zbot = adj_zbot(zbot,ztop,zthick_elev_min) # ensure no negative thickness cells
    
    return zbot

def cull_layers(in_array=None,isthick=True):
    if len(in_array.shape)==3:
        min_max_list=[]
        for ilay in range(in_array.shape[0]):
            minval = np.nanmin(in_array[ilay])
            maxval = np.nanmax(in_array[ilay])
            min_max_list.append([minval,maxval])
            
        mm_array = np.array(min_max_list)
        mdif=np.diff(mm_array,axis=1)
        if isthick:
            mdif[np.isnan(mdif)] = 0
            find_novals = (np.cumsum(mdif[::-1])>0.)[::-1].nonzero()[0][-1]
            out_array = in_array[:find_novals+1]
        else:
            out_lays = ((mdif>=0.) & (mm_array[:,1]>0)).nonzero()[0]
            out_array = in_array[np.arange(out_lays[0],out_lays[-1]+1)]
        return out_array,np.arange(out_array.shape[0])
            
def adj_zbot(zbot,ztop,zthick_elev_min=None):
    '''Adjust cell bottom elevations.
    '''
    if zthick_elev_min is not None: 
        if zthick_elev_min > 0:
            # Minimum thickness
            zbot[-1,:,:] = np.minimum(ztop-zthick_elev_min,zbot[-1,:,:])
        else:
            # Minimum elevation
            zbot[-1,:,:] = np.minimum(zthick_elev_min,zbot[-1,:,:]) # ensure no negative thickness cells
        
    return zbot

def zbot_topofix(ztop=None, zbot=None,filtsize=(3,3),filtmode='nearest',
                 zadjust=2.,lay2fix=[0]):
    '''Create continuous cells in top layer.
    '''    
    from scipy.ndimage.filters import minimum_filter
    new_zbot = zbot.copy()
    

    if len(zbot.shape)>2:    
        

        for ilay in lay2fix:
            
            # Find minimum topography value within filter kernel
            if ilay==0:
                ztopmin = minimum_filter(ztop,size=filtsize,mode=filtmode)
            else:
                ztopmin = minimum_filter(zbot[ilay-1],size=filtsize,mode=filtmode)
                        
#            new_zbot[ilay,ztopmin<=zbot[ilay]] = ztopmin[ztopmin<=zbot[ilay]]-zadjust
            # Calculate cell-by-cell vertical shift needed
            shift_cells = ztopmin < zbot[ilay]
            deltaz = np.zeros_like(new_zbot[0])
            deltaz[shift_cells] = new_zbot[ilay,shift_cells]-ztopmin[shift_cells]+zadjust
            
            # Where the bottom of a cell is above the minimum cell elevation,
            # assign new bottom elevation, shift all consequent layers down with active layer, ilay
            new_zbot[ilay:] = new_zbot[ilay:]-deltaz         

            # Calculate current layer thicknesses            
            thick_temp = np.diff(np.vstack([new_zbot[::-1],ztop.reshape(np.hstack((1,ztop.shape)))]),axis=0)[::-1]
            
            # Use relative thickness to proportionately adjust thicknesses
            # 1) Calculate relative thickness of each consequent layer
            rel_thick  = thick_temp[ilay+1:]/np.sum(thick_temp[ilay+1:],axis=0)
            
            # 2) Assign zbot shift upward based on relative thickness
            new_zbot[ilay+1:] = new_zbot[ilay+1:] + (rel_thick * deltaz)
            

#            # ---- Not working due to assignment issues------
#            # Accomodate deltaz in highest layer with enough thickness
#            zbot_adjustz = np.zeros_like(new_zbot)
#            for ilay2 in np.arange(ilay+1,zbot.shape[0]):
#                # Select where layer is thick enough and hasn't been assigned already
#                thickness_bool = (zbot_adjustz[:ilay2,shift_cells].sum(axis=0)==0.) & \
#                                 (thick_temp[ilay2,shift_cells] > deltaz)
#                
#                zbot_adjustz[ilay2,shift_cells][thickness_bool] = deltaz[thickness_bool] # <---- fix this
#
#            
#            # Make certain layers thinner based on original thickness
#            new_zbot = new_zbot + zbot_adjustz
            # -----------------------------------------------
    else:
        ztopmin = minimum_filter(ztop,size=filtsize,mode=filtmode)
        new_zbot[ztopmin<=zbot] = ztopmin[ztopmin<=zbot]-zadjust
        

#    # Make sure all layer thicknesses are finite, done in script
#    new_zbot = cru.make_layers_deeper(new_zbot,deeper_amount=zadjust)
    return new_zbot
    

def K_to_layers(ztop,zbot,hk_array,z_array,z_is_thick=True,
                force_top_layer=False,nan_lower=True,propkdown=False,
                last_layer_nan=True):
    '''
    Create hydraulic conductivity layer-based array (though can be any spatially varying parameter).
    
    Parameters
    ----------
    ztop: np.ndarray
        nrow x ncol array of model top elevations
    
    zbot: np.ndarray
        nlay x nrow x ncol array of model bottom elevations
        
    hk_array: np.ndarray
        nKlay interfaces x nrow x ncol of hydraulic conductivity values
        hk_array[0,:,:] is hk for LayerA
    
    z_array: np.ndarray
        nKlay-1 interfaces x nrow x ncol of top elevations or thicknesses of K interfaces
        ztop_k[0,:,:] is interface elevation between LayerA and LayerB
    
    Returns
    --------
    hk_out: np.ndarray
        nlay x nrow x ncol of K values
    '''
    if (zbot.shape==z_array.shape) and z_is_thick:
        hk_out = hk_array.copy()
        hk_out[:,np.isnan(ztop)] = np.nan
        with np.errstate(invalid='ignore'):
            z_array[np.isnan(z_array) | (z_array<0.)] = 0.
        zsum = np.sum(z_array,axis=0)
        hk_out[:,zsum==0.] = np.nan # where all layers have 0 thickness
        if propkdown:
            # assign k for layers with zero thickness (changed to zthickmin) to next nonnan layer down
#            for ilay in range(hk_out.shape[0]-1):
#                hk_out[ilay,z_array[ilay]==0.] = np.nan # reset hk values in active layer
#                # Find first nonzero value moving down section
#                for ilay2 in range(ilay+1,hk_out[ilay+1:].shape[0]):
#                    search_bool = ~np.isnan(hk_out[ilay2]) & (hk_out[ilay2]>0.) &  \
#                                  np.isnan(hk_out[ilay]) & (z_array[ilay2]>0.)
#                    hk_out[ilay,search_bool] = hk_out[ilay2,search_bool].copy() # assign k layer value 
            fill_dict = {'array':hk_out,'threshold_array':z_array,'threshold_value':0.}
            hk_out= fill_nan_layer(**fill_dict)
        if last_layer_nan:
            hk_out[-1,z_array[-1]==0.] = np.nan # last layer w/ no thickness to nan               
        
    else:
        if len(zbot.shape)>2:
            hk_out = np.nan*np.zeros_like(zbot)
        else:
            hk_out = np.nan*np.zeros(np.hstack([1,zbot.shape]))
        if hasattr(ztop,'mask'):
            ztop = np.ma.filled(ztop,fill_value=np.nan)
        active_cells = ~np.isnan(ztop)
        if z_is_thick:
            # Calculate bottom elevations using ztop
            z_array[np.isnan(z_array)]=0.
            z_array[z_array<0.] = 0.
            z_elev= ztop-np.cumsum(z_array,axis=0)
            thickness_array = z_array.copy()
        else:
            z_elev = z_array.copy()
            thickness_array = np.diff(z_array[::-1],axis=0)[::-1]
        
        if len(zbot.shape)>2:
            z_layers = np.vstack([ztop.reshape(np.hstack([1,ztop.shape])),zbot])
            z_centers = zbot + np.diff(z_layers[::-1],axis=0)[::-1]/2.
        else:
            z_layers = np.vstack([ztop.reshape(np.hstack([1,ztop.shape])),zbot.reshape(np.hstack([1,zbot.shape]))])
            z_centers = zbot.reshape(np.hstack([1,zbot.shape])) + np.diff(z_layers[::-1],axis=0)[::-1]/2.
        if z_elev.shape[0] > 1:
            for iklay in list(range(hk_array.shape[0])):
                if iklay-1==-1: # first layer
                    layer_bool = (z_centers>z_elev[iklay]) & active_cells &\
                                    (thickness_array[iklay]!=0.)# above interface
                elif iklay < z_elev.shape[0]:
                    layer_bool = (z_centers>z_elev[iklay]) & (z_centers<=z_elev[iklay-1]) \
                                & (thickness_array[iklay]!=0.) \
                                & active_cells # between interfaces
                else: # last layer
                    layer_bool = (z_centers<z_elev[iklay-1]) & active_cells
                    
                hk_out[layer_bool] = np.tile(hk_array[iklay],(hk_out.shape[0],1,1))[layer_bool]
                
        if force_top_layer:
            hk_out[0,~np.isnan(hk_out[0])] = hk_array[0,~np.isnan(hk_out[0])].copy()
    
        # Assign cells with missing hk to next lower real hk value, if exists
        hk_out[hk_out<=0.] = np.nan
        if propkdown:
            fill_dict = {'array':hk_out,'threshold_array':thickness_array,'threshold_value':0.}
            hk_out= fill_nan_layer(**fill_dict)
        if last_layer_nan:    
            hk_out[-1,z_array[-1]==0.] = np.nan # last layer w/ no thickness to nan 
#            for ilay in range(hk_out.shape[0]-1):
#                # Find first nonzero value moving down section
#                for ilay2 in range(ilay+1,hk_out[ilay+1:].shape[0]):
#                    search_bool = ~np.isnan(hk_out[ilay2]) &  \
#                                  np.isnan(hk_out[ilay]) & (thickness_array[ilay2] > 0.)
#                    hk_out[ilay,search_bool] = hk_out[ilay2,search_bool].copy() # assign k layer value             
                   
    if nan_lower and len(zbot.shape)>2:
        # set to nan values for all layers below a nan value
        for ilay in range(hk_out.shape[0]-1):
            hk_out[ilay:,np.isnan(hk_out[ilay])] = np.nan  
            
    return hk_out

def fill_nan_layer(array=None,direction='down',threshold_array=None, threshold_value = 0.):
    
    array_out = array.copy()
    if direction in ['down']:
        # Assign first finite and nonnan value moving down the layers
        for ilay in range(array_out.shape[0]-1):
            # Look recursively through lower layers
            for ilay_lower in range(ilay+1,array_out[ilay+1:].shape[0]):
                if threshold_array is not None:
                    search_bool = ~np.isnan(array_out[ilay_lower]) &  \
                                      np.isnan(array_out[ilay]) & \
                                      (threshold_array[ilay_lower] > threshold_value)
                else:
                    search_bool = ~np.isnan(array_out[ilay_lower]) &  \
                                  np.isnan(array_out[ilay])
                # Set values found                  
                array_out[ilay,search_bool] = array_out[ilay_lower,search_bool].copy()
    
    return array_out
                  

def clean_ibound(ibound,min_area=None,check_inactive=False):
    '''
    Removes isolated active cells from the IBOUND array.
    
    Assumes only active and inactive ibound conditions (i.e., no constant heads).
    
    Source: modified after Wesley Zell, PyModflow.pygrid.grid_util.clean_ibound, Jul 17 2013
    '''
    from scipy.ndimage import measurements
    # Distinguish disconnected clusters of active cells in the IBOUND array.
    cluster_ibound = ibound.copy()
    cluster_ibound[ibound != 0] = 1
    
    
    array_of_cluster_idx,num = measurements.label(cluster_ibound)
    
    # Identify the cluster with the most active cells; this is the main active area
    areas = measurements.sum(cluster_ibound,array_of_cluster_idx,\
                             index=np.arange(array_of_cluster_idx.max()+1))
    
    clean_ibound_array = np.zeros_like(ibound)                         
    if (min_area is None):
        # Use only largest area
        cluster_idx = np.argmax(areas)
        # Activate all cells that belong to primary clusters
        clean_ibound_array[array_of_cluster_idx == cluster_idx] = 1
    else:
        cluster_idx = (areas >= min_area).nonzero()[0]
        
        # Activate all cells that belong to primary clusters
        for idx_active in cluster_idx:
            clean_ibound_array[array_of_cluster_idx==idx_active] = 1

    if check_inactive:
        # Identify inactive clusters surrounded by active cells
        
        cluster_ibound2 = 1-clean_ibound_array.copy() # Flip values
        clean_ibound_array2 = clean_ibound(cluster_ibound2,min_area=min_area)
        clean_ibound_array[clean_ibound_array2==1] = 0
        clean_ibound_array[clean_ibound_array2==0] = 1
        
        
    return clean_ibound_array    

def calc_vcont(vk=None,zbot=None,ztop=None,confining_bed=False,
               dvcb=None,vkcb=None):
    '''Calculate vertical conductance of model layers from vk.
    '''
    # Calculate vertical cell thicknesses
    z_all = np.vstack([ztop.reshape(np.hstack((1,ztop.shape))),zbot])                                                   
    zthick = np.diff(z_all[::-1,:,:],axis=0) # nlay,nrow,ncol array
                                                   
    if confining_bed:
        vcont = []
        for ilay in list(range(zthick.shape[0]-1)):
            vcont.append(cgu.vcont_func(zthick[ilay,:,:].squeeze(),
                                    zthick[ilay+1,:,:].squeeze(),
                                    vk[ilay,:,:].squeeze(),
                                    vk[ilay,:,:].squeeze(),
                                    vkcb=vkcb[ilay],
                                    dvcb=dvcb[ilay]))
    else:
        vcont = []
        for ilay in list(range(zthick.shape[0]-1)):
            vcont.append(cgu.vcont_func(zthick[ilay,:,:].squeeze(),
                                    zthick[ilay+1,:,:].squeeze(),
                                    vk[ilay,:,:].squeeze(),
                                    vk[ilay,:,:].squeeze()))
    return np.array(vcont) # nlay-1,nrow,ncol array

def make_K_arrays(array_dims,hk,hk_in=None,hk_in_botm=None,vk_array=None,
                  ztop=None,zbot=None,k_decay=None,
                  v_ani_ratio = 1.,h_ani_ratio=1.,
                  conversion_mask=None, elev_array=None,
                  z_is_thick=True, calc_vcont_flag=False,
                  thin_layer_m=2.,nan_lower=True,propkdown=False,
                  last_layer_nan=True):
    '''Create input hydraulic conductivity data for flow packages.
    '''
    nlay,nrow,ncol = array_dims
    # --------- Horizontal hydraulic conductivity ---------
    if (hk_in is not None) and (hk_in_botm is not None):
        if conversion_mask is not None:
            hk_in = cgu.shrink_ndarray(hk_in,conversion_mask,array_dims)
            hk_in_botm = cgu.shrink_ndarray(hk_in_botm,conversion_mask,array_dims)
        
        hk = K_to_layers(elev_array,zbot,hk_in,hk_in_botm,
                                   z_is_thick=z_is_thick,nan_lower=nan_lower,
                                   propkdown=propkdown,last_layer_nan=last_layer_nan)
    elif (hk is not None) and (hk_in is None):
        # hk supplied separately
        if isinstance(hk,(float,int)):
            # make hk into array
            hk = hk*np.ones_like(zbot)
        elif isinstance(hk,np.ndarray):
            if len(hk.shape)==2:
                # multiply for each layer
                hk = np.tile(hk,[zbot.shape[0],1,1])
    
    if (k_decay is not None):
        if hk.shape != array_dims:
            array_dims = hk.shape # shape change due to re-run/crash
        k_decay = cgu.to_nD(k_decay,array_dims)
        hk = k_decay*cgu.to_nD(hk,array_dims)
    
    hk = cgu.fill_mask(hk)
    
    # --------- Vertical hydraulic conductivity ---------
    if (vk_array is None):
        # Use hk_array and vertical anisotropy if vk not provided
        v_ani_array = cgu.to_nD(v_ani_ratio,array_dims)
        vk = hk/v_ani_array

    else:
        vk = cgu.to_nD(vk_array,array_dims)
    
    # Remove anisotropy for thin top layer
    #vk[0,(ztop-zbot[0])<=thin_layer_m] = hk[0,(ztop-zbot[0]) <= thin_layer_m]    
    
    if calc_vcont_flag:
        # Calculate vertical conductance from vk_array
        vcont = calc_vcont(vk,zbot=zbot,ztop=ztop)     
    else:
        vcont = np.nan
        
    return hk,vk,vcont

def outer_ghb(ztop=None,zbot=None,zthick=None,cell_types=None,cell_spacing=None,
              dampener=1.,sea_level=0,hk=None,inactive_val=0,max_lay=None,cut_off=1e-2,
              marine_bool=False):
    out_ghb_array = None
    zthick2 = np.diff(np.vstack([zbot[::-1],ztop.reshape(np.hstack((1,ztop.shape)))]),axis=0)[::-1]
    cc_elev = zbot+zthick2/2.
    if len(zbot.shape)==3:
        if max_lay is not None:
            ilayers = max_lay
        else:
            ilayers = zbot.shape[0]
         
        ghb_bounds =  cru.raster_edge(cell_types=cell_types,search_val=-2,zsize=20,
                                              min_area=1)
        rc_inds_top = np.array(ghb_bounds.nonzero()).T
        ghb_rc = None    
        for ilay in np.arange(1,ilayers):
            active_zthick_vals = zthick[ilay,rc_inds_top[:,0],rc_inds_top[:,1]]
            # Use layer thickness to calculate largest gradient area
            mag_array,orient_array = calc_gradient(zthick[ilay],cell_spacing)
            mag_cluster = cru.cluster_array(mag_array).astype(bool)
            
            if len(mag_cluster.nonzero()[0])!=0 and len(mag_cluster.nonzero()[0])<mag_cluster.size and not marine_bool:
                # Check for boundary cells
                non_ghb_bounds =  cru.raster_edge(cell_types=cell_types,search_val=1,min_area=1)
                
                # Gather row, col indexes for creating ghb array
                ghb_rc = np.array((mag_cluster & ~non_ghb_bounds &\
                                   (cc_elev[ilay]<0.) & (cell_types!=inactive_val)).nonzero()).T
                
                if ghb_rc.shape[0] > 0:

#                    cgu.quick_plot((mag_cluster & ~non_ghb_bounds & (cell_types!=inactive_val)))
                    ghb_in_dict = {'rc_inds':ghb_rc,'zthick':zthick2[ilay],'cc_elev':cc_elev[ilay],
                                   'dampener':dampener,'sea_level':sea_level,'k_array':hk[ilay],
                                   'cell_size':cell_spacing,'ilay':ilay}
                    layer_ghb_array = make_ghb_array(**ghb_in_dict)
                    if out_ghb_array is None:
                        out_ghb_array = layer_ghb_array.copy()
                    else:
                        out_ghb_array = np.vstack([out_ghb_array,layer_ghb_array])
                if len((~np.isnan(active_zthick_vals)).nonzero()[0])>0:
                    ghb_rc=None
                    # Try assigning layers below edge ghb to ghb
                    ghb_in_dict = {'rc_inds':rc_inds_top,'zthick':zthick2[ilay],'cc_elev':cc_elev[ilay],
                                       'dampener':dampener,'sea_level':sea_level,'k_array':hk[ilay],
                                       'cell_size':cell_spacing,'ilay':ilay}
                    layer_ghb_array = make_ghb_array(**ghb_in_dict)
                    if out_ghb_array is None:
                        out_ghb_array = layer_ghb_array.copy()
                    else:
                        # remove duplicates
                        temp_lrc=np.vstack([layer_ghb_array[:,:3],out_ghb_array[:,:3]])
                        u1,uind,uinv = cgu.unique_rows(temp_lrc,sort=False,return_inverse=True)
                        uind = uind[uind<layer_ghb_array.shape[0]]
                        out_ghb_array = np.vstack([out_ghb_array,layer_ghb_array[uind,:]])
            elif len((~np.isnan(active_zthick_vals)).nonzero()[0])>0:
                ghb_rc=None
                # Try assigning layers below edge ghb to ghb
                ghb_in_dict = {'rc_inds':rc_inds_top,'zthick':zthick2[ilay],'cc_elev':cc_elev[ilay],
                                   'dampener':dampener,'sea_level':sea_level,'k_array':hk[ilay],
                                   'cell_size':cell_spacing,'ilay':ilay}
                layer_ghb_array = make_ghb_array(**ghb_in_dict)
                if out_ghb_array is None:
                    out_ghb_array = layer_ghb_array.copy()
                else:
                    out_ghb_array = np.vstack([out_ghb_array,layer_ghb_array])
                
        if max_lay is not None:
            # Assign layers below max_lay to have the same ghb
            for ilay in np.arange(max_lay+1,zbot.shape[0]):
                if isinstance(ghb_rc,np.ndarray):

#                    cgu.quick_plot((mag_cluster & ~non_ghb_bounds & (cell_types!=inactive_val)))
                    ghb_in_dict = {'rc_inds':ghb_rc,'zthick':zthick2[ilay],'cc_elev':cc_elev[ilay],
                                   'dampener':dampener,'sea_level':sea_level,'k_array':hk[ilay],
                                   'cell_size':cell_spacing,'ilay':ilay}
                    layer_ghb_array = make_ghb_array(**ghb_in_dict)
                else:
                    ghb_in_dict = {'rc_inds':rc_inds_top,'zthick':zthick2[ilay],'cc_elev':cc_elev[ilay],
                                   'dampener':dampener,'sea_level':sea_level,'k_array':hk[ilay],
                                   'cell_size':cell_spacing,'ilay':ilay}
                    layer_ghb_array = make_ghb_array(**ghb_in_dict)
                
                out_ghb_array = np.vstack([out_ghb_array,layer_ghb_array])
                  
        return out_ghb_array
    
    else:
        print("Error: Not made for single layers")
        return
    
def make_ghb_array(rc_inds=None,zthick=None,cc_elev=None,
                   sea_level=0,k_array=None,cell_size=None,dampener=1.,ilay=None):
    
    active_hk_vals = k_array[rc_inds[:,0],rc_inds[:,1]]
    # Cells have hk values, continue with assignment
    water_height = sea_level-cc_elev[rc_inds[:,0],rc_inds[:,1]]
    fw_head = cgu.calc_fw_head(water_height,cc_elev[rc_inds[:,0],rc_inds[:,1]])
    cond = active_hk_vals*(cell_size*zthick[rc_inds[:,0],rc_inds[:,1]])/cell_size

    # Check for negative freshwater heads - indicates incorrect assignment
    neg_fwhead = fw_head<0.

    # Stack output arrays and create entries for layer
    out_ghb_array = np.hstack([ilay*np.ones_like(active_hk_vals[~neg_fwhead].reshape((-1,1))),
                               rc_inds[~neg_fwhead,0].reshape((-1,1)),
                               rc_inds[~neg_fwhead,1].reshape((-1,1)),
                               fw_head[~neg_fwhead].reshape((-1,1)),
                               dampener*cond[~neg_fwhead].reshape((-1,1))])

    return out_ghb_array    
    
    
    
#def outer_ghb_old(hk=None, ztop=None,zbot=None,cell_types=None,
#              min_area=None,search_val=-2,invalid_val=0,
#              size=3,zsize=3,sea_level=0,cell_size=None,dampener=1.):
#    
#    ilay_start = 0
#    out_ghb_array = []
#    zthick = np.diff(np.vstack([zbot[::-1],ztop.reshape(np.hstack((1,ztop.shape)))]),axis=0)[::-1]
#    cc_elev = zbot+zthick/2.
#    
#    outer_dict = {'cell_types':cell_types,'search_val':search_val,
#                  'invalid_val':invalid_val,'size':size,'zsize':zsize,
#                  'min_area':min_area}
#    
#    while ilay_start <= hk.shape[0]:
#        
#        bound_dict = {'outer_dict':outer_dict,'cell_size':cell_size,
#                      'sea_level':sea_level,'k_array':hk,
#                      'zthick':zthick,'cc_elev':cc_elev,'start_layer':ilay_start,
#                      'out_ghb_array':out_ghb_array,'dampener':dampener}
#    
#        out_ghb_array_it, ilay_out = calc_ghb_bounds(**bound_dict)
#        out_ghb_array = np.vstack([out_ghb_array,out_ghb_array_it])
#    
#
#    return out_ghb_array   
#
#
#
#def calc_ghb_bounds(outer_dict=None,cell_size=None,sea_level=0,k_array=None,
#                    zthick=None,cc_elev=None,start_layer=0,out_ghb_array=[],
#                    dampener=1.):
#    
#    outer_bool = cru.raster_edge(**outer_dict)
#    rc_inds = np.array(outer_bool.nonzero()).T
#    
#    for ilay in np.arange(start_layer+1,k_array.shape[0]):
#        active_hk_vals = k_array[ilay,rc_inds[:,0],rc_inds[:,1]]
#        if len(~np.isnan(active_hk_vals).nonzero()[0])>0:
#            # Cells have hk values, continue with assignment
#            water_height = sea_level-cc_elev[ilay,rc_inds[:,0],rc_inds[:,1]]
#            fw_head = cgu.calc_fw_head(water_height,cc_elev[ilay,rc_inds[:,0],rc_inds[:,1]])
#            cond = active_hk_vals*(cell_size*zthick[ilay,rc_inds[:,0],rc_inds[:,1]])/cell_size
#            # Stack output arrays and create entries for layer
#            out_ghb_array = np.vstack([out_ghb_array,
#                                       np.hstack([ilay*np.ones((np.active_hk_vals.shape[0],1)),
#                                       rc_inds[:,0].reshape((-1,1)),
#                                       rc_inds[:,1].reshape((-1,1)),
#                                       fw_head,
#                                       dampener*cond.reshape((-1,1))])])
#        else:
#            # No active cells found for layer, do analysis over with different constraints
#            start_layer = ilay
#            continue
#    else:
#        # loop completed succesfully
#        start_layer = ilay+1
#            
#    return out_ghb_array,start_layer

def shrink_domain(dicts_obj,XY=None):
    '''Remove rows and columns with only inactive cells.
    '''
    cell_types_array = dicts_obj.cell_types.copy()
    cell_types_array[cell_types_array==cgu.grid_type_dict['inactive']] = np.nan # set inactive cells to nan
    if (XY is None):
        x=np.arange(0,cell_types_array.shape[1],dtype=float)
        y=np.arange(0,cell_types_array.shape[0],dtype=float)
        XY = np.meshgrid(x,y)
        
    newX,newY,new_cell_types,old2new_mask = cru.remove_nan_rc(XY[0],XY[1],cell_types_array,return_mask=True)
    dicts_obj.dis_obj.nrow,dicts_obj.dis_obj.ncol = newX.shape
    new_cell_types[np.isnan(new_cell_types)] = cgu.grid_type_dict['inactive'] # reset nan's to inactive
    dicts_obj.cell_types = new_cell_types
    return old2new_mask            

def change_dir(in_fname,new_outdir,old_outdir=None):
    """
    Helper function to save model outputs in separate folder.
    """
    fname = os.path.basename(in_fname)
       
    # if know old_outdir, use relative paths
    if old_outdir is not None:     
        out_fname = os.path.join(os.path.relpath(new_outdir,old_outdir),fname)
    else:
        out_fname = os.path.join(new_outdir,fname)
    return out_fname
        
    
# ---- Output functions for flopy model results ----    
def calc_watertable(h_array):
    '''Extract head from first non-null layer of the model.
    '''
    nlay,nrow,ncol = h_array.shape
    newmat = np.nan*np.zeros((nrow,ncol))
    laymat = np.nan*np.zeros((nrow,ncol))
    itrue = True
    ilay_count = 0
    
    while itrue:
        cond1 = ((~np.isnan(h_array[ilay_count,:,:])) & (np.isnan(newmat)))
        newmat[cond1] = h_array[ilay_count,cond1]
        laymat[cond1] = ilay_count
        
        if len((np.isnan(newmat)==True).nonzero()[0]) > 1:
            # More nan values to fill
            ilay_count += 1
        else:
            itrue=False
        
        if ilay_count == nlay:
            # All layers checked
            itrue = False
    
    return newmat,laymat

def load_hds(model_name=None,workspace=None,inactive_head=np.nan,
             time2get=-1,min_head=-998.,info_array=None, calc_wt=False,
             ext='hds'):
    ''' Load groundwater head elevations from Modflow output.
    '''
    
    # Create the headfile object and load head
    headobj = fu.binaryfile.HeadFile(os.path.join(workspace,'{}.{}'.format(model_name,ext)))
    times = headobj.get_times()
    head = np.squeeze(headobj.get_data(totim=times[time2get]))
    headobj.close()
    
    if info_array is not None:  
    
        if len(head.shape)>2:
            head[:,info_array==cgu.grid_type_dict['inactive']] = inactive_head
        else:
            head[info_array==cgu.grid_type_dict['inactive']] = inactive_head
            
    head[head<min_head] = inactive_head
    
    if calc_wt:
        if len(head.shape)==3:
            head_out,wt_layer_mat = calc_watertable(head)
        else:
            head_out=head.copy()
            wt_layer_mat = np.zeros_like(head)
    
    else: 
        wt_layer_mat = np.nan
        head_out = head.copy()
        
    return head_out,wt_layer_mat

def save_head_geotiff(load_hds_dict=None, ref_dict=None, 
                      XY=None,dxdy=None,ilay=0,active_cells=None,
                      elevation=None,save_all=False):

    head_all,wt_layer = load_hds(**load_hds_dict)   
    proj = ref_dict['proj']
    x,y=XY
    out_tifs=[]
    if ref_dict['rotation']<(-np.pi/2.):
        x,y,head_all = x[::-1,::-1],y[::-1,::-1],head_all[::-1,::-1]
        if active_cells is not None:
            active_cells = active_cells[::-1,::-1]
        if elevation is not None:
            elevation  = elevation[::-1,::-1]

    outfile = os.path.join(load_hds_dict['workspace'],'{}_head.tif'.format(load_hds_dict['model_name']))
    out_tifs.append(outfile)
    if active_cells is not None:
        with np.errstate(invalid='ignore'):
            head_all[active_cells==0] = np.nan
            
    cru.write_gdaltif(outfile,x,y,head_all, proj_wkt=proj) # rot_xy=grid_rot,
    if save_all:
        # Save cell_types
        out_ct = os.path.join(load_hds_dict['workspace'],'{}_celltypes.tif'.format(load_hds_dict['model_name']))
        out_tifs.append(out_ct)
        cru.write_gdaltif(out_ct,x,y,active_cells, proj_wkt=proj)
        if elevation is not None:
            # Save water table depth
            out_wt = os.path.join(load_hds_dict['workspace'],'{}_wtdepth.tif'.format(load_hds_dict['model_name'])) 
            out_tifs.append(out_wt)
            cru.write_gdaltif(out_wt,x,y,elevation-head_all, proj_wkt=proj)
            
    return out_tifs
    
def load_zeta(model_name=None,workspace=None,dicts_obj=None,fill_val=np.nan,tstep=-1):
    ''' Load zeta elevations from SWI2 output.
    '''
    zetafile = os.path.join(workspace, '{}.zta'.format(model_name))
    zobj = fu.CellBudgetFile(zetafile)
    zkstpkper = zobj.get_kstpkper()
    zeta = zobj.get_data(kstpkper=zkstpkper[tstep], text='      ZETASRF  1')[0]
    zeta = np.squeeze(zeta)
    if dicts_obj is not None:
        info_array = dicts_obj.cell_types
        if len(zeta.shape)==3:
            zeta[:,info_array==cgu.grid_type_dict['inactive']] = fill_val
        else:
            zeta[info_array==cgu.grid_type_dict['inactive']] = fill_val
    
    zobj.close()
    return zeta
    
def calc_swi(zeta,zbot=None,min_dz = 1e-1,set_floor_nan=True):
    '''Calculate seawater interface (swi) from zeta and layers.
    '''
    swi_out = zbot[-1].copy()
    if len(zeta.shape)==2:
        zeta = np.expand_dims(zeta,0) # for one layer models
        
    for zeta_layer,zbot_layer in zip(zeta[::-1],zbot[::-1]):
        bool_test = np.abs(zeta_layer-zbot_layer)>min_dz
        swi_out[bool_test] = zeta_layer[bool_test]
        
    if set_floor_nan:
        swi_out[swi_out==zbot[-1]] = np.nan
        swi_out[swi_out>0] = np.nan
    return swi_out

def load_list_budget(model_name=None,workspace=None,to_df=True,
                     time_to_get=-1,use_index=True,start_datetime=None):
    fname = os.path.join(workspace,'{}.list'.format(model_name))
    Mflist_obj=fu.MfListBudget(fname)
    if to_df and Mflist_obj.isvalid():
        df_inc,df_cum = Mflist_obj.get_dataframes(start_datetime=start_datetime)
        tsteps= Mflist_obj.get_times()
        if use_index:
            time_to_get = tsteps[time_to_get]
        
        df = df_cum.loc[time_to_get]
        df.name = model_name
        df['convergence'] = list_convergence(fname)
        return df
    elif Mflist_obj.isvalid() and not to_df:
        return Mflist_obj
    else:
        return None

def list_convergence(list_fpath,fail_key = r'****FAILED'):
    convergence_flag = True
    with open(list_fpath,'r') as f_in:
        for line in f_in:
            if fail_key in line:
                convergence_flag=False
                continue
    return convergence_flag          
        
        
def load_model(modelname=None,workspace=None,load_only=None,namefile=None):
    if namefile is None:
        namefile = '{}.nam'.format(modelname)
        
    m = mf.Modflow.load(namefile,model_ws=workspace,
                        load_only=load_only)
    return m
    
def cgw_plot(model_name=None,workspace=None,domain_obj=None,dicts_obj=None,
                         run_swi=False,head_cont_int = None,
                         save_fig=False,save_fig_fpath=None,
                         last_array = None,cbar_label_name=None,
                         head_vlims=[0.,50.],wtdepth_vlims=[0.,50.],topomax=20,
                         swi_depth=False,cc_type='ll',axis_bg_color = [.95,.95,.95]):
    '''Plot modflow results as multipanel figure.
    '''    
      
    info_array = dicts_obj.cell_types

    # Load head data
    head_consolidated,wt_layer_mat = load_hds(model_name=model_name,
                                              workspace=workspace,
                                              info_array=info_array, calc_wt=True)
    head_masked = np.ma.masked_invalid(head_consolidated)
    
    # Elevation
    elev_array = domain_obj.elevation.copy()
    elev_array[info_array==cgu.grid_type_dict['inactive']] = np.nan
    ztop_ma = np.ma.masked_invalid(elev_array)
    
    # X,Y
    if cc_type in ['ll']:
        ccx,ccy =  domain_obj.cc_ll
        xlabel = 'Longitude'
        ylabel = 'Latitude'
#        ccx=np.ma.masked_array(ccx,mask=np.isnan(elev_array))
#        ccy=np.ma.masked_array(ccy,mask=np.isnan(elev_array))
    elif cc_type in ['proj']:
        ccx,ccy =  domain_obj.cc_proj
        xlabel = 'Model projected x [m]'
        ylabel = 'Model projected y [m]'
#        ccx=np.ma.masked_array(ccx,mask=np.isnan(elev_array))
#        ccy=np.ma.masked_array(ccy,mask=np.isnan(elev_array))
    else:
        ccx,ccy =  domain_obj.cc
        ccx,ccy = ccx/1e3,ccy/1e3
        xlabel = 'Model x [km]'
        ylabel = 'Model y [km]'
#        ccx=np.ma.masked_array(ccx/1e3,mask=np.isnan(elev_array))-np.tile(ccx[:,:1]/1e3,(1,ccx.shape[1]))
#        ccy=np.ma.masked_array(ccy/1e3,mask=np.isnan(elev_array))-np.tile(ccy[:1,:]/1e3,(ccy.shape[0],1))
        
    
    
    # Water table depth
    wt_depth = ztop_ma-head_masked
    
    subplot_array = [141,142,143,144]   
    # read swi results
    if run_swi:

        zeta = load_zeta(model_name=model_name,workspace=workspace,
                         dicts_obj=dicts_obj,fill_val=np.nan)
        zeta_interface = calc_swi(zeta,zbot=dicts_obj.zbot)
    
    # Elevation
    active_cmap = plt.cm.get_cmap('PuBu')
#    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(subplot_array[0])
    im1=ax.pcolormesh(ccx,ccy,ztop_ma,vmin=0,vmax=topomax,edgecolor='none',cmap=plt.cm.terrain)
    ax.set_title('Topography')
    ax.set_facecolor(axis_bg_color)
#    ax.set_xlim(np.min(ccx),np.max(ccx))
#    ax.set_ylim(np.min(ccy),np.max(ccy))
    ax.set(aspect=1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(visible=False)
    cb1=plt.colorbar(im1,ax=ax,orientation='horizontal',extend='both',
                     ticks=np.linspace(0,topomax,6))
    cb1.ax.set_xlabel('Elevation [m]')    
    
    # Water table elevation
    ax2 = fig.add_subplot(subplot_array[1],sharex=ax,sharey=ax)
    if (head_cont_int is None):
        contour_interval = 10.
    else:
        contour_interval = head_cont_int
    im2=ax2.contourf(ccx,ccy,np.ma.masked_array(head_consolidated,mask=info_array==cgu.grid_type_dict['nearshore']),
                    np.arange(head_vlims[0],head_vlims[1]+contour_interval/5.,contour_interval/5.),extend='max',
                    cmap=active_cmap)
    if (head_cont_int is not None):                
        ax2.contour(ccx,ccy,head_masked,np.arange(head_vlims[0],head_vlims[1]+contour_interval,contour_interval),colors='grey')
    ax2.set_title('Head')
    ax2.set_facecolor(axis_bg_color)
    ax2.set_xlabel(xlabel)
    ax2.set(aspect=1)
    cb2=plt.colorbar(im2,ax=ax2,orientation='horizontal',extend='both',
                     ticks=np.linspace(head_vlims[0],head_vlims[1],6))
    cb2.ax.set_xlabel('Water table elevation [m]')

    # Water table depth
    ax3 = fig.add_subplot(subplot_array[2],sharex=ax,sharey=ax)
    im3=ax3.pcolormesh(ccx,ccy,np.ma.masked_array(wt_depth,mask=info_array==cgu.grid_type_dict['nearshore']),
                      vmin=wtdepth_vlims[0],vmax = wtdepth_vlims[1],
                      edgecolor='none',cmap=active_cmap)
    ax3.set_title('Water table depth')
    ax3.set_facecolor(axis_bg_color)
    ax3.set_xlabel(xlabel)
    ax3.set(aspect=1)
    cb3a=plt.colorbar(im3,ax=ax3,orientation='horizontal',extend='max',
                      ticks=np.linspace(wtdepth_vlims[0],wtdepth_vlims[1],6))
    cb3a.ax.set_xlabel('Water table depth [m]')

    if run_swi:   
        ax4 = fig.add_subplot(subplot_array[3],sharex=ax,sharey=ax)

        if swi_depth:
            zeta_out_2 = ztop_ma-zeta_interface
            
            swix_label = 'Swi depth [m]'
        else:
            zeta_out_2 = zeta_interface
            swix_label = 'Swi elevation [m]'
        im0=ax4.pcolormesh(ccx,ccy,np.ma.masked_invalid(zeta_out_2),cmap=plt.cm.RdBu)

        ax4.set_title('Saltwater interface')
        ax4.set_axis_bgcolor(axis_bg_color)
        cb4 = plt.colorbar(im0,ax=ax4,orientation='horizontal')
        cb4.ax.set_xlabel(swix_label)
        ax4.set_xlabel(xlabel)
        if save_fig:
            model_daspect = head_consolidated.shape[0]/float(head_consolidated.shape[1])
            x_inch = 10.
            y_inch = 1.3+(x_inch*model_daspect)/4.
            fig_dict = {'dpi':300,'format':'png',
                        }
            fig.set_size_inches(x_inch, y_inch,forward=True)
            fig.savefig(save_fig_fpath,**fig_dict)
            plt.close('all')
        return head_consolidated, wt_layer_mat,zeta_out_2
    else:
        if (last_array is None):
            last_array = info_array.copy()
            last_array[np.isnan(last_array)]= 0.
            
            cbar_label_name = 'Cell type'
        ax4 = fig.add_subplot(subplot_array[3],sharex=ax,sharey=ax)
        
        # Collect unique values of info_array        
        unq_vals,unq_inv = np.unique(last_array,return_inverse=True)
        from matplotlib.colors import BoundaryNorm
        
        cmap=plt.cm.brg
        unq2 = np.unique(unq_inv)
        bounds = np.arange(-.5,unq2[-1]+1.,1.)
        norm = BoundaryNorm(bounds, cmap.N)
        
        
        im0=ax4.pcolormesh(ccx,ccy,np.ma.masked_invalid(unq_inv.reshape(last_array.shape)),cmap=cmap,norm=norm)
        ax4.set_title(cbar_label_name)
        ax4.set_facecolor(axis_bg_color)
        ax4.set_xlabel(xlabel)
        ax4.set(aspect=1)
        dict_vals = list(cgu.grid_type_dict.values())
        dict_keys = list(cgu.grid_type_dict.keys())
        plt_labels = [dict_keys[dict_vals.index(i)] for i in unq_vals[unq2]]
        cb4 = plt.colorbar(im0,ax=ax4,orientation='horizontal',
                           spacing='proportional',
                           ticks=unq2, boundaries=bounds, format='%1i')
        cb4.ax.set_xticklabels(plt_labels)
        cb4.ax.set_xlabel(cbar_label_name)
        if save_fig:
            # nrows/ncols
            model_daspect = head_consolidated.shape[0]/float(head_consolidated.shape[1])
            x_inch = 14.
            y_inch = 1.5+(x_inch*model_daspect)/4.
            fig_dict = {'dpi':300,'format':'png',
                        'bbox_inches':'tight'}
            fig.set_size_inches(x_inch, y_inch)
            fig.savefig(save_fig_fpath,**fig_dict)
            plt.close('all')
        return head_consolidated,wt_layer_mat,[]


def load_cbc(model_name=None,workspace=None,extract_time=-1,
             entries_out = ['   CONSTANT HEAD','FLOW RIGHT FACE',
                            'FLOW FRONT FACE',
                            'FLOW LOWER FACE','          DRAINS',
                            ' HEAD DEP BOUNDS','        RECHARGE'],
             budget_out=True):
    cbb = fu.CellBudgetFile(os.path.join(workspace,'{}.cbc'.format(model_name)))
    out_cbc_dict = {}
    budget_array = None   
    for entry in entries_out:
        mat_out = cbb.get_data(text=entry,full3D=True)[extract_time]
        mat_out = cgu.fill_mask(mat_out,fill_value=0.)
        out_cbc_dict.update({entry.strip():mat_out})
        if budget_out:
            if budget_array is None:
                budget_array = np.zeros_like(mat_out)

            budget_array += mat_out
            
    cbb.close()
    
    return out_cbc_dict, budget_array

def plot_mf_flux(model_name,workspace,cc_pts,q_lay=0):
    ccx,ccy = cc_pts
    nrow,ncol = ccx.shape
    # Load cell budget file
    cbb = fu.CellBudgetFile(workspace+'\\'+model_name+'.cbc')
    frf = cbb.get_data(text='FLOW RIGHT FACE')[-1]
    fff = cbb.get_data(text='FLOW FRONT FACE')[-1]
    cbb.close()
    # Average flows to cell centers
    qx_avg = np.empty(frf.shape, dtype=frf.dtype)
    qx_avg[:, :, 1:] = 0.5 * (frf[:, :, 0:ncol-1] + frf[:, :, 1:ncol])
    qx_avg[:, :, 0] = 0.5 * frf[:, :, 0]
    qy_avg = np.empty(fff.shape, dtype=fff.dtype)
    qy_avg[:,1:, :] = 0.5 * (fff[:,0:nrow-1, :] + fff[:,1:nrow, :])
    qy_avg[:, 0, :] = 0.5 * fff[:, 0, :]
    
    qsurf = np.sqrt(qx_avg[q_lay,:,:]**2+qy_avg[q_lay,:,:]**2)
    
    fig,ax = plt.subplots()
    qsurf[qsurf==0] = np.nan
    im5=ax.pcolormesh(ccx,ccy,np.ma.masked_invalid(qsurf),vmax=500)
    ax.set_title('Discharge magnitude [m/d]')
    c1 = plt.colorbar(im5)
    c1.ax.set_ylabel('Volumetric flux magnitude [m^3/d]')

def calc_gradient(in_array,spacing_xy):
    from scipy.signal import convolve2d
    scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                       [-10+0j, 0+ 0j, +10 +0j],
                       [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

    grad = convolve2d(in_array, scharr, boundary='symm', mode='same')
    mag_array = np.absolute(np.ma.masked_invalid(grad)/(spacing_xy+spacing_xy*1j))
    orient_array = np.ma.masked_invalid(np.angle(grad,deg=True))
    return mag_array, orient_array
    
def plot_gradient(cc_pts,in_array,spacing_xy,return_bool=False):
     
    ccx,ccy = cc_pts
    mag,orient_array = calc_gradient(in_array,spacing_xy)

    fig,(ax,ax2) = plt.subplots(1,2)

    im2=ax.pcolormesh(ccx,ccy,mag,edgecolor='none',cmap='gray')
    ax.set_title('Head gradient')
    cb2=plt.colorbar(im2,ax=ax,orientation='horizontal')
    cb2.ax.set_xlabel('Head gradient [m/m]')
    
    im3=ax2.pcolormesh(ccx,ccy,orient_array,edgecolor='none',cmap='hsv')
    ax2.set_title('Gradient orientation')
    plt.colorbar(im3,ax=ax2, orientation='horizontal')
    if return_bool:
        return mag,orient_array

def plot_sections(mf_model,nums,nrows,indata,row_or_col = 'Column',active_cmap=None):
    '''Plot cross sections from Flopy Modflow model
    
    '''
    
    ncols = np.ceil(float(len(nums))/nrows)
    fig = plt.figure()
    icount=0
    for num in nums:
        icount+=1
        ax = fig.add_subplot(nrows,ncols, icount)
        modelxsect = fplt.ModelCrossSection(model=mf_model, line={row_or_col: num})
#        line_collection = modelxsect.plot_grid()
        if active_cmap is None:
            active_cmap = plt.cm.PuBu
        csa = modelxsect.plot_array(indata,cmap=active_cmap) #edgecolor='none',
#        quiver = modelxsect.plot_discharge(frf, fff, head=indata, 
#                                   hstep=10, color='green', 
#                                   scale=30, headwidth=3, headlength=3, headaxislength=3,
#                                   zorder=10)
        
        line_collection = modelxsect.plot_ibound()#edgecolor='none'
        wt = modelxsect.plot_surface(indata, color='blue', lw=1)
#        line_collection = modelxsect.plot_bc('DRN')
#        ax.set_ylim(np.min(zbot),np.nanmax(ztop))
        ax.set_title(row_or_col + ' ' + str(num) + ' cross-section')
        cb = plt.colorbar(csa, shrink=0.5)

def plot_xsec(mf_model,nums,row_or_col='Column',bc='GHB',return_items=False,
            grid_on=True,ibound_on=True,array=None,fp_kwargs=None,cmap='rainbow',
            surf_kwargs=None):
    '''Plot modflow model cross-section.
    '''
    
    if row_or_col in ['col']:
        row_or_col = 'Column'
    
    botms = mf_model.dis.botm.array
    botms[botms<-1e5]=np.nan
    fig,ax = plt.subplots()
    if fp_kwargs is not None:
        xsect=fplt.ModelCrossSection(model=mf_model,line={row_or_col:nums},**fp_kwargs)
    else:
        xsect=fplt.ModelCrossSection(model=mf_model,line={row_or_col:nums})
    if (bc is not None):
        xsect.plot_bc(bc)
    if grid_on:
        xsect.plot_grid()
    else:
        surf_kwargs.update({'linewidth':0})
    
    if array is not None:
        if len(array.shape)==3:
            if surf_kwargs is not None:
                csa = xsect.plot_array(array,cmap=cmap,**surf_kwargs)
            else:
                csa = xsect.plot_array(array,cmap=cmap)
            cb = plt.colorbar(csa, shrink=0.5)
        else:
            indata = array.reshape(np.hstack([1,array.shape]))
            if surf_kwargs is not None:
                wt = xsect.plot_surface(indata, **surf_kwargs)
            else:
                wt = xsect.plot_surface(indata, color='blue', lw=1)
    if ibound_on:
        xsect.plot_ibound()
    ax.set_ylim(np.nanmin(botms),np.nanmax(mf_model.dis.top.array))
    ax.set_title('{} = {}, showing {}'.format(row_or_col,nums,bc))
    if return_items:
        return fig,ax,xsect
        