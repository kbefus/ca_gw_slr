# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:45:00 2016

@author: kbefus


path to the cgw_model needs to be added to the path before importing
cgw_package_tools

e.g.:
import sys
kbpath = 'C:/Research/Coastalgw/Model_develop/'
sys.path.insert(1,kbpath)

"""
from __future__ import print_function
import os
import numpy as np
import geopandas as gpd
from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_raster_utils as cru
from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_utils import cgw_caf_utils as ccu
from cgw_model.cgw_modflow import cgw_mf_utils as cmfu
#from cgw_model.cgw_zonebudget import cgw_zb_utils as zbu

class Model_domain_driver(object):
    '''
    Create coastal groundwater domain model class
    '''
    
    def __init__(self,domain_poly=None,
                 cell_spacing=None, input_dir='',model_in_dir=None,
                 domain_shp=None, active_domain=None,
                 elev_fname=None, rchg_fname=None,
                 k_fnames=None,sea_level=0.,in_proj=None,use_ll=True):
       self.cell_spacing = cell_spacing
       self.input_dir = input_dir
       self.model_in_dir = model_in_dir
       self.domain_shp = domain_shp
       self.domain_poly = domain_poly
       self.active_domain = active_domain
       self.sea_level = sea_level
       self.in_proj = in_proj
       self.use_ll = use_ll
       
       if (elev_fname is not None):
           self.elev_path = os.path.join(self.input_dir,elev_fname)
        
       if (rchg_fname is not None):
           self.rchg_path = os.path.join(self.input_dir,rchg_fname)
           
       if (self.domain_shp is not None) and (self.domain_poly is None):
           if hasattr(self.domain_shp,'to_file'):
               
               self.domain_poly = self.domain_shp.loc[self.active_domain].geometry
           else:
               self.domain_poly = cfu.shp_to_polygon(self.domain_shp)[0][self.active_domain]
    
       if (k_fnames is not None):
           self.k_fnames = k_fnames
    
    def make_modflow_grid(self):
        
        
        to_grid_dict = {'shp':self.domain_poly,
                        'cell_spacing':self.cell_spacing,
                        'active_feature':self.active_domain,
                        'in_proj':self.in_proj}
        self.XYnodes,self.model_polygon,self.grid_transform = cfu.shp_to_grid(**to_grid_dict)
        
        self.domain_extent =  [self.XYnodes[0].min(),self.XYnodes[1].min(),self.XYnodes[0].max(),self.XYnodes[1].max()]
     
    def make_cc(self):
        self.cc,self.cc_proj,self.cc_ll = cfu.nodes_to_cc(self.XYnodes,self.grid_transform)
        self.nrow,self.ncol = self.cc[0].shape
        dlat,dlong = self.cc_ll[1][0,0]-self.cc_ll[1][1,0],self.cc_ll[0][0,0]-self.cc_ll[0][0,1]
        self.domain_extent_ll =  [self.cc_ll[0].min()-dlong/2.,self.cc_ll[1].min()-dlat/2.,
                                  self.cc_ll[0].max()+dlong/2.,self.cc_ll[1].max()+dlat/2.]
        return
    
    def find_active_cells(self,rmv_isolated=True,min_area=1e3):
        '''Find active model cells using polygon area.
        
        Inputs
        ----------
        rmv_isolated: bool, remove isolated areas
        min_area: float, minimum area/number of cells to be considered not isolated
        '''
        ir,ic,_ = cfu.gridpts_in_shp(self.model_polygon,self.cc)
        self.active_cells = cgu.define_mask(self.cc,[ir,ic])
        if rmv_isolated:
            self.active_cells = cmfu.clean_ibound(self.active_cells,check_inactive=True,min_area=min_area)
        return
    
    def load_nc_grids(self,nc_fname=None):

        if os.path.isfile(nc_fname):
            f = cru.netCDF4.Dataset(nc_fname)
            self.elevation = f.variables['elev'][:]
            self.recharge = f.variables['recharge'][:]
            self.nrow,self.ncol = self.recharge.shape
            if 'hk' in f.variables:
                self.hk_in_array = f.variables['hk'][:]
                if 'thick' in list(f.variables.keys()):
                    self.hk_in_botm = f.variables['thick'][:]
                    # Use hk_in_botm to cull layers
                    self.hk_in_botm,maxnlay = cmfu.cull_layers(in_array=self.hk_in_botm)
                else:
                    self.hk_in_botm = []

                if 'vk' in list(f.variables.keys()):
                    self.vk_in_array = f.variables['vk'][:]
                if 'thick' in list(f.variables.keys()):
                    if maxnlay.shape[0] < self.hk_in_array.shape[0]:
                        self.hk_in_array=self.hk_in_array[maxnlay]
                        self.vk_in_array=self.vk_in_array[maxnlay]
                    
            if 'density' in f.variables:
                self.density = f.variables['density'][:]
            if 'sea_level' in f.variables:
                self.sea_level = f.variables['sea_level'][:]
                
            f.close()
            if hasattr(self, 'active_cells'):
                if self.elevation.shape != self.active_cells.shape:
                    # Need to account for no data and clipping from previous run
                    ref_name=os.path.join(self.model_in_dir,'usgs.model.reference')
                    if os.path.isfile(ref_name):
                        ref_dict=cgu.read_model_ref(ref_name)
                        grid_tform = cgu.make_grid_transform(ref_dict,from_ref=True)
                        mask_array = cgu.match_grid_size(mainXY=self.cc_proj,new_xyul=grid_tform[1],
                                                         new_shape = self.elevation.shape)
                        
                        apply_dict = {'func':cgu.shrink_ndarray,
                                      'func_args':{'mask_in':mask_array,
                                      'shape_out':self.elevation.shape},'skip_vals':['conversion_mask']}
                        
                        cgu.recursive_applybydtype(self,**apply_dict)
                
            return True
        else:
            print("Loading grid data...({} not found)".format(os.path.basename(nc_fname)))
            return False
    
    def load_griddata(self,use_mask=True, load_vk=False,
                      save_nc_grids=False,load_nc_grids_bool=False,
                      max_elev=1e5, landfel=False,n_proc=8):
        
        if use_mask:
            mask = ~self.active_cells
        else:
            mask = None            
        
        if load_nc_grids_bool:
            out_nc_name = os.path.join(self.model_in_dir,'{}.nc'.format(os.path.basename(self.model_in_dir)))
            load_bool = self.load_nc_grids(nc_fname=out_nc_name)
            save_nc_grids = not load_bool
        
        if save_nc_grids:
            out_nc_name = os.path.join(self.model_in_dir,'{}.nc'.format(os.path.basename(self.model_in_dir)))
                
            save_grid_dict = {'out_desc':'Grid data for cgw model {}'.format(os.path.basename(self.model_in_dir)),
                              'dims':{'dim_order':['Y','X'],
                                      'X':{'attr':{'var_desc':'X coordinate of cell center in model reference frame',
                                                   'long_name':'X','units':'m'},
                                           'data':self.cc[0][0,:]},
                                      'Y':{'attr':{'var_desc':'Y coordinate of cell center in model reference frame',
                                                   'long_name':'Y','units':'m'},
                                           'data':self.cc[1][:,0]}},
                               'vars':[]}
        
        self.gdict = {'crs':self.in_proj}
        if self.use_ll:
            self.gdict.update({'new_xy':self.cc_ll})#,'in_extent':self.domain_extent_ll})
        else:
            self.gdict.update({'new_xy':self.cc_proj})#,'in_extent':self.domain_extent})
            
        if hasattr(self,'elev_path') and not hasattr(self,'elevation'):
            self.elevation = cru.gdal_loadgrid(self.elev_path,**self.gdict)
            if landfel:
                # Fill pits in dem in areas above sea level
                fel_dict = {'work_dir':self.model_in_dir,
                            'xy':self.cc_proj,'elev_data':self.elevation,
                            'proj': self.grid_transform[0],'n_proc':n_proc,
                            'sea_level':self.sea_level}
                self.elevation = cru.landfel_taudem(**fel_dict)
                
            with np.errstate(invalid='ignore'):
                self.elevation[np.abs(self.elevation)>1e5] = np.nan
            if save_nc_grids:
                save_grid_dict.update({'elev':{'attr':{'var_desc':'Land surface elevation in meters above sea level',
                                                       'long_name':'elevation','units':'m'},
                                              'dims':('Y','X'),
                                              'data':self.elevation}})
                save_grid_dict['vars'].append('elev')   

        elif hasattr(self,'elevation') and save_nc_grids:
                save_grid_dict.update({'elev':{'attr':{'var_desc':'Land surface elevation in meters above sea level',
                                                       'long_name':'elevation','units':'m'},
                                              'dims':('Y','X'),
                                              'data':self.elevation}})
                save_grid_dict['vars'].append('elev') 
                            
        if hasattr(self,'rchg_path') and not hasattr(self,'recharge'):
            self.recharge = cru.gdal_loadgrid(self.rchg_path,**self.gdict)
            if save_nc_grids:
                save_grid_dict.update({'recharge':{'attr':{'var_desc':'Groundwater recharge in meters per year',
                                                           'long_name':'recharge','units':'m/yr'},
                                              'dims':('Y','X'),
                                              'data':self.recharge}})
                save_grid_dict['vars'].append('recharge')
        elif hasattr(self,'rchg_path') and save_nc_grids:
            save_grid_dict.update({'recharge':{'attr':{'var_desc':'Groundwater recharge in meters per year',
                                                       'long_name':'recharge','units':'m/yr'},
                                          'dims':('Y','X'),
                                          'data':self.recharge}})
            save_grid_dict['vars'].append('recharge')
        
        if hasattr(self,'density') and save_nc_grids:
            save_grid_dict.update({'density':{'attr':{'var_desc':'Seawater density',
                                                       'long_name':'Density of seawater','units':'kg/m3'},
                                          'dims':('Y','X'),
                                          'data':self.density}})
            save_grid_dict['vars'].append('density')
        
        if hasattr(self,'sea_level') and save_nc_grids:
            save_grid_dict.update({'sea_level':{'attr':{'var_desc':'Sea level',
                                                       'long_name':'Sea level elevation above NAVD88','units':'m'},
                                          'dims':('Y','X'),
                                          'data':self.sea_level}})
            save_grid_dict['vars'].append('sea_level')
        
        if hasattr(self,'k_fnames') and not hasattr(self,'hk_in_array'):
            self.hk_in_array,self.vk_in_array,self.hk_in_botm = cru.read_hk(self.k_fnames,griddata_dict=self.gdict,
                                                                            load_vk=load_vk)
            if len(self.hk_in_botm) > 0:                                                                
                # Remove purely null data layers to save time and space with Modflow files
                self.hk_in_botm,maxnlay = cmfu.cull_layers(in_array=self.hk_in_botm)
            
                if maxnlay.shape[0] < self.hk_in_array.shape[0]:
                    self.hk_in_array=self.hk_in_array[maxnlay]
                    if len(self.vk_in_array)>0:
                        self.vk_in_array=self.vk_in_array[maxnlay]

            if save_nc_grids:
                save_grid_dict['dims']['dim_order'].insert(0,'layers')
                save_grid_dict['dims'].update({'layers':{'attr':{'var_desc':'Layers of hydrogeologic framework data',
                                                                'long_name':'layers','units':'n/a'},
                                                         'data':np.arange(self.hk_in_array.shape[0])}})
                save_grid_dict.update({'hk':{'attr':{'var_desc':'Horizontal hydraulic conductivity in meters per day',
                                              'long_name':'Horizontal hydraulic conductivity','units':'m/day'},'dims':('layers','Y','X'),
                                              'data':self.hk_in_array}})
                save_grid_dict['vars'].append('hk')                              

                if self.vk_in_array is not None:                              
                    save_grid_dict.update({'vk':{'attr':{'var_desc':'Vertical hydraulic conductivity in meters per day',
                                              'long_name':'Vertical hydraulic conductivity','units':'m/day'},
                                                  'dims':('layers','Y','X'),
                                                  'data':self.hk_in_array}})
                    save_grid_dict['vars'].append('vk')
                
                if len(self.hk_in_botm)>0:
                    save_grid_dict.update({'thick':{'attr':{'var_desc':'Thickness of layer in meters','long_name':'layer thickness',
                                                            'units':'m'},
                                                  'dims':('layers','Y','X'),
                                                  'data':self.hk_in_botm}})
                    save_grid_dict['vars'].append('thick')
        elif hasattr(self,'hk_in_array') and save_nc_grids:
            save_grid_dict['dims']['dim_order'].insert(0,'layers')
            save_grid_dict['dims'].update({'layers':{'attr':{'var_desc':'Layers of hydrogeologic framework data',
                                                            'long_name':'layers','units':'n/a'},
                                                     'data':np.arange(self.hk_in_array.shape[0])}})
            save_grid_dict.update({'hk':{'attr':{'var_desc':'Horizontal hydraulic conductivity in meters per day',
                                          'long_name':'Horizontal hydraulic conductivity','units':'m/day'},'dims':('layers','Y','X'),
                                          'data':self.hk_in_array}})
            save_grid_dict['vars'].append('hk')                              

            if self.vk_in_array is not None:                              
                save_grid_dict.update({'vk':{'attr':{'var_desc':'Vertical hydraulic conductivity in meters per day',
                                          'long_name':'Vertical hydraulic conductivity','units':'m/day'},
                                              'dims':('layers','Y','X'),
                                              'data':self.hk_in_array}})
                save_grid_dict['vars'].append('vk')
            if len(self.hk_in_botm)>0:    
                save_grid_dict.update({'thick':{'attr':{'var_desc':'Thickness of layer in meters','long_name':'layer thickness',
                                                        'units':'m'},
                                              'dims':('layers','Y','X'),
                                              'data':self.hk_in_botm}})
                save_grid_dict['vars'].append('thick')
            
        if save_nc_grids:    
            cru.save_nc(fname=out_nc_name,out_data_dict=save_grid_dict)  
                                  
        return
    
    def run_make_domain(self,use_mask=True, load_vk=False, save_nc_grids=False,load_nc_grids_bool=False):
        self.make_modflow_grid()
        self.make_cc()
        self.find_active_cells()
        self.load_griddata(use_mask=use_mask,load_vk=load_vk,
                           save_nc_grids=save_nc_grids,load_nc_grids_bool=load_nc_grids_bool)


class Assign_cell_types(object):
    '''
    Use Model_domain_driver object with ws_shp feature of watershed boundaries to 
    and loaded grids to define model cell types
    '''
    def __init__(self,domain_obj=None,ws_shp=None,waterbody_shp=None):
        self.domain_obj = domain_obj
        self.ws_shp = ws_shp
        self.waterbody_shp = waterbody_shp

    def find_inland_cells(self,find_water=True):
        self.caf_land,self.caf_water = ccu.find_land_water(self.ws_shp,self.domain_obj.domain_poly,find_water=find_water)
        inland_inds = cfu.gridpts_in_shp(self.caf_land[0],self.domain_obj.cc_ll)       
        self.land_cells = cgu.define_mask(self.domain_obj.cc_ll,inland_inds[:2])
    
    def find_waterbody_cells(self):
#        waterbody_fname = os.path.join(self.domain_obj.input_dir,self.waterbody_shp)
        waterbody_inds = cfu.gridpts_in_shp(self.waterbody_shp,self.domain_obj.cc_ll)
        self.wb_cells = cgu.define_mask(self.domain_obj.cc_ll,waterbody_inds[:2])
    
    def run_assignment(self,use_elevation=True, use_ws=True,assign_wb = False,elev_uncertainty=0.):
        if use_ws:
            self.find_inland_cells()
            
        # Initialize cell_types as all zeros (inactive)    
        self.cell_types = np.zeros(self.domain_obj.cc[0].shape)

        if hasattr(self,'land_cells'):
            if isinstance(self.domain_obj.active_cells,np.ndarray):
                self.cell_types[self.domain_obj.active_cells & self.land_cells] = cgu.grid_type_dict['active']
                self.cell_types[self.domain_obj.active_cells & ~self.land_cells] = cgu.grid_type_dict['nearshore']
            else:
                self.cell_types[self.land_cells] = cgu.grid_type_dict['active']
            
        if assign_wb:
            self.find_waterbody_cells()
            # Assign non-coastal, active cells to matching waterbodies
            self.cell_types[(self.cell_types==cgu.grid_type_dict['active']) & self.wb_cells] = cgu.grid_type_dict['waterbody']        
            
        if use_elevation:
            if not use_ws and not assign_wb:
                # Need to assign active and water cells by elevation only
                 with np.errstate(invalid='ignore'):
                    self.cell_types[(self.domain_obj.elevation>self.domain_obj.sea_level) &\
                                    self.domain_obj.active_cells &\
                                    ~np.isnan(self.domain_obj.elevation)] = cgu.grid_type_dict['active']
                    self.cell_types[(self.domain_obj.elevation<=self.domain_obj.sea_level) &\
                                    self.domain_obj.active_cells &\
                                    ~np.isnan(self.domain_obj.elevation)]=cgu.grid_type_dict['nearshore']
                    
            else:
                if isinstance(self.domain_obj.elevation,np.ndarray):
                    # Use elevation data to assign land areas that the buffer method missed
                    with np.errstate(invalid='ignore'):
                        self.cell_types[(self.domain_obj.elevation>self.domain_obj.sea_level) &\
                                        (self.cell_types==cgu.grid_type_dict['nearshore'])] = cgu.grid_type_dict['active'] # convert to land
                    
                        self.cell_types[(self.domain_obj.elevation<=self.domain_obj.sea_level+elev_uncertainty) &\
                        (self.cell_types==cgu.grid_type_dict['waterbody'])]=cgu.grid_type_dict['nearshore'] # convert to marine; for East coast, elev_uncertainty=2 m
                        
                        # Assign submerged areas that the land polygon covered
                        self.cell_types[(self.domain_obj.elevation <= self.domain_obj.sea_level) &\
                        (self.cell_types==cgu.grid_type_dict['active'])] = cgu.grid_type_dict['nearshore']
                        
                        # Assign any missing areas to inactive
                        self.cell_types[np.isnan(self.domain_obj.elevation)] = cgu.grid_type_dict['inactive']
                
        
            
        
  
        
        
        