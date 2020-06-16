# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:14:14 2016

@author: kbefus
"""
from __future__ import print_function
import os
import numpy as np

from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_raster_utils as cru
from cgw_model.cgw_zonebudget import cgw_zb_utils as zbu

class Assign_zones(object):
    '''
    Define zones for ZoneBudget analysis using cgw_model.cgw_package_tools.Assign_cell_types object.
    '''
    def __init__(self,cell_type_obj=None,zones=None):
        
        self.zones = zones
        self.ws_zones_assigned = False
        self.cwb_zones_assigned = False
        
        if (cell_type_obj is not None):
            self.cell_types = cell_type_obj.cell_types
            self.caf_water = cell_type_obj.caf_water
            self.caf_land = cell_type_obj.caf_land
            self.cc_ll = cell_type_obj.domain_obj.cc_ll
        
        if (self.zones is None):
            self.zones = np.abs(self.cell_types).astype(np.int).copy()
            
        # Do not treat inland waterbodies as a unique zone
        self.zones[self.zones==np.abs(cgu.grid_type_dict['waterbody'])] = cgu.grid_type_dict['active']
        
        self.n_other_zones = np.max(self.zones)
    
    def combine_duplicates(self):
        '''Merge duplicate input attributes to single zone.'''
        
        self.caf_land = zbu.merge_attr(self.caf_land)
        self.caf_water = zbu.merge_attr(self.caf_water)
        
    def calc_ws_zones(self):
        '''Assign watershed zones.'''
        if self.ws_zones_assigned:
            print('Warning: Watershed zones already computed')
            print('To re-run watershed zone assignment:')
            print('\t 1) reset ws_zones_assigned to False')
            print('\t 2) reset n_other_zones')
        else:
            caf_land_polys = self.caf_land[1] # watershed polygons
            self.ws_zones,self.ws_caf_assignment = zbu.assign_zone(caf_land_polys,self.cc_ll,
                                                         n_other_zones=self.n_other_zones)
            self.n_other_zones = np.max(self.ws_zones) # warning: can infinitely add more zones if run multiple times                                                     
            self.ws_zones_assigned = True
    
    def calc_cwb_zones(self):
        '''Assign coastal waterbody zones.
        '''
        if self.cwb_zones_assigned:
            print('Warning: Coastal waterbody zones already computed')
            print('To re-run cwb zone assignment:')
            print('\t 1) reset cwb_zones_assigned to False')
            print('\t 2) reset n_other_zones')
        else:
            caf_water_polys = self.caf_water[0] # water polygons
            self.cwb_zones,self.cwb_caf_assignment = zbu.assign_zone(caf_water_polys,self.cc_ll,
                                                         n_other_zones=self.n_other_zones)
            self.n_other_zones = np.max(self.cwb_zones) # warning: can infinitely add more zones if run multiple times                                                     
            self.cwb_zones_assigned = True

    def add_layers(self,max_layer=None,new_zones=True):
        '''Create new zones for each layer to max_layer.
        '''
        if hasattr(self,'zone3d'):
            nzone_layers = self.zones.shape[0]
            if max_layer > nzone_layers:
                self.zones = np.concatenate((self.zones,np.tile(self.zones[-1,:,:],(max_layer-nzone_layers,1,1))),axis=0)
                for ilay in range(nzone_layers,max_layer):
                    self.zones[ilay,:,:],num_add = zbu.add_zb_layer(self.zones[ilay,:,:],n_other_zones=self.n_other_zones)
                    
                    
        elif len(self.zones.shape)==2:
            self.zone3d=False
            
        if (max_layer is not None) and (self.zone3d==False):
            # Convert to 3D array
            self.zones = self.zones.reshape(np.hstack([1,self.zones.shape]))
            self.zones = np.tile(self.zones,(max_layer,1,1))
            
            if self.cwb_zones_assigned:
                self.cwb_caf_assignment = np.hstack([self.cwb_caf_assignment,np.zeros((self.cwb_caf_assignment.shape[0],1))])
            
            if self.ws_zones_assigned:            
                self.ws_caf_assignment = np.hstack([self.ws_caf_assignment,np.zeros((self.ws_caf_assignment.shape[0],1))])
                
            num_add=np.max(self.zones[0,:,:])+1
            for ilay in range(1,max_layer):
                self.zones[ilay,:,:],_ = zbu.add_zb_layer(self.zones[ilay,:,:],new_zones=new_zones,num_add=num_add)
                
                # Update zone assignment matrixes
                if self.cwb_zones_assigned:
                    self.cwb_caf_assignment = zbu.update_zb_assignment(assign_array=self.cwb_caf_assignment,
                                                               num_add=num_add,ilay=ilay)
                if self.ws_zones_assigned:
                    self.ws_caf_assignment = zbu.update_zb_assignment(assign_array=self.ws_caf_assignment,
                                                               num_add=num_add,ilay=ilay)
                num_add = np.max(self.zones[ilay,:,:])+1
                
            self.zone3d = True
    
    def run_assign_zones(self,assign_cwb=True,
                         assign_ws=False,fill_zones=True,
                         row_col_buff=50,repeat_fill=2,combine_attr=False):
        '''Run zone assignment functions.
        '''
        if combine_attr:
            self.combine_duplicates()
        if assign_cwb:
            if not self.cwb_zones_assigned:
                self.calc_cwb_zones()
            # Do not assign land cells to coastal water bodies
            self.cwb_zones[self.cell_types==cgu.grid_type_dict['active']] = 0.
            self.zones[self.cwb_zones!=0] = self.cwb_zones[self.cwb_zones!=0].astype(np.int)
        
        if assign_ws:
            if not self.ws_zones_assigned:
                self.calc_ws_zones()
            # Only assign watersheds to land or waterbody cells
            self.ws_zones[(self.cell_types!=cgu.grid_type_dict['active']) &\
                          (self.cell_types!=cgu.grid_type_dict['waterbody'])] = 0.
            self.zones[self.ws_zones!=0] = self.ws_zones[self.ws_zones!=0].astype(np.int)
            
        if fill_zones:
            # Fill unassigned land and nearshore areas with closes. Note returns 2d zone array
            if assign_ws:
                # Fill land zones
                for itime in range(repeat_fill):
                    self.zones=zbu.fill_closest_zone(self.zones,zone_range=self.ws_caf_assignment[:,1],
                                                     fill_active=True,row_col_buff=row_col_buff)
                
            if assign_cwb:
                # Fill nearshore zones
                for itime in range(repeat_fill):
                    self.zones=zbu.fill_closest_zone(self.zones,zone_range=self.cwb_caf_assignment[:,1],
                                                     fill_wb=True,row_col_buff=row_col_buff)
                

class Make_ZoneBudget(object):
    '''Create inputs and run ZoneBudget.
    '''
    def __init__(self,model_name=None,output_ws=None, dims=None,
                 zones=None,zone_layer=None,composite_zones=None):
        self.model_name = model_name
        self.output_ws = output_ws
        self.dims = dims
        self.zones = zones
        self.zone_layer = zone_layer
        self.composite_zones = composite_zones
  
    def run_ZoneBudget(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_ws
            
        self.zb_file = zbu.ModflowZoneBudget(self.model_name, output_dir,
                                             self.zones,self.dims,
                                           zone_layer=self.zone_layer,
                                           composite_dict=self.composite_zones)
        
            
        self.zb_output = os.path.join(output_dir,'{}_ZONBUD.csv'.format(self.model_name))
          
class Process_ZoneBudget(object):
    '''Process ZoneBuget output files.
    '''
    def __init__(self,zb_output=None,zone_obj=None):

        self.zone_obj = zone_obj
        self.zb_output = zb_output
        self.zb_ws = os.path.dirname(self.zb_output)

    def load_output(self):
        self.zb_dfs = zbu.load_ZONBUD(self.zb_output)
        # Check to see if more than one time/stress period of data returned
        if isinstance(self.zb_dfs[0],(list,tuple)):
            self.multiple_times = True
        else:
            self.multiple_times = False
    
    def map_ids(self,caf_water_huc_ind=2,caf_land_huc_ind=3):
        '''Define mapping between zones and features.
        '''
        self.zone_mapping = zbu.zone_type_dict.copy()
        # Find mapping for water CAF HUCs to Zones
        if self.zone_obj.cwb_zones_assigned:
            cwb_HUC_dict,HUC_names = zbu.zones_to_caf(cafind_to_zone=self.zone_obj.cwb_caf_assignment,
                                                      caf_var=self.zone_obj.caf_water[caf_water_huc_ind],
                                                      output_fmt = 'W{}_lay{}')
            # Upadate mapping dictionary
            self.zone_mapping.update(cwb_HUC_dict)
        
        # Find mapping for watershed (i.e., land) CAF HUCs to Zones
        if self.zone_obj.ws_zones_assigned:
            ws_HUC_dict,HUC_names = zbu.zones_to_caf(cafind_to_zone=self.zone_obj.ws_caf_assignment,
                                                      caf_var=self.zone_obj.caf_land[caf_land_huc_ind],
                                                      output_fmt = 'L{}_lay{}') # caf_var is caf_FID
            # Upadate mapping dictionary
            self.zone_mapping.update(ws_HUC_dict)

                                                  
    def assign_ids(self, active_time=-1):
        '''Assign mapping from zones to feature data.
        '''
        if self.multiple_times:
            self.zb_dfs_active = self.zb_dfs[active_time]
        else:
            self.zb_dfs_active = self.zb_dfs
          
        for idf in range(len(self.zb_dfs_active)):
            # Rename zones
            zbu.rename_df_entry(self.zb_dfs_active[idf],
                              rename_dict =self.zone_mapping,
                              rename_index='both')
     
    def output_Qwb(self,save_shp=False,active_layer=0):
        '''Save ZoneBudget analysis to shp or csv.
        '''
        out_name = '{}_Qwb'.format(os.path.basename(self.zb_ws))
        out_fname = os.path.join(self.zb_ws,out_name)
        self.wb_influx_dict,self.output_data = zbu.calc_cgw_flux(self.zb_dfs_active[0],
                                                                 active_layer=active_layer)
        self.fluxsave_outputs=zbu.save_cgw_flux(self.output_data ,self.zone_obj.caf_water,
                             fname=out_fname,save_shp=save_shp)
    def save_rchg(self):
        out_name = '{}_RechargeDrain.csv'.format(os.path.basename(self.zb_ws))
        out_fname = os.path.join(self.zb_ws,out_name)
        zbu.save_RchgDrain_flux(in_df=self.zb_dfs_active[0],out_fname=out_fname)
        
    def save_grid(self,grid=None,grid_name=None):
        out_name = '{}_{}.txt'.format(os.path.basename(self.zb_ws),grid_name)
        out_fname = os.path.join(self.zb_ws,out_name)
        header = ("# Model {}\n"\
                  "# Grid name={}\n"\
                  "# nrows={}, ncols={}\n").format(os.path.basename(self.zb_ws),grid_name,grid.shape[0],grid.shape[1])
        cru.save_txtgrid(fname=out_fname,data=grid,header=header)
        
    def run_all(self,save_data=False,save_shp=False):
        '''Run all functions in Process_ZoneBudget in order.
        '''
        self.load_output()
        self.map_ids()
        self.assign_ids() 
        if save_data:
            self.save_rchg()
            self.output_Qwb(save_shp=save_shp)

          
          
          
          