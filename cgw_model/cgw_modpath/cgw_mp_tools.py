# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:04:17 2016

@author: kbefus
"""

import os
import numpy as np
import flopy.modpath as mp
import flopy.utils as fu

from cgw_model.cgw_modpath import cgw_mp_utils as cmpu

class Modpath(object):
    '''Modpath creation and run object.'''
    
    def __init__(self,m_run_obj=None,zb_data=None,track_dir='Forward',ibound=None,
                 porosity=None,mf_model=None,out_dir=None):
        '''Collect relevant pieces from Modflow model and ZoneBudget analysis.
        '''
        # Allow some variables to be set as inputs or derived from m_run_obj
        if mf_model is None:
            self.mf_model = m_run_obj.model_obj.mf_model
        else:
            self.mf_model = mf_model
        
        if ibound is None:
            self.ibound = m_run_obj.model_obj.dict_obj.ibound
        else:
            self.ibound = ibound
            
        if porosity is None:
            self.porosity = m_run_obj.model_obj.dict_obj.porosity
        else:
            self.porosity = porosity
        
        # Must have .dis file in out_dir
        if out_dir is None:
            self.out_dir = self.mf_model.model_ws
        else:
            self.out_dir = out_dir
        
        # Other variables are always defined by inputs
        self.track_dir = track_dir
        self.zb_data = zb_data
        
    def mpath(self,exe_name='mp6x64.exe'):
        ''' Create main Modpath model.
        
        Note: MP7 requires slightly different formats to run ('MPath7.exe')
            and not currently supported by Flopy commands.'''
        
        
        mp_dict = {'modelname':self.mf_model.name,'modflowmodel':self.mf_model,
                   'model_ws':self.out_dir,'exe_name':exe_name,'dis_file':self.mf_model.dis.file_name[0]}
        if len(self.mf_model.oc.file_name)>1:
            mp_dict.update({'budget_file':self.mf_model.oc.file_name[3],
                          'head_file':self.mf_model.oc.file_name[1],
                          })
        else: # updated FloPy uses different location for output files
            mp_dict.update({'budget_file':self.mf_model.output_fnames[0],
                          'head_file':self.mf_model.output_fnames[1],
                          })
            
        self.mp_model = mp.Modpath(**mp_dict)
    
    def mpbas(self):
        '''Create Modpath Basic component.'''        
        
        mpBas_dict = {'ibound':self.ibound,
              'prsity':self.porosity}
        
        mp.ModpathBas(self.mp_model,**mpBas_dict)
#        mpBAS.write_file()
        
    def mpsim(self, useZBdata=True,input_zone_data=None,
              zone_stop_array=0,active_iface=[6],pdiv_rc=[4,4],
              weak_sinksource=['Stop','Stop'],sim_type=2,
              track_active=False,land_active=True):
        ''' Create Modpath Simulation component.
        
        
        weak_sinksource: str
            'Allow' or 'Stop' condition for particles reaching weak sinks and
            sourses.
        
        '''
        self.land_active = land_active
        
        if useZBdata:
            # Currently, can only define start/end cells using these zones

            zone_zb_dict = cmpu.mp_zone_from_zb(zones=self.zb_data['zones'],
                                              zone_mapping=self.zb_data['zone_mapping'],
                                              track_dir=self.track_dir,track_active=track_active,
                                              land_active=land_active)
            # Convert zone_array to 2d flopy array                                  
            if zone_zb_dict['zone_array'] is not None:
                zone_zb_dict['zone_array'] = fu.Util2d(self.mf_model,(self.mf_model.nrow,self.mf_model.ncol),
                                                       dtype=np.int,value=zone_zb_dict['zone_array'],
                                                       name='stop_zone',locat=32)
                                   
            # Extract data from zonebudget prep for zone, mask, and group information    
            self.zone_array = zone_zb_dict['zone_array']
            zone_flag = zone_zb_dict['zone_flag']
            mask_array = zone_zb_dict['mask_array']
            release_time = zone_zb_dict['release_time']
            ngroups = zone_zb_dict['ngroups']
            self.group_names = zone_zb_dict['group_names']
            zone_stop_array = zone_zb_dict['stop_zones']
        elif input_zone_data is not None:
            # Use input data for zone, mask, and group information
            self.zone_array = input_zone_data['zone_array']
            zone_flag = input_zone_data['zone_flag']
            mask_array = input_zone_data['mask_array']
            release_time = input_zone_data['release_time']
            ngroups = input_zone_data['ngroups']
            self.group_names = input_zone_data['group_names']
        
        
        # Convert mask_arrays to flopy 2d objects
        mask_arrays = [fu.Util2d(self.mf_model,(self.mf_model.nrow,self.mf_model.ncol),dtype=np.int,value=mask_array[i],name='mask_1lay',locat=32)\
                         for i in range(ngroups)]
        mask_layer = ['0']*ngroups

        # Prepare mpsim input data, eventually add to mpsim inputs
        mpSim_opt_dict = {'SimulationType':sim_type, # 1 for endpt only, 2 for endpt and pthline
                          'TrackingDirection':cmpu.tracking_dict[self.track_dir],
                          'WeakSinkOption':cmpu.weaksink_dict[weak_sinksource[0]],
                          'WeakSourceOption':cmpu.weaksource_dict[weak_sinksource[1]],
                          'ReferenceTimeOption':1,
                          'StopOption':2,
                          'ParticleGenerationOption':1,
                          'TimePointOption':1,
                          'BudgetOutputOption':1,
                          'ZoneArrayOption':cmpu.zonearray_dict[zone_flag],
                          'RetardationOption':1, 'AdvectiveObservationsOption':1}
        
        group_placement_dict = {'Grid':1, 'GridCellRegionOption':3,
                        'PlacementOption':1, 'ReleaseStartTime':release_time,
                        'ReleaseOption':1, 'CHeadOption':1}
                        
        # Order dictionaries                        
        mpSim_opt = cmpu.mp_sim_order_options(mpSim_opt_dict=mpSim_opt_dict)
        group_placement_array = cmpu.mp_order_groups(group_placement_dict=group_placement_dict,ngroups=ngroups)
                 
        # Create final mpsim dictionary
        pdiv_per_row,pdiv_per_col = pdiv_rc        
        iface_rowcount_colcount = [[[active_iface[j],pdiv_per_row,pdiv_per_col] for j in range(len(active_iface))] for i in range(ngroups)]
        face_count = [1]*ngroups                  
        mpSim_dict = {'option_flags':mpSim_opt,
                      'ref_time': group_placement_dict['ReleaseStartTime'],
                      'group_name':self.group_names,
                      'group_placement':group_placement_array,
                      'mask_layer':mask_layer,
                      'mask_1lay':mask_arrays,
                      'stop_zone':zone_stop_array,
                      'zone':self.zone_array,
                      'ifaces':iface_rowcount_colcount,
                      'face_ct':face_count}
        mp.ModpathSim(self.mp_model,**mpSim_dict)
#        mpSIM.write_file()
        
        
    def run_mp(self, write_mp=True, run_mp=True,mpath_kwargs=None,
               mpbas_kwargs=None, mpsim_kwargs=None):
        
        if mpath_kwargs is not None:
            self.mppath(**mpath_kwargs)
        else:
            self.mpath()
        
        if mpbas_kwargs is not None:
            self.mpbas(**mpbas_kwargs)
        else:
            self.mpbas()
        
        if mpsim_kwargs is not None:
            self.mpsim(**mpsim_kwargs)
        else:
            self.mpsim()
        
        if write_mp:
            self.mp_model.write_input()

        
        if run_mp:            
            cmpu.run_mp_process(mp_model=self.mp_model)            

class Modpath_results(object):
    
    def __init__(self,mp_obj=None,XYnodes=None,particle_groups=None):
        self.mp_model = mp_obj.mp_model
        self.mp_obj = mp_obj
        self.XYnodes = XYnodes
        self.particle_groups = particle_groups
        
    def load_endpts(self):
        '''Load Modpath endpoints output file, mpend.'''
        ws = self.mp_model.model_ws
        self.endpt_df = cmpu.load_mp_endpt(os.path.join(ws,'{}.mpend'.format(self.mp_model.name)),
                                           group_ct=len(self.mp_obj.group_names))
    
    def load_pathlines(self):
        '''Load Modpath pathlines output file, mppth.'''
        ws = self.mp_model.model_ws
        self.pthline_fname = os.path.join(ws,'{}.mppth'.format(self.mp_model.name))
        self.pathline_df = cmpu.load_mp_pthln(self.pthline_fname) 
       
    def set_active_particles(self,active_mask=None,start_or_end='end'):
        
        if ~hasattr(self,'endpt_df'):
            self.load_endpts()
        
        if ~hasattr(self,'bool_array') and (active_mask is None):
            zone_kwargs={'zb_dict':self.mp_obj.zb_data['zone_mapping'],
                         'zones':self.mp_obj.zb_data['zones']}
            
            if start_or_end in ['start']:
                zone_kwargs.update({'search_terms':['l','L']})

            self.bool_array,inds = cmpu.collect_zonebytype(**zone_kwargs)
            
        elif active_mask is not None:
            self.bool_array = active_mask.copy()
        
        self.active_particles = cmpu.select_particles_by_mask(mp_endpt_df=self.endpt_df,
                                                              mask=self.bool_array,
                                                              loc_name = start_or_end)
        
    def make_endpt_grid(self,all_endpts=False,loc_name='start'):
        '''Identify active endpt locations.
        
        loc_name is opposite of start_or_end in Modpath_results.set_active_particles().
        '''
        if ~hasattr(self,'active_particles') and not all_endpts:
            self.set_active_particles()
        elif all_endpts:
            self.active_particles = self.endpt_df['ParticleID'].unique()
        
        # Make grid of pt locations that meet endpoint criteria
        self.pt_locs = cmpu.mp_pt_array(mp_endpt_df=self.endpt_df[self.endpt_df['ParticleID'].isin(self.active_particles)],
                                      array_shp=self.XYnodes[0].shape,
                                      loc_name=loc_name)    
        
        if self.mp_obj.land_active:
            # Assign pt cells to catchments and rename consecutively
            assign_dict = {'pt_locs':self.pt_locs,'zones':self.mp_obj.zb_data['zones'],
                           'zb_dict':self.mp_obj.zb_data['zone_mapping']}
            self.pt_locs,self.group_names = cmpu.assign_zbzones(**assign_dict)
        else:
            
            self.group_names = self.mp_obj.group_names
            
    def save_endpt_shapefile(self,inproj=None,xyshift=None,rot_angle=None):
        
        model_name = self.mp_model.name
        ws = self.mp_model.model_ws
        out_fname = os.path.join(ws,'{}_mpendpts_extent.shp'.format(model_name))
        
        # Project to UTM ('proj_out':None) or Geographic from model coordinates
        proj_kwargs = {'proj_in':inproj,'xyul':xyshift,'mf_model':self.mp_model.mf,'rotation':rot_angle}
        out_dict = {'XY':self.XYnodes,'Z':self.pt_locs,
                    'proj_kwargs':proj_kwargs,'out_fname':out_fname,
                    'group_names':self.group_names}
        self.endpt_poly,self.endpt_poly_proj = cmpu.save_endpt_shp(**out_dict)
        
    def check_file_size(self,maxRAMgb=16.):
        '''Check file size vs RAM allotment.'''
        file_data = os.stat(os.path.join(self.mp_model.model_ws,'{}.mppth'.format(self.mp_model.name)))
        self.gb_size = file_data.st_size/1e9 # convert to gigabytes
        if self.gb_size > maxRAMgb:
            self.run_in_chunks = True
        else:
            self.run_in_chunks = False
    
    def pathline_extent(self):
        '''Calculate extent of each pathline group.'''
        # Load pathlines if not already loaded
        if ~hasattr(self,'pathline_df'):
            self.load_pathlines()
                   
        if self.mp_obj.group_names is None:
            self.mp_obj.group_names = self.pathline_df['ParticleGroup'].unique()
        
        extent_dict = {'mp_path_df':self.pathline_df,'active_groups':None,
                       'XY':self.XYnodes,'zone_kwargs':{'zb_dict':self.mp_obj.zb_data['zone_mapping'],
                                                        'zones':self.mp_obj.zb_data['zones']}}
                       
        self.pathline_hits,self.maxpath_polys, self.final_groups = cmpu.calc_max_extent(**extent_dict)
    
    def iter_pathline_extent(self,chunksize=1e7):
        '''Iteratively load and calculate pathline extents.'''
        
        self.pthline_fname = os.path.join(self.mp_model.model_ws,'{}.mppth'.format(self.mp_model.name))
        if chunksize is not None:
            mp_path_iter = cmpu.iter_load_mp_pthln(self.pthline_fname,chunksize=chunksize)
        else:
            mp_path_iter = cmpu.iter_load_mp_pthln(self.pthline_fname)
        
        extent_dict = {'active_groups':None,
                       'XY':self.XYnodes,'zone_kwargs':{'zb_dict':self.mp_obj.zb_data['zone_mapping'],
                                                        'zones':self.mp_obj.zb_data['zones']}}
                       
        iter_kwargs = {'in_iter_df':mp_path_iter,'extent_kwargs':extent_dict}

        self.pathline_hits,self.maxpath_polys, self.final_groups = cmpu.iter_extent_pathlines(**iter_kwargs)    

    def calc_pathline_extent(self):
        '''Calculate extent using iterations or directly.'''
        self.check_file_size()
        
        if self.run_in_chunks:
            self.iter_pathline_extent()
        else:
            self.pathline_extent()
            
            
    def export_mp_extentshp(self,fname=None,inproj=None,xyshift=None,rot_angle=None):
        '''Save pathline areas to shapefile.'''
        model_name = self.mp_model.name
        ws = self.mp_model.model_ws
        out_fname = os.path.join(ws,'{}_mp_extent.shp'.format(model_name))
        col_name_order = ['CAFmodel','CAF_FID','MPgroup']
        if self.particle_groups is None:
            self.particle_groups = self.mp_obj.group_names
        CAFws = np.array([int(igroup.split('_')[0][1:]) for igroup in self.mp_obj.group_names])[np.array(self.final_groups)-1]
        data = np.array(zip([int(model_name.split('_')[1])]*len(self.particle_groups),CAFws,np.array(self.final_groups)))
        field_dict = cmpu.cfu.df_field_dict(None,col_names=col_name_order,col_types=['int','int','int'])
  
        # Project to UTM from model coordinates
        proj_kwargs = {'proj_in':inproj,'proj_out':None,'xyul':xyshift,'mf_model':self.mp_model.mf,'rotation':rot_angle}
        proj_dict = {'polys':self.maxpath_polys,'proj_kwargs':proj_kwargs}

        self.maxpath_polys_utm = cmpu.cfu.proj_polys(**proj_dict)
        # Remove incomplete entries
        ind_keep = [i for i,temp_poly in enumerate(self.maxpath_polys_utm) if hasattr(temp_poly,'area')]
        self.maxpath_polys_utm = np.array(self.maxpath_polys_utm)[ind_keep]

        if len(ind_keep)!=data.shape[0]:
            data = data[ind_keep,:]

        shp_dict = {'polys':self.maxpath_polys_utm,'data':data,'out_fname':out_fname,
                               'field_dict':field_dict,'col_name_order':col_name_order,
                               'write_prj_file':True,'inproj':inproj}
                               
        cmpu.cfu.write_shp(**shp_dict)
    
    def save_pathhits_nc(self,new_XY=None):
        
        # Can input other XY grids (i.e., projected or lat long)
        if new_XY is None:
            new_XY = self.XYnodes
        
        nc_fname = os.path.join(self.mp_model.model_ws,'{}_pathline_count.nc'.format(self.mp_model.name))
        cmpu.save_mp_nc(out_fname=nc_fname,model_name=self.mp_model.name,
                   XY=new_XY, grid=self.pathline_hits,grid_name='mp_hits')
        
    def run_pathline_analysis(self,shp_kwargs=None,nc_kwargs=None):
       
       self.calc_pathline_extent()
       self.export_mp_extentshp(**shp_kwargs)
       if nc_kwargs is not None:
           self.save_pathhits_nc(**nc_kwargs)

    def run_endpt_analysis(self,part_kwargs=None,grid_kwargs=None,shp_kwargs=None):
        
        if part_kwargs is not None:
            self.set_active_particles(**part_kwargs)
        else:
            self.set_active_particles()
        
        if grid_kwargs is not None:
            self.make_endpt_grid(**grid_kwargs)
        else:
            self.make_endpt_grid()
        
        if shp_kwargs is not None:
            self.save_endpt_shapefile(**shp_kwargs)
        
        
        