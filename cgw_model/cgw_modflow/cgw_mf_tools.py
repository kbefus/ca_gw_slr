# -*- coding: utf-8 -*-
"""
Created on Fri May 13 13:00:00 2016

@author: kbefus



"""
#%%
from __future__ import print_function
import os,time
import numpy as np
import flopy.modflow as mf
import flopy.utils as fu
from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_raster_utils as cru
from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_modflow import cgw_mf_utils as cmfu


class Model_info(object):
    '''
    Organization class for defining model name and working directories
    '''
    def __init__(self, workspace=None, data_dir=None, model_name=None,
                 cbc_unit=53):
        self.workspace = workspace
        self.model_name = model_name
        self.data_dir = data_dir
        self.cbc_unit = cbc_unit

        if (self.data_dir is not None) and (self.workspace is not None):
             # Assign and create array directory if needed
            self.array_dir = os.path.join(self.workspace,self.data_dir)
            if os.path.exists(self.array_dir)==False:  
                os.mkdir(self.array_dir)
            

class Model_maker(object):   
    '''
    Create flopy Modflow model object and call packages using package dictionaries
    created in Model_dict class.
    '''
    def __init__(self,dict_obj=None,info_obj=None,
                 exe_name='mf2005.exe',
                 external_path=None,output_path=None):
        self.dict_obj = dict_obj
        self.info_obj = info_obj
        self.external_path = external_path
        self.output_path = output_path
        
        if self.dict_obj.solver not in ['nwt']:
            self.exe_name = exe_name
            self.model_version = 'mf2005'
        else:
            self.exe_name = 'MODFLOW-NWT_64.exe'
            self.model_version = 'mfnwt'
        
    def model_init(self):
        self.mf_model = mf.Modflow(self.info_obj.model_name,
                                   model_ws=self.info_obj.workspace,
                                   exe_name = self.exe_name,
                                   version=self.model_version,
                                   external_path=self.external_path)
    
    def make_packages(self):
        mf.ModflowDis(self.mf_model,**self.dict_obj.dis_dict)
        mf.ModflowBas(self.mf_model,**self.dict_obj.bas_dict)
        if self.dict_obj.solver not in ['nwt']:
            mf.ModflowBcf(self.mf_model,**self.dict_obj.bcf_dict)
            mf.ModflowGmg(self.mf_model,**self.dict_obj.gmg_dict)
        else:
            mf.ModflowUpw(self.mf_model,**self.dict_obj.upw_dict)
            mf.ModflowNwt(self.mf_model,**self.dict_obj.nwt_dict)
            
        mf.ModflowRch(self.mf_model,**self.dict_obj.rchg_dict)
        if len(self.dict_obj.drn_dict['stress_period_data'][0])>0:
            mf.ModflowDrn(self.mf_model,**self.dict_obj.drn_dict)
        if self.dict_obj.ghb_dict is not None:
            if len(self.dict_obj.ghb_dict['stress_period_data'][0])>0:
                mf.ModflowGhb(self.mf_model,**self.dict_obj.ghb_dict)
        mf.ModflowOc(self.mf_model,**self.dict_obj.oc_dict)
        
        if self.dict_obj.run_swi:
            mf.ModflowSwi2(self.mf_model,**self.dict_obj.swi_dict)
    def run(self,output_exts=['.hds','.ddn','.cbc','.list','.zta']):
        
        self.model_init()
        self.make_packages()
        if self.output_path is not None:
            # Note, not currently implemented for swi2 output files
            self.mf_model.oc.file_name = [cmfu.change_dir(f1,new_outdir=self.output_path,old_outdir=self.mf_model.model_ws) \
                                            if os.path.splitext(f1)[1] in output_exts else f1 for f1 in self.mf_model.oc.file_name]
            self.mf_model.output_fnames =  [cmfu.change_dir(f1,new_outdir=self.output_path,old_outdir=self.mf_model.model_ws) \
                                            if os.path.splitext(f1)[1] in output_exts else f1 for f1 in self.mf_model.output_fnames]                                   
            self.mf_model.lst.file_name[0] = cmfu.change_dir(self.mf_model.lst.file_name[0],new_outdir=self.output_path,old_outdir=self.mf_model.model_ws)
            
            

class Model_run(object):
    '''
    Initiate flopy Modflow execution using model objects created with Model_maker
    '''
    def __init__(self,model_obj=None,run_mf=True,inputs_exist=False):
        self.model_obj=model_obj
        self.run_mf = run_mf
        self.inputs_exist = inputs_exist
    
    def write_input(self,SelPackList=False):
        self.model_obj.mf_model.write_input(SelPackList=SelPackList)
        self.inputs_exist=True
    
    def save_model_ref(self,model_info_dict=None):
        # Save usgs.model.reference file.
        
        cgu.write_model_ref(model_info_dict=model_info_dict)  
        # update spatial reference info in mf_model
        
        self.model_obj.mf_model._sr = fu.SpatialReference(delr=self.model_obj.mf_model.dis.delr[0],
                                                          delc=self.model_obj.mf_model.dis.delc[0],
                                                          lenuni=model_info_dict['length_units'],
                                                          xul=model_info_dict['xul'],
                                                          yul=model_info_dict['yul'],
                                                          rotation=np.rad2deg(model_info_dict['rotation']),
                                                          proj4_str=model_info_dict['proj'])                                           
        return
    
    def save_model_bounds(self,xycorners=None,XY_cc=None,XY_proj=None,save_active_poly=True,inproj=None,save_bounds=True,
                          proj_kwargs=None):
        ''' Save model bounds to shapefiles.
        
        XY_cc: list of np.ndarray's
                XY_cc should be the grid x,y coordinates for the model coordinate system
                for best results (i.e., m_domain.cc)
        '''
        model_name=self.model_obj.info_obj.model_name
        out_fname = os.path.join(os.path.dirname(os.path.dirname(self.model_obj.output_path)),
                                 'georef',
                                 '{}.shp'.format(model_name))

        col_name_order = ['id','delrc_m']
        data = [[int(model_name.split('_')[1]),int(model_name.split('_')[-1][:-1])]]

        #field_dict = cfu.df_field_dict(None,col_names=col_name_order,col_types=['int','int'])
        shp_dict = {'out_fname':out_fname,
                    'col_name_order':col_name_order,'data':data}
         
        if save_bounds:
#            try:
            if not os.path.exists(os.path.dirname(out_fname)):
                os.makedirs(os.path.dirname(out_fname))
            cfu.write_model_bound_shp(xycorners,**shp_dict)
#            except:
#                print('Writing bounds did not work.')
        
        # Save outline of active model cells to shapefile        
        if save_active_poly:
            active_out_fname = os.path.join(os.path.dirname(os.path.dirname(self.model_obj.output_path)),
                                 'active_bounds',
                                 '{}.shp'.format(model_name))
            
            if hasattr(self,'out_tifs'):
                cs,gt = cru.load_grid_prj(self.out_tifs[-1],gt_out=True)
            
                Zarray = self.model_obj.dict_obj.cell_types.copy()
                in_dict = {'XY':XY_proj,'Z':Zarray,'in_proj':inproj,'out_shp':active_out_fname,
                           'gt':gt}
                cfu.raster_to_polygon_gdal(**in_dict)
            else:
                print('Warning: No raster to extract geodata from. Skipping active bounds.')
#            Zarray[Zarray==0.] = np.nan
#            
#            active_shp,zval,ncells = cfu.raster_to_polygon(XY=XY_cc,Z=Zarray,
#                                                           cell_spacing=self.model_obj.dict_obj.dis_obj.cell_spacing,
#                                                           unq_Z_vals=False)
#            proj_kwargs['proj_out'] = None # keep projected coordinate system
#            proj_dict= {'polys':active_shp,'proj_kwargs':proj_kwargs,'model_coords':True}
#            active_shp = cfu.proj_polys(**proj_dict)
#            col_name_order.append('Ncells')
#            data.append(ncells)
#            field_dict = cfu.df_field_dict(None,col_names=col_name_order,col_types=['int','int','int'])
#            active_shp_dict = {'polys':active_shp,'data':[data],'out_fname':active_out_fname,
#                               'field_dict':field_dict,'col_name_order':col_name_order,
#                               'write_prj_file':True,'inproj':inproj,'write_prj_file':True}            
#            cfu.write_shp(**active_shp_dict)
            
    
    def save_head_geotiff(self,save_ref_kwargs=None,XY_proj=None,save_all=True):
        load_hds_dict = {'model_name':self.model_obj.info_obj.model_name,
                         'workspace':self.model_obj.output_path,'calc_wt':True}
        
        input_kwargs = {'ref_dict':save_ref_kwargs['model_info_dict'],
                        'XY':XY_proj, 'load_hds_dict':load_hds_dict,
                        'dxdy':[self.model_obj.mf_model.dis.delc[0],self.model_obj.mf_model.dis.delr[0]],
                        'active_cells':self.model_obj.dict_obj.cell_types,
                        'elevation':self.model_obj.dict_obj.ztop,
                        'save_all':save_all}
        self.out_tifs = cmfu.save_head_geotiff(**input_kwargs)
            
    def run(self,mf_silent=True,mf_pause=False,
            save_model_ref=True,save_ref_kwargs=None,
            mf_report=True,mf_exception=True,
            xycorners=None,XY_cc=None,XY_proj=None,
            save_active_poly=True,inproj=None,outproj='NAD83'):
        print("Prepping MODFLOW files: {}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
            
        if self.run_mf:
#            print "Writing MODFLOW files if needed: {}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
            if not self.inputs_exist:
                self.write_input()

            # Try to delete the output files, to prevent accidental use of older files
            workspace = self.model_obj.output_path
            model_name = self.model_obj.info_obj.model_name
            try:
                os.remove(os.path.join(workspace, model_name + '.hds'))
                os.remove(os.path.join(workspace, model_name + '.cbc'))
                if self.model_obj.dict_obj.run_swi:
                    os.remove(os.path.join(workspace, '{}.zta'.format(model_name)))
            except:
                pass
            
            # Run the model
            print("Running model {}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
            model_start = time.time()
            self.success, self.mfoutput = self.model_obj.mf_model.run_model(silent=mf_silent, pause=mf_pause, report=mf_report)
            model_elapsed = time.time()-model_start
            if not self.success and mf_exception:
                raise Exception('MODFLOW did not terminate normally.')
            
            self.list_df = cmfu.load_list_budget(model_name=model_name,workspace=workspace)
            if self.list_df is not None:
                self.list_df['success'] = self.success
                self.list_df['SolTime_s'] = model_elapsed
            
            self.save_head_geotiff(save_ref_kwargs=save_ref_kwargs,XY_proj=XY_proj)
            
        if save_model_ref:
            self.save_model_ref(**save_ref_kwargs)
            if xycorners is not None:
                proj_kwargs = {'proj_in':inproj,
                               'xyul':[save_ref_kwargs['model_info_dict']['xul'],save_ref_kwargs['model_info_dict']['yul']],
                               'xyul_m':[XY_cc[0][0,0],XY_cc[1][0,0]],
                               'rotation':-save_ref_kwargs['model_info_dict']['rotation'],'proj_out':outproj}
                self.save_model_bounds(xycorners=xycorners,XY_cc=XY_cc,XY_proj=XY_proj,
                                       save_active_poly=save_active_poly,
                                       inproj=inproj,proj_kwargs=proj_kwargs)
                
class Model_dis(object):
    '''
    Organization class for discretization of Modflow model
    '''
    def __init__(self,cell_spacing=None, nlay=1, nrow=1, ncol=1,delv=None,
                 zthick_elev_min=None, min_elev=None,
                 length_unit=2):
                     
        self.cell_spacing = cell_spacing
        
        if isinstance(self.cell_spacing,(float,int)):
            self.delr = self.cell_spacing
            self.delc = self.cell_spacing
        elif (self.cell_spacing is not None):
            self.delr = self.cell_spacing[0]
            self.delc = self.cell_spacing[1]
        
        self.nlay = nlay
        self.nrow = nrow
        self.ncol = ncol
        self.delv = delv
        self.zthick_elev_min = zthick_elev_min
        self.min_elev = min_elev
        self.length_unit = length_unit # 0=undefined,1=ft,2=m
    
    def time_dis(self,nper=1,perlen=1,nstp=1,steady=True,time_unit=4):
        self.nper = nper
        self.perlen = perlen
        self.nstp = nstp
        self.steady = steady
        self.time_unit = time_unit # 2=minutes,3=hrs,4=days,5 yrs
        self.time_dict = {'itmuni':self.time_unit,'nper':self.nper,
                          'perlen':self.perlen,'nstp':self.nstp,'steady':self.steady}
        
    
    def output_control(self, oc_in_dict=None):
        self.oc_in_dict = oc_in_dict
        if (self.oc_in_dict is None) and hasattr(self,'nstp'):
            self.oc_in_dict = {(self.nper-1, self.nstp-1): ['save head', 'save budget']}   # save only last step
        else:
            raise ValueError("Run time_dis first to supply time period information or supply oc_dict")
            
            
class Model_dict(object):
    '''
    Create python dictionaries of flopy model components for quick flopy module
    calls (e.g., mf.ModflowRch(mf_model,**Model_dict_obj.rchg_dict))

    Depends on Model_dis    
    '''
    def __init__(self, dis_obj=None, cell_types=None,
                 elev_array=None, rchg_array=None,
                 hk = None, k_decay=None, porosity=None,
                 cbc_unit=53, run_swi=False, solver='gmg',
                 rho_f=1e3,rho_s=1025.,sea_level=0.):
        
        self.dis_obj = dis_obj
        self.cell_types = cell_types
        self.cbc_unit = cbc_unit
        self.run_swi = run_swi
        self.solver=solver
        
        self.elev_array = elev_array
        self.rchg_array = rchg_array
        self.hk = hk
        self.porosity = porosity
        self.k_decay = k_decay   

        self.rho_f = rho_f
        self.rho_s = rho_s
        self.sea_level = sea_level            
           
    def dis(self,zbot=None,zthick=None,min_zthick=1.,
            time_dis_options=None,
            smooth_zbot=True,laycbd=0,quasi3dthick=0.0,
            nlayer_continuous=None):
        '''
        Create the top and layers of Modflow domain. Other inputs for the DIS
        package are in dis_obj created by Model_dis above.
        '''
           
        self.ztop = self.elev_array.copy()
        self.ztop = cgu.fill_mask(self.ztop,fill_value=0.)
        if (zbot is None) and (zthick is None):
            self.zbot = cmfu.make_zbot(self.ztop,
                                       [self.dis_obj.nlay,self.dis_obj.nrow,self.dis_obj.ncol],
                                       self.dis_obj.delv,self.dis_obj.zthick_elev_min)
            zbot = self.zbot.copy()
        elif (zbot is not None):
            
            self.zbot = zbot.copy()
            if hasattr(self,'conversion_mask'):
                self.zbot = cgu.shrink_ndarray(self.zbot,self.conversion_mask,self.elev_array.shape)
            self.dis_obj.nlay, self.dis_obj.nrow, self.dis_obj.ncol = self.zbot.shape
        
        elif (zthick is not None) and (zbot is None):
            if hasattr(self,'conversion_mask'):
                zthick2 = cgu.shrink_ndarray(zthick,self.conversion_mask,self.elev_array.shape)            
            else:
                zthick2 = zthick.copy()
                
            self.cell_types[np.sum(zthick2,axis=0)==0] = cgu.grid_type_dict['inactive']
            with np.errstate(invalid='ignore'):
                find_nan_thick = (zthick2<=0) | np.isnan(zthick2)
            zthick_adj = zthick2.copy()
            zthick_adj[find_nan_thick] = min_zthick

            self.zbot = self.ztop-np.cumsum(zthick_adj,axis=0) # bottom elevations
            # Use up to nlay layers if nlay assigned before this step
            if self.dis_obj.nlay is not None and self.dis_obj.nlay>1:
                self.zbot = self.zbot[:self.dis_obj.nlay]
                
            self.dis_obj.nlay = self.zbot.shape[0]
            self.zbot=cmfu.adj_zbot(self.zbot,self.ztop,self.dis_obj.zthick_elev_min)
            zbot = self.zbot.copy()
        
        
        
        if smooth_zbot:
            self.zbot = cru.smooth_array(self.zbot)
        
        self.zbot=cgu.fill_mask(self.zbot,fill_value=0.)
        
        # Lower bottom of first layer below ztop - may have occurred during smoothing
        
        if len(self.zbot.shape)>2:
            bool_thick = (self.ztop-self.zbot[0])<min_zthick
            self.zbot[0,bool_thick] = self.ztop[bool_thick]-min_zthick
        else:
            bool_thick = (self.ztop-self.zbot)<min_zthick
            self.zbot[bool_thick] = self.ztop[bool_thick]-min_zthick
            self.zbot = self.zbot.reshape([self.dis_obj.nlay,self.dis_obj.nrow,self.dis_obj.ncol])
        
        # Make layers continuous
        if nlayer_continuous is not None:
            zbot_new = np.ma.masked_array(self.zbot,mask=np.tile(self.cell_types==cgu.grid_type_dict['inactive'],np.hstack([self.zbot.shape[0],1,1])))
            self.zbot = cgu.fill_mask(cmfu.zbot_topofix(zbot=zbot_new,ztop=self.ztop,
                                          zadjust=min_zthick,
                                          lay2fix=np.arange(nlayer_continuous)),0)
        all_zlayers = np.vstack([self.ztop.reshape(np.hstack((1,self.ztop.shape))),self.zbot])
        self.zbot = cru.make_layers_deeper(all_zlayers,deeper_amount=min_zthick)[1:,:,:]
        
        if isinstance(laycbd,(np.ndarray,list)):
            quasi3dthick_array = cgu.to_nD(quasi3dthick,self.zbot.shape)
            temp_zbot_thick = np.diff(np.vstack([self.zbot[::-1],self.ztop.reshape(np.hstack((1,self.ztop.shape)))]),axis=0)[::-1]
            quasi3dthick_array[temp_zbot_thick<=min_zthick] = 1e-5
            quasi3dthick_array[-1] = 0.
            # Need to add quasi3d bottom elevations
            new_zbot = np.zeros(np.hstack([self.zbot.shape[0]+len((laycbd!=0).nonzero()[0]),self.zbot.shape[1:]]))
            icount=-1
            for ilay in range(self.zbot.shape[0]):
                icount+=1
                if np.mean(laycbd[ilay])!=0.0:
                    # Only assign confining bed for layer with n x the thickness of the layer
                    thicker_layer = (temp_zbot_thick[ilay]>4.*quasi3dthick_array[ilay])
                    new_zbot[icount] = self.zbot[ilay]+1e-5
                    new_zbot[icount,thicker_layer] = self.zbot[ilay,thicker_layer]+quasi3dthick_array[ilay,thicker_layer]
                    
                    # Assign bottom elevation of confining unit as original layer bottom                    
                    icount+=1
                    new_zbot[icount] = self.zbot[ilay]
                else:
                    # No confining unit, keep same bottom
                    new_zbot[icount] = self.zbot[ilay]
             
            out_zbot = new_zbot
        else:
            out_zbot = self.zbot.copy()

        #self.ztop[self.ztop<self.zbot[0]] = self.zbot[0,self.ztop<self.zbot[0]]+min_zthick # opposite of above fix to zbot but changes surface elevation
        self.dis_dict = {'nlay':self.dis_obj.nlay,'nrow':self.dis_obj.nrow,
                         'ncol':self.dis_obj.ncol,'delr':self.dis_obj.delr,
                         'delc':self.dis_obj.delc,
                         'top':self.ztop,
                         'botm':out_zbot.reshape([self.dis_obj.nlay,self.dis_obj.nrow,self.dis_obj.ncol]),'laycbd':laycbd,
                         'lenuni':self.dis_obj.length_unit}
        
        # Add time discretization dictionary and run dis_obj.time_dis if necessary                 
        if not hasattr(self.dis_obj,'nper'): # time_dict not yet assigned
            if (time_dis_options is not None):
                self.dis_obj.time_dis(**time_dis_options)
            else:
                self.dis_obj.time_dis()
        
        self.dis_dict.update(self.dis_obj.time_dict)
    
    def bas(self,hk_ibound_clip=True,
            high_elev=None,min_thick_calc=None,min_surf_thick=50.,
            ibound_thick_threshold=True,min_area=1e4,clean_ibound_switch=True,
            clean_by_layer=True,ibound_minelev_threshold=None,check_inactive=True,
            use_fweq_head=True,set_marine_to_constant=False):
        '''
        Create ibound and starting heads for Modflow BAS package

        Requires output from self.dis() and a flow package (self.bcf or self.upw)
        ***Note*** .bas() can also run .dis()
        '''
        
        if not hasattr(self,'ztop'):
            # Run dis function to make ztop and zbot
            self.dis()

        inactive_val = cgu.grid_type_dict['inactive']
        
        self.ibound = np.ones(np.hstack([self.dis_obj.nlay,self.cell_types.shape]))
        self.dis_obj.nlay,self.dis_obj.nrow,self.dis_obj.ncol = self.ibound.shape
        if hasattr(self,'conversion_mask'):
            self.ibound = cgu.shrink_ndarray(self.ibound,self.conversion_mask,self.cell_types.shape)
                
        self.ibound[:,self.cell_types==inactive_val] = 0 # set inactive cells
        if self.dis_obj.min_elev is not None:
            self.ibound[:,(self.ztop<self.dis_obj.min_elev)\
                        | np.isnan(self.elev_array)] = 0 # all cells with depth less than min_elev set to inactive (i.e., deep water)
        else:
            self.ibound[:,np.isnan(self.elev_array)] = 0
            
        # Set areas with high elevation and thin  domain to no flow (e.g., poorly constrained mountainous areas) 
        if (ibound_minelev_threshold is not None):
            # Compare lowest cell elevations with threshold at set to no flow
            self.ibound[:,self.zbot[-1]>=ibound_minelev_threshold] = 0
            self.cell_types[self.ibound[0]==0] = inactive_val
        
        # Set ibound to inactive where K data is absent
        if hk_ibound_clip:
            with np.errstate(invalid='ignore'):
                self.cell_types[(self.hk[0]<=0) | np.isnan(self.hk[0])] = inactive_val
                self.ibound[:,self.cell_types==inactive_val] = 0
                self.ibound[(self.hk<=0) | np.isnan(self.hk)] = 0
                # Set nan K values to small values (in m/day)
                self.hk[np.isnan(self.hk) | (self.hk<=0)]=1e-3
                self.vk[np.isnan(self.vk) | (self.vk<=0)]=1e-4
        
        if clean_ibound_switch and not ibound_thick_threshold:
            if clean_by_layer:
                for ilay in np.arange(self.ibound.shape[0]):
                    self.ibound[ilay] = cmfu.clean_ibound(self.ibound[ilay],check_inactive=check_inactive,min_area=min_area)
            else:
                # Only clean using first layer
                self.ibound[0] = cmfu.clean_ibound(self.ibound[0],check_inactive=check_inactive,min_area=min_area)
            self.cell_types[self.ibound[0]==0] = inactive_val
            self.cell_types[(self.cell_types==inactive_val) & (self.ibound[0]!=0)] = cgu.grid_type_dict['active']
            
        # propagate no flow down section
        for ilay in range(self.ibound.shape[0]-1):
            self.ibound[ilay:,self.ibound[ilay]==0] = 0
        
        if ibound_thick_threshold and self.dis_obj.nlay>1:
            # Set ibound to inactive where model domain is thin due to inactive lower layers
            temp_zbot_thick = np.diff(np.vstack([self.zbot[::-1],self.ztop.reshape(np.hstack((1,self.ztop.shape)))]),axis=0)[::-1]
            if (min_thick_calc is None):            
                min_thick_calc = temp_zbot_thick.min(axis=0)
            
            min_thick_calc = cgu.to_nD(min_thick_calc,self.ibound.shape)
            min_total_thick = np.sum(min_thick_calc[1:],axis=0)
            temp_zbot2 = np.sum(temp_zbot_thick[1:],axis=0)
            
            # Set grid to inactive where subsurface layers are all thinner than minimum threshold
            self.ibound[1:,temp_zbot2 <= min_total_thick] = inactive_val
            self.ibound[0,(self.ibound[1]==inactive_val) & (temp_zbot_thick[0]<min_surf_thick)] = inactive_val # Note: doesn't allow top layer to be only active layer       
            if clean_by_layer:
                for ilay in np.arange(self.ibound.shape[0]):
                    self.ibound[ilay] = cmfu.clean_ibound(self.ibound[ilay],check_inactive=True,min_area=min_area)
            else:
                # Only clean using first layer
                self.ibound[0] = cmfu.clean_ibound(self.ibound[0],check_inactive=True,min_area=min_area)
            
            self.cell_types[self.ibound[0]==0] = inactive_val
            
            # propagate no flow down section, take 2
            for ilay in range(self.ibound.shape[0]-1):
                self.ibound[ilay:,self.ibound[ilay]==inactive_val] = inactive_val       
         

            
        # Force inactive cell_types to ibound one more time
        self.ibound[:,self.cell_types==inactive_val] = 0
        
        # Remove anisotropy from thin layers at high elevation to allow excess recharge to discharge far from coast        
        if (high_elev is not None):        
            high_and_thin_cells = (self.ztop>high_elev) &\
                                  (temp_zbot_thick[0]<=min_thick_calc[0]*5.)
            self.vk[0,high_and_thin_cells] = self.hk[0,high_and_thin_cells]
        
        # Make starting heads
        if self.dis_obj.nlay == 1:
            self.starting_head = cgu.fill_mask(self.ztop,fill_value=0.) # top of domain, hydrostatic
        else:
            self.starting_head = []# for multiple layers
            ztoptemp = cgu.fill_mask(self.ztop,fill_value=0.)
            for ilay in np.arange(self.dis_obj.nlay):
                if ilay==0:
                    self.starting_head.append(ztoptemp)
                else:
                    self.starting_head.append(self.zbot[ilay-1,:,:])
    
        self.starting_head = np.array(self.starting_head).reshape((self.dis_obj.nlay,self.dis_obj.nrow,self.dis_obj.ncol))
        self.starting_head[np.isnan(self.starting_head)] = 0.
        
        if use_fweq_head:
            water_height = self.sea_level-self.ztop
            fw_head = cgu.calc_fw_head(water_height,self.ztop,rho_f=self.rho_f,rho_s=self.rho_s)
            self.starting_head[0,self.ztop<self.sea_level] = fw_head[self.ztop<self.sea_level]
        
        if set_marine_to_constant:
            if isinstance(self.sea_level.shape,(float,int)):
                self.starting_head[0,self.ztop<=self.sea_level] = self.sea_level
            else:
                self.starting_head[0,self.ztop<=self.sea_level] = self.sea_level[self.ztop<=self.sea_level]
            self.ibound[0,(self.ztop<=self.sea_level) & (self.cell_types!=inactive_val)] = -1
        
        #self.starting_head[1:] = cru.smooth_array(self.starting_head[1:]) # use if applying fw head to all layers
        self.bas_dict = {'ibound':self.ibound,'strt':self.starting_head}
        
    def rchg(self,rchg_units=[2,4],rchg_ratio=1., wb_rchg_ratio=None, nrchop=3,
             reduce_lowk_rchg=False,lowk_rchg_ratio=1.,lowk_val=1e-3,lowk_lay=0):
        '''
        Create recharge array for Modflow RCH package        
        '''        
        if rchg_units != [2,4]:
            # need to convert...
            if rchg_units == [2,5]:
                 self.rchg_array =  self.rchg_array * (1./365.25) # m/day from m/yr
            else:
                print("Recharge: check units")
                
        if hasattr(self.rchg_array,'mask'):
            with np.errstate(invalid='ignore'):
                self.rchg_array = np.ma.filled(self.rchg_array,fill_value=0.)
        
        with np.errstate(invalid='ignore'):        
            self.rchg_array[np.isnan(self.rchg_array) | (self.rchg_array<0.)]= 0.
        self.rchg_array = rchg_ratio *  self.rchg_array        
        
        # Change recharge to waterbodies
        if (wb_rchg_ratio is not None):
            self.rchg_array[self.cell_types==cgu.grid_type_dict['waterbody']] = wb_rchg_ratio*self.rchg_array[self.cell_types==cgu.grid_type_dict['waterbody']] 
        
        # No recharge to nearshore waters (i.e., GHBs)
        self.rchg_array[self.cell_types==cgu.grid_type_dict['nearshore']] = 0.
        
        if reduce_lowk_rchg:
            self.rchg_array[self.hk[lowk_lay,:,:]<=lowk_val] = lowk_rchg_ratio * self.rchg_array[self.hk[lowk_lay,:,:]<=lowk_val]
        
        self.rchg_dict = {'ipakcb':self.cbc_unit,'rech':cgu.fill_mask(self.rchg_array,fill_value=0.),'nrchop':nrchop}

    def drn(self,z_bed_thickness=1.,dampener=1.,run_bcf=False,
            elev_cutoff=1e2,elev_damp=1.,min_z = 1e-2,noprint=True):
        '''
        Create stress period data for Modflow DRN package using cell_types        
        '''
        if ~hasattr(self,'vk') and run_bcf:
            self.bcf() # make k information

        surface_conductance = (self.dis_obj.cell_spacing**2.*self.vk[0,:,:].squeeze())/((self.ztop-self.zbot[0])/2.) # units of L^2/time
        
        drn_indexes = np.array((~np.isnan(surface_conductance) & (self.ztop>min_z) & \
                                ((self.cell_types==cgu.grid_type_dict['river']) \
                             | (self.cell_types==cgu.grid_type_dict['active']) \
                             | (self.cell_types==cgu.grid_type_dict['waterbody']))).nonzero()).T
        
        drn_stress_array = np.hstack([np.zeros((drn_indexes.shape[0],1)), \
                        drn_indexes[:,0].reshape((-1,1)), \
                        drn_indexes[:,1].reshape((-1,1)), \
                        self.ztop[drn_indexes[:,0],drn_indexes[:,1]].reshape((-1,1)), \
                        dampener*surface_conductance[drn_indexes[:,0],drn_indexes[:,1]].reshape((-1,1))])
        
        # Change conductance for high elevation drains
        if (elev_cutoff is not None):
            drn_stress_array[drn_stress_array[:,3]>elev_cutoff,4] = elev_damp*drn_stress_array[drn_stress_array[:,3]>elev_cutoff,4]                
        
        drn_stress_dict = {0:drn_stress_array.tolist()} # only for steady state, or constant drn conditions
        
        self.drn_dict = {'ipakcb':self.cbc_unit,'stress_period_data':drn_stress_dict}
        if noprint:
            self.drn_dict.update({'options':['NOPRINT']})
        
    def ghb(self,z_bed_thickness=1.,dampener=1.,min_z = 1e-2,
            outer_bound_bool=False,new_zthick=None,marine_bool=False,max_marine_lay=None,
            noprint=True,slr=0, skip_ghb=False):
        '''
        Create stress period data for Modflow GHB package using cell_types        
        '''
        if not skip_ghb:
            ghb_indexes = np.array(((self.cell_types==cgu.grid_type_dict['nearshore']) \
                                        & (self.ztop<=self.sea_level)).nonzero()).T
            
            # Calculate freshwater head using bathymetry
            ztop_temp = cgu.fill_mask(self.ztop,fill_value=0.)[ghb_indexes[:,0],ghb_indexes[:,1]].reshape((-1,1))
            
            if isinstance(self.sea_level,(float,int)):
                # Force cells w/ elev very near sea_level to be min_z less than sea_level
                ztop_temp[np.abs(ztop_temp)<=(self.sea_level+min_z)] = self.sea_level-min_z
                water_height = slr+self.sea_level-ztop_temp
    #            water_head = self.sea_level*np.ones_like(ztop_temp)
            else:
                # add slr as higher sea level only for ghb cells using present-day sl (hold ground scenario)
                temp_sl = slr+self.sea_level[ghb_indexes[:,0],ghb_indexes[:,1]].reshape((-1,1))
                ztop_temp[np.abs(ztop_temp)<=(temp_sl+min_z)] = temp_sl[np.abs(ztop_temp)<=(temp_sl+min_z)]-min_z # 
                water_height = temp_sl-ztop_temp
    #            water_head = temp_sl.copy()
            
            if isinstance(self.rho_s,(float,int)):
                fw_head = cgu.calc_fw_head(water_height,ztop_temp,rho_f=self.rho_f,rho_s=self.rho_s)
            else:
                fw_head = cgu.calc_fw_head(water_height,ztop_temp,rho_f=self.rho_f,rho_s=self.rho_s[ghb_indexes[:,0],ghb_indexes[:,1]].reshape((-1,1)))
                
            surface_conductance = (self.dis_obj.cell_spacing**2.*self.vk[0,:,:].squeeze())/z_bed_thickness # units of L^2/time
            surface_conductance[np.isnan(surface_conductance)] = np.nanmin(surface_conductance)
            ghb_stress_array = np.hstack([np.zeros((ghb_indexes.shape[0],1)), \
                            ghb_indexes[:,0].reshape((-1,1)), \
                            ghb_indexes[:,1].reshape((-1,1)), \
                            fw_head, \
                            dampener*surface_conductance[ghb_indexes[:,0],ghb_indexes[:,1]].reshape((-1,1))])
            
            if outer_bound_bool:
                if new_zthick.shape != self.zbot.shape:
                    new_zthick = cgu.shrink_ndarray(new_zthick,self.conversion_mask,self.cell_types.shape)
                
                outer_dict = {'ztop':self.ztop,'zbot':self.zbot,'zthick':new_zthick,
                          'cell_types':self.cell_types,'dampener':dampener,'self.sea_level':self.sea_level+slr,
                          'hk':self.hk,'cell_spacing':self.dis_obj.cell_spacing,
                          'marine_bool':marine_bool,'max_lay':max_marine_lay}
                outer_ghb_array = cmfu.outer_ghb(**outer_dict)
                ghb_stress_array = np.vstack([ghb_stress_array,outer_ghb_array])
            ghb_stress_dict = {0:ghb_stress_array.tolist()}
            
            self.ghb_dict = {'ipakcb':self.cbc_unit,'stress_period_data':ghb_stress_dict}
            if noprint:
                self.ghb_dict.update({'options':['NOPRINT']})
        else:
            self.ghb_dict = None
    
    def bcf(self,hk_in=None,hk_botm=None,vk_in=None,vk_array=None,v_ani_ratio = 1.,
            h_ani_ratio=1.,min_hk=None,wb_k_change_log10=None,
            res_increase_log10=1,tran=0.,iwdflg=0,hk_extent_clip=True,
            z_is_thick=True, calc_vcont_flag=True, default_laytype = 3,
            nan_lower=True,propkdown=True,smooth_k=False,last_layer_nan=True):
        '''
        Create block-centered flow inputs for Modflow BCF package        
        '''
        array_dims = [self.dis_obj.nlay,self.dis_obj.nrow,self.dis_obj.ncol]
        nlay,nrow,ncol = array_dims
        # Assign layer types
        self.layer_type = default_laytype*np.ones((nlay)) # 1 for unconfined (change-able), 0 for always confined, 3 fully convertible
        self.layer_type[0]=1 # force top layer to be unconfined
        
        make_K_arrays_dict = {'hk_in':hk_in,'hk_in_botm':hk_botm,'vk_array':vk_array,
                                  'ztop':self.ztop,'zbot':self.zbot,'k_decay':self.k_decay,
                                  'v_ani_ratio':v_ani_ratio,'h_ani_ratio':h_ani_ratio,
                                  'nan_lower':nan_lower,
                                  'elev_array':self.elev_array,
                                  'z_is_thick':z_is_thick, 'calc_vcont_flag':calc_vcont_flag,
                                  'last_layer_nan':last_layer_nan}
                                  
        if hasattr(self,'conversion_mask'):
            make_K_arrays_dict.update({'conversion_mask':self.conversion_mask})
                                  
        if (vk_in is not None):
            # If vertical k array provided, calculate hk and vk separately                            
            self.hk,_,self.vcont = cmfu.make_K_arrays(array_dims,self.hk,**make_K_arrays_dict)
            args = {'mask_in':self.conversion_mask,'shape_out':self.cell_types.shape}            
            self.vk = cmfu.K_to_layers(self.elev_array,self.zbot,
                                         cgu.shrink_ndarray(vk_in,**args),
                                         cgu.shrink_ndarray(hk_botm,**args) ,
                                         z_is_thick=True)
        else:
            self.hk,self.vk,self.vcont = cmfu.make_K_arrays(array_dims,self.hk,**make_K_arrays_dict)
        
        # Use where isnan(hk) to set inactive parts of domain
        if hk_extent_clip:
            self.cell_types[np.all(np.isnan(self.hk),axis=0)]=cgu.grid_type_dict['inactive']
            self.cell_types[np.isnan(self.cell_types)] = cgu.grid_type_dict['inactive']
            
        # Change k of waterbodies
        if (wb_k_change_log10 is not None):
            # Increase resistance due to fine grain materials across lakebed
            self.vk[0,self.cell_types==cgu.grid_type_dict['waterbody']] = (10.**(-wb_k_change_log10-res_increase_log10)) * self.vk[0,self.cell_types==cgu.grid_type_dict['waterbody']]
            # Adjust waterbody K to change flow through conditions
            self.hk[0,self.cell_types==cgu.grid_type_dict['waterbody']] = (10.**wb_k_change_log10) * self.hk[0,self.cell_types==cgu.grid_type_dict['waterbody']]
         
        # Fill nans
        if (min_hk is not None):
            with np.errstate(invalid='ignore'):
                self.hk[np.isnan(self.hk) | (self.hk < min_hk)]=min_hk
                self.vk[np.isnan(self.vk) | (self.vk < (min_hk*1e-1))] = min_hk*1e-1
        
        if smooth_k:
            self.hk = cru.smooth_array(self.hk)
            self.vk = cru.smooth_array(self.vk)        
        
        self.vcont = cgu.fill_mask(self.vcont,fill_value=np.nanmin(self.vcont))  
            
        self.bcf_dict = {'hy':self.hk,'laycon':self.layer_type,
                         'trpy':h_ani_ratio,
                         'vcont':self.vcont,'tran':tran,'iwdflg':iwdflg,
                         'ipakcb':self.cbc_unit}            
    
    def upw(self,hk_in=None,hk_botm=None,vk_in=None,vk_array=None,v_ani_ratio = 1.,
            h_ani_ratio=1.,min_hk=None,wb_k_change_log10=None,
            res_increase_log10=1,layvka=0,laywet=0,hk_extent_clip=True,
            z_is_thick=True, default_laytype = 3,calc_vcont_flag=False,
            nan_lower=True,propkdown=True,vkcb=0.0,smooth_k=False,
            iphdry=1,last_layer_nan=True):
        ''' Create upstream weighting package, required for using NWT solver 
        
        Note: iphdry must be set to >0 for Modpath to work correctly.
        '''
        
        array_dims = [self.dis_obj.nlay,self.dis_obj.nrow,self.dis_obj.ncol]
        nlay,nrow,ncol = array_dims
        # Assign layer types
        self.layer_type = default_laytype*np.ones((nlay)) # 1 for unconfined (change-able), 0 for always confined, 3 fully convertible
        self.layer_type[0]=1 # force top layer to be unconfined
        
        make_K_arrays_dict = {'hk_in':hk_in,'hk_in_botm':hk_botm,'vk_array':vk_array,
                                  'ztop':self.ztop,'zbot':self.zbot,'k_decay':self.k_decay,
                                  'v_ani_ratio':v_ani_ratio,'h_ani_ratio':h_ani_ratio,
                                  'elev_array':self.elev_array,
                                  'z_is_thick':z_is_thick, 'calc_vcont_flag':calc_vcont_flag,
                                  'nan_lower':nan_lower,'propkdown':propkdown,
                                  'last_layer_nan':last_layer_nan}
                                  
        if hasattr(self,'conversion_mask'):
            make_K_arrays_dict.update({'conversion_mask':self.conversion_mask})
        
        if (vk_in is not None):
            # If vertical k array provided, calculate hk and vk separately                            
            self.hk,_,self.vcont = cmfu.make_K_arrays(array_dims,self.hk,**make_K_arrays_dict)
            args = {'mask_in':self.conversion_mask,'shape_out':self.cell_types.shape}            
            self.vk = cmfu.K_to_layers(self.elev_array,self.zbot,
                                         cgu.shrink_ndarray(vk_in,**args),
                                         cgu.shrink_ndarray(hk_botm,**args) ,
                                         z_is_thick=True)
        else:
            self.hk,self.vk,self.vcont = cmfu.make_K_arrays(array_dims,self.hk,**make_K_arrays_dict)
        
        # Use where isnan(hk) to set inactive parts of domain
        if hk_extent_clip:
            with np.errstate(invalid='ignore'):
                self.cell_types[(np.isnan(self.hk[0]) | (self.hk[0]<=0.))]=cgu.grid_type_dict['inactive']
                self.cell_types[np.isnan(self.cell_types)] = cgu.grid_type_dict['inactive']
            
        # Change k of waterbodies
        if (wb_k_change_log10 is not None):
            # Increase resistance due to fine grain materials across lakebed
            self.vk[0,self.cell_types==cgu.grid_type_dict['waterbody']] = (10.**(-wb_k_change_log10-res_increase_log10)) * self.vk[0,self.cell_types==cgu.grid_type_dict['waterbody']]
            # Adjust waterbody K to change flow through conditions
            self.hk[0,self.cell_types==cgu.grid_type_dict['waterbody']] = (10.**wb_k_change_log10) * self.hk[0,self.cell_types==cgu.grid_type_dict['waterbody']]
        
        
        # Fill nans, necessary for fortran happiness
        if (min_hk is not None):
            with np.errstate(invalid='ignore'):
                self.hk[np.isnan(self.hk) | (self.hk < min_hk)]=min_hk
    #            v_ani_array = cgu.to_nD(v_ani_ratio,array_dims)
    #            self.vk[np.isnan(self.vk)]=min_hk/v_ani_array[np.isnan(self.vk)]
    #            self.vk[np.isnan(self.vk)]=1e-4
                self.vk[np.isnan(self.vk) | (self.vk < (min_hk*1e-1))] = min_hk*1e-1
#        else:
#            self.hk[np.isnan(self.hk)]=1e-3
#            self.vk[np.isnan(self.vk)]=1e-4
         
        #self.vcont = cgu.fill_mask(self.vcont,fill_value=np.nanmin(self.vcont))
        
        if smooth_k:
            self.hk = cru.smooth_array(self.hk)
            self.vk = cru.smooth_array(self.vk)
        
        self.upw_dict = {'hk':self.hk,'laytyp':self.layer_type,
                         'hani':h_ani_ratio, 'sy':self.porosity,
                         'vka':self.vk,'laywet':laywet,'layvka':layvka,
                         'ipakcb':self.cbc_unit,'vkcb':vkcb,'iphdry':iphdry}
        
    def oc(self,oc_in_dict=None,compact=True):
        '''
        Create output control stress period data for Modflow OC package    
        '''
        if (oc_in_dict is None) and ~hasattr(self.dis_obj,'oc_in_dict'):
            # Run output_control
            self.dis_obj.output_control()
        else:
            self.dis_obj.oc_in_dict = oc_in_dict
            
        self.oc_dict = {'stress_period_data':self.dis_obj.oc_in_dict,'compact':compact}
    
    def gmg(self,rclose=1e-2,hclose=1e-2,damp=5e-1,iadamp=1,mxiter=50,iiter=30,
            ioutgmg=0,isc=1,ism=0):
        '''
        Assign model solver parameters for Modflow GMG package        
        '''
        self.gmg_dict = {'rclose':rclose,'hclose':hclose,'damp':damp,
                         'iadamp':iadamp,'mxiter':mxiter,'iiter':iiter,
                         'ioutgmg':ioutgmg,'isc':isc,'ism':ism}
    def nwt(self,headtol=1e-4, fluxtol=500, maxiterout=100, thickfact=1e-05,
            linmeth=1, iprnwt=0, ibotav=0, options='COMPLEX', Continue=False):
        
        self.nwt_dict = {'headtol':headtol,'fluxtol':fluxtol,'maxiterout':maxiterout,
                         'thickfact':thickfact,'linmeth':linmeth,'iprnwt':iprnwt,
                         'ibotav':ibotav,'options':options,'Continue':Continue}
        
    def swi(self,elev_cutoff = None,iswizt=55,
            nsrf=1, istrat=1, nu=None,nsolver=2,
            min_zthick=1.,solver_dict=None,options=None,
            nadptmx=1,nadptmn=1,adptfct=1.0,smooth_zeta=True,force_layer_bounds=False):
        '''
        Create input data for Modflow SWI2 package        
        '''
        if self.run_swi:
            if elev_cutoff is None:
                elev_cutoff = self.sea_level
            if nu is None:
                # Define nu using rho_s and rho_f, use min and max values for now
                nu = [0.,(np.max(self.rho_s)-np.min(self.rho_f))/(np.min(self.rho_f))]
            
            # Set up seawater interface module       
            self.isource = np.zeros_like(self.ibound,dtype=np.int)
#            self.isource[1:,self.cell_types==cgu.grid_type_dict['active']] = 1
            self.isource[:,self.cell_types==cgu.grid_type_dict['nearshore']] = -2
            self.isource[0,(self.cell_types==cgu.grid_type_dict['waterbody']) &\
                            (self.elev_array<=elev_cutoff)] = -2 # make nearshore waterbodies salty
            
            #self.isource[:,(self.cell_types==cgu.grid_type_dict['nearshore']) &\
            #                (self.elev_array<=-20)] = -2 # deep saltwater
                            
            # Starting location for interface (surface)
            self.zeta = np.zeros_like(self.ibound,dtype=np.float32)
            self.non_nearshore = (self.cell_types==cgu.grid_type_dict['river']) \
                             | (self.cell_types==cgu.grid_type_dict['active']) \
                             | (self.cell_types==cgu.grid_type_dict['waterbody'])
            if isinstance(self.sea_level,(float,int)):
                self.zeta[:,self.non_nearshore] = self.sea_level-self.starting_head[0,self.non_nearshore]*(1./nu[-1])
            else:                 
                self.zeta[:,self.non_nearshore] = self.sea_level[self.non_nearshore]-self.starting_head[0,self.non_nearshore]*(1./nu[-1])#[self.non_nearshore]) - for nu as an array, crashes 5/31/18
            
            if smooth_zeta:            
                self.zeta = cru.smooth_array(self.zeta)
            
            if isinstance(self.sea_level,(float,int)):
                self.zeta[:,self.cell_types==cgu.grid_type_dict['nearshore']] = self.sea_level # assigned to sea level
            else:
                self.zeta[:,self.cell_types==cgu.grid_type_dict['nearshore']] = self.sea_level[self.cell_types==cgu.grid_type_dict['nearshore']] # assigned to sea level
            
            # Force zeta to be within layer bounds
            if force_layer_bounds:
                self.zeta = np.maximum(self.zbot,self.zeta)
                self.zeta[1:] = np.minimum(self.zeta[1:],self.zbot[:-1])
#            self.zeta[1:,self.cell_types==cgu.grid_type_dict['nearshore']] = self.zbot[:-1,self.cell_types==cgu.grid_type_dict['nearshore']]

#            self.zeta = cru.make_layers_deeper(self.zeta,deeper_amount=min_zthick)            
            
            if (solver_dict is None):
               solver_dict = {'mxiter':25,'iter1': 500,'npcond': 1, 'zclose': 1e-2,
                   'rclose': 1e-2, 'relax': 1.0, 'nbpol': 2, 'damp': 1.0, 'dampt': 1.0}
            
            self.swi_dict = {'nsrf':nsrf,'istrat':istrat,'nu':nu,
                             'zeta':self.zeta,'isource':self.isource,
                             'ssz':self.porosity,
                             'nsolver':nsolver,'solver2params':solver_dict,
                             'iswizt':iswizt}
            if options is not None:
                self.swi_dict.update({'options':options,'nadptmx':nadptmx,
                                      'nadptmn':nadptmn,'adptfct':adptfct})
                           
    def run_all(self,bas_kwargs=None,dis_kwargs=None,
                rchg_kwargs=None,bcf_kwargs=None,
                drn_kwargs=None,ghb_kwargs=None,gmg_kwargs=None,
                oc_kwargs=None,swi_kwargs=None,upw_kwargs=None,
                nwt_kwargs=None,
                shrink_domain=True):
        
        if (dis_kwargs is not None):
            self.dis(**dis_kwargs)
        else:
            self.dis()
        
#        if shrink_domain:
#           self.conversion_mask = cmfu.shrink_domain(self)
#           apply_dict = {'func':cgu.shrink_ndarray,
#                          'func_args':{'mask_in':self.conversion_mask,
#                          'shape_out':self.cell_types.shape},'skip_vals':['conversion_mask']}
#           cgu.recursive_applybydtype(self,**apply_dict) 
#                                      
#           # Run shrink domain on existing package dictionaries
#           existing_package_dicts = cgu.match_keys(self,'_dict')
#           dict_obj,_=cgu.dict_contents(self)
#           for dict_name in existing_package_dicts:                            
#               cgu.recursive_applybydtype(dict_obj[dict_name],**apply_dict)                           
#           
#           if 'dis_dict' in existing_package_dicts:
#               self.dis_dict['nrow'],self.dis_dict['ncol'] = self.dis_dict['top'].shape
#                
#        
        if self.solver not in ['nwt']:
            if (bcf_kwargs is not None):    
                self.bcf(**bcf_kwargs) # need vk before drn and ghb
            else:
                self.bcf()        
        else:
            if (upw_kwargs is not None):    
                self.upw(**upw_kwargs) # need vk before drn and ghb
            else:
                self.upw()
                    
        if (bas_kwargs is not None): 
            self.bas(**bas_kwargs) # can also run self.dis()
        else:
            self.bas()
            
        if shrink_domain:
           self.conversion_mask = cmfu.shrink_domain(self)
           apply_dict = {'func':cgu.shrink_ndarray,
                          'func_args':{'mask_in':self.conversion_mask,
                          'shape_out':self.cell_types.shape},'skip_vals':['conversion_mask']}
           cgu.recursive_applybydtype(self,**apply_dict) 
                                      
           # Run shrink domain on existing package dictionaries
           existing_package_dicts = cgu.match_keys(self,'_dict')
           dict_obj,_=cgu.dict_contents(self)
           for dict_name in existing_package_dicts:                            
               cgu.recursive_applybydtype(dict_obj[dict_name],**apply_dict)                           
           
           if 'dis_dict' in existing_package_dicts:
               self.dis_dict['nrow'],self.dis_dict['ncol'] = self.dis_dict['top'].shape
                
        
        if (rchg_kwargs is not None):
            self.rchg(**rchg_kwargs)
        else:
            self.rchg()                    
        
        if (drn_kwargs is not None):
            self.drn(**drn_kwargs)
        else:
            self.drn()
        
        if (ghb_kwargs is not None):
            self.ghb(**ghb_kwargs)
        else:
            self.ghb()
        if self.solver not in ['nwt']:
            if (gmg_kwargs is not None):
                self.gmg(**gmg_kwargs)
            else:
                self.gmg()
        else:
            if (nwt_kwargs is not None):
                self.nwt(**nwt_kwargs)
            else:
                self.nwt()
        
        if (oc_kwargs is not None):
            self.oc(**oc_kwargs)
        else:
            self.oc()
            
        if self.run_swi:
            if (swi_kwargs is not None):
                self.swi(**swi_kwargs)
            else:
                self.swi()
        
                
            
            
            
        