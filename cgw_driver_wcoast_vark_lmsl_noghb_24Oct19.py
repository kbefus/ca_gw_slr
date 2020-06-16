# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 13:46:22 2016

@author: kbefus
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 16 09:19:36 2016

@author: kbefus
"""

import sys,os
import numpy as np
import time 
from shutil import copyfile

res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
sys.path.insert(1,code_dir)

from cgw_model import cgw_package_tools as cpt
from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_raster_utils as cru
from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_modflow import cgw_mf_tools as cmft
from cgw_model.cgw_modflow import cgw_mf_utils as cmfu
from cgw_model.cgw_zonebudget import cgw_zb_tools as czbt

import geopandas as gpd
#from cgw_model.cgw_modpath import cgw_mp_tools as cmpt

#%% California inputs
all_start = time.time()  

ca_regions = ['norca','paca','sfbay','cenca','soca']


out_research_dir = 'None'#r'C:\research\CloudStation\research\coastal_gw\ca_slr' 
#out_main_model_dir=os.path.join(out_research_dir,'model')

research_dir_main = os.path.join(res_dir,'ca_slr')
#research_dir = r'C:\research\kbefus\ca_slr'
#research_dir = out_research_dir
research_dir = r'/mnt/762D83B545968C9F'
main_model_dir = os.path.join(research_dir,'model_lmsl_noghb')
out_main_model_dir = main_model_dir

    
data_dir = os.path.join(research_dir_main,'data')
nc_dir = os.path.join(main_model_dir,'nc_inputs')
ref_name = 'usgs.model.reference'


dem_date = '11Feb19'
elev_dir = os.path.join(data_dir,'gw_dems{}'.format(dem_date))
nmodel_domains_shp = os.path.join(data_dir,'ca_{}_slr_gw_domains_{}.shp'.format('n',dem_date))
ndomain_df = gpd.read_file(nmodel_domains_shp)

smodel_domains_shp = os.path.join(data_dir,'ca_{}_slr_gw_domains_{}.shp'.format('s',dem_date))
sdomain_df = gpd.read_file(smodel_domains_shp)

nallmodels = sdomain_df.shape[0]+ndomain_df.shape[0]

id_col = 'Id'
sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])# m

# ----------- Model run options --------------
overwrite_switch = False # Delete model directory and all contents prior to run, ***use at own risk***
force_write_inputs= True # Force writing of modflow input files
plot_results = True # Plot results of modflow model run
prep_zb,run_zb = False,False # Prepare and run zonebudget analysis

run_mf = True
run_mp = False
clear_all = True

load_nc_grids_bool=True # Tries to load grid files from previous run
save_nc_grids=True # automatically switches to false if nc_grid is loaded
model_type = 'fw' # 'swi' or 'fw'

# swi options
run_swi = False
plt_swi=False
zb_swi = False
run_swi_model = False

solver = 'nwt'
use_solver_dir = False


# ----------- Model parameterization -----------
# Flow options
#v_ani = None # if provided by spatial datasets
v_ani = 10. # kh/kv, vertical anisotropy
porosity = 0.2
Kh_vals = [0.1,1.,10.]
#Kh = 1. #m/day, use None if supplying raster data
min_hk = 1e-4
k_decay = 1.

# Discretization options
nlay = 1#None for all, len(layer_thick)
cell_spacing = 10. # meters
layer_thick = 50.

elev_thick_min = -50# if negative: elevation, if positive: thickness, or None
min_surface_elev = None#-100.
min_zthick = .3048

# Time options
nyears = 10. # rough time in years, will be a few days more than this to get an integer number of days
ndays = np.ceil(nyears*365.25)
nper,nstp = 1, 1
perlen = np.int(np.ceil(ndays/nstp)*nstp)
steady=True
rchg_ratio = 1.

# ZoneBudget options
max_zb_layer = 1#m_maker.dict_obj.zbot.shape[0] # None, or 1 to nlay

# Sea level and salinity inputs
datum_type = 'LMSL'
cs_sl_sal = 'NAD83'
sl_fname = os.path.join(data_dir,'sea_level','CA_sl_{}_12Feb18.txt'.format(datum_type))
sl_data,_ = cru.read_txtgrid(sl_fname)
sl_data = np.array(sl_data) # lon, lat, sl

sal_fname = os.path.join(data_dir,'salinity','CA_sal_12Feb18.txt')
sal_data,_ = cru.read_txtgrid(sal_fname)
sal_data = np.array(sal_data) # lon, lat, density

rerun=True
rerun_sl_dens = True

rerun_older_date = "Sep 12 2019 06:00AM"
date_fmt = '%b %d %Y %I:%M%p'
rr_date = time.strptime(rerun_older_date,date_fmt)
rr_date_s = time.mktime(rr_date)

# Make model_management file to keep track of currently and previously run models
active_date = '24Oct19'
model_name_fmt = '{0:s}_{1:d}_{2}_slr{3:3.2f}m_Kh{4:3.2f}_{5:.0f}m'
other_model_name_fmt = '{0:s}_{1:d}_{2}_slr{3:3.2f}m_Kh{4:3.1f}_{5:.0f}m'
dirname_fmt = '_{0}_res{1}m_sl{2:3.2f}m_Kh{3:3.2f}'
other_dirname_fmt = '_{0}_res{1}m_sl{2:3.2f}m_Kh{3:3.1f}' 
#%%
model_mng_file = os.path.join(main_model_dir,'model_management_{0}.txt'.format(active_date))
for Kh in Kh_vals:
    
#    if Kh==Kh_vals[0]:
#        fmt = model_name_fmt
#        otherfmt = other_model_name_fmt
#        dirfmt = dirname_fmt
#    else:
    fmt = other_model_name_fmt
    otherfmt = model_name_fmt
    dirfmt = other_dirname_fmt
    
    for sealevel_elev in sealevel_elevs:
        for ca_region in ca_regions: # loop through CA regions
            # ----------- Region directory information -----------
            
            region_dir = os.path.join(main_model_dir,ca_region)
            out_region_dir = os.path.join(out_main_model_dir,ca_region)
            
            results_dir = os.path.join(region_dir,'output')
            
            use_other_dir = dirfmt.format(datum_type,cell_spacing,sealevel_elev,Kh)#'_{}lay'.format(nlay)
            if use_solver_dir:
                model_inputs_dir =  os.path.join(region_dir,'model{}'.format(solver))
                model_outputs_dir = os.path.join(region_dir,'output{}'.format(solver))
            elif use_other_dir is not None:
                model_inputs_dir =  os.path.join(region_dir,'model{}'.format(use_other_dir))
                model_outputs_dir = os.path.join(region_dir,'output{}'.format(use_other_dir))
            else:
                model_inputs_dir =  os.path.join(region_dir,'model')
                model_outputs_dir = os.path.join(region_dir,'output')
            
            figs_dir = os.path.join(model_outputs_dir,'figures')
                
            for temp_dir in [figs_dir,nc_dir,model_inputs_dir,model_outputs_dir]:
                if not os.path.isdir(temp_dir):
                    os.makedirs(temp_dir)
            
            
            # Define model information for region
            if ca_region in ['soca']:
                domain_df = sdomain_df.copy()
                model_domains_shp = smodel_domains_shp
                r_prefix = 's'
            else:
                domain_df = ndomain_df.copy()
                model_domains_shp = nmodel_domains_shp
                r_prefix = 'n'
                
            # Select only models for current region
            active_models = domain_df.loc[domain_df['ca_region']==ca_region,id_col].values
            nmodels = domain_df.shape[0]
            
            
            budget_outfile = os.path.join(model_outputs_dir,'{}_budget_summary.csv'.format(ca_region))
            if os.path.isfile(budget_outfile):
                model_budget_df = czbt.zbu.pd.read_csv(budget_outfile)
                model_budget_df.set_index('model_name',inplace=True)
            else:
                model_budget_df = czbt.zbu.pd.DataFrame()
            
            # ----------- Supporting spatial data -----------
            rchg_fname = os.path.join(data_dir,"{}_wcoast_rc_eff_0011_utm.tif".format(r_prefix))
            
            # Set model projection for region
            elev_fname = os.path.join(elev_dir,'{0}_{1:02.0f}_dem_landfel.tif'.format(ca_region,active_models[0]))
            temp_proj = cru.gdal.Open(elev_fname)
            in_proj = temp_proj.GetProjectionRef()
            temp_proj = None
            
            # Project salinity and sea-level data
            sal_xy_proj = cru.projectXY(sal_data[:,:2],inproj=cs_sl_sal,outproj=in_proj)
            sl_xy_proj = cru.projectXY(sl_data[:,:2],inproj=cs_sl_sal,outproj=in_proj)
            sal_data_proj = np.column_stack([sal_xy_proj,sal_data[:,2]])
            sl_data_proj = np.column_stack([sl_xy_proj,sl_data[:,2]])
            
            for active_domain in active_models: # or in active_domains
                model_start = time.time()
                
                active_domain_data = domain_df.loc[domain_df[id_col]==active_domain,:]
                
                # Set model output directories
                model_name = fmt.format(ca_region,
                                                active_domain,datum_type,
                                                sealevel_elev,Kh,cell_spacing)
                model_in_dir = os.path.join(model_inputs_dir,model_name)
                model_out_dir = os.path.join(model_outputs_dir,model_name)
                # Print loop info
                print('------------- Model {} of {} -------------'.format(active_domain+1,nallmodels))
                print('Model: {}, sea level = {} m'.format(model_name,sealevel_elev))
                print('Start time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
                
                main_hds = os.path.join(model_out_dir,'{}.hds'.format(model_name))
                out_hds = os.path.join(out_region_dir,os.path.basename(model_outputs_dir),model_name,'{}.hds'.format(model_name))
                new_nc_fname = os.path.join(model_in_dir,'{}.nc'.format(model_name))
                new_nc_folder = os.path.join(nc_dir,model_name)
                
                if  os.path.isfile(main_hds):
                    if os.stat(main_hds).st_size>0 and os.path.getmtime(main_hds) > rr_date_s:                    
                        running_bool,model_list = cgu.update_txt(model_mng_file,model_name) # write model_name into management file
                        print('Model already run. Moving on to next')
                        print('--------------------------------------------\n')
                        
                        if os.path.isfile(new_nc_fname) and not os.path.isfile(os.path.join(new_nc_folder,ref_name)): # copy over nc file to nc_folder
                            # Copy nc file into general shared folder
                            if not os.path.isdir(new_nc_folder):
                                os.makedirs(new_nc_folder)
                            
                            store_nc_fname = os.path.join(new_nc_folder,os.path.basename(new_nc_fname))
                            copyfile(new_nc_fname,store_nc_fname)
                            copyfile(os.path.join(model_in_dir,ref_name),os.path.join(new_nc_folder,ref_name))
                    
                        
                        continue # skip this file
                elif os.path.isfile(out_hds): # and not rerun
                    if os.stat(out_hds).st_size>0 and os.path.getmtime(main_hds) > rr_date_s:
                        print('Model already run. Moving on to next')
                        print('--------------------------------------------\n')
                        continue # skip this file
                
                # Check management file
                if os.path.isfile(model_mng_file):
                    running_bool,model_list = cgu.update_txt(model_mng_file,model_name)
                    
                    if running_bool:
                        print('Model already run or running. Moving on to next')
                        print('--------------------------------------------\n')
                        continue
                    
                    other_model_name = otherfmt.format(ca_region,
                                                    active_domain,datum_type,
                                                    sealevel_elev,Kh,cell_spacing)
                    running_bool,model_list = cgu.update_txt(model_mng_file,other_model_name)
                    if running_bool:
                        print('Model already run or running. Moving on to next')
                        print('--------------------------------------------\n')
                        continue
                else:
                    # Make new file
                    running_bool,model_list = cgu.update_txt(model_mng_file,model_name)
                    
                
    
                for temp_dir in [model_in_dir,model_out_dir]:
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                        
                # See if nc file exists for other Kh models and copy nc file if so
                nc_found = False
                
                if not os.path.isfile(new_nc_fname):
                    for other_kh in Kh_vals:
                        for other_sl in sealevel_elevs:
                            for mnamefmt in [otherfmt,fmt]:
                                if not nc_found:
                                    other_model_name = mnamefmt.format(ca_region,
                                                            active_domain,datum_type,
                                                            other_sl,other_kh,cell_spacing)
                                    
                                    model_nc_dir = os.path.join(nc_dir,other_model_name)
                                    
                                    if os.path.isdir(model_nc_dir):
                                        model_nc_fname=os.path.join(model_nc_dir,'{}.nc'.format(other_model_name))
                                        if os.path.exists(model_nc_fname):
                                            # copy files
                                            copyfile(model_nc_fname,new_nc_fname)
                                            copyfile(os.path.join(model_nc_dir,ref_name),os.path.join(model_in_dir,ref_name))
                                            nc_found=True
                                
                # ----------- Develop model domain -----------
                elev_fname = os.path.join(elev_dir,'{0}_{1:02.0f}_dem_landfel.tif'.format(ca_region,active_domain))
                if ca_region in ['soca']:
                    active_domain  = active_domain-domain_df.iloc[0][id_col]
                    
                domain_dict = {'cell_spacing':cell_spacing,'input_dir':data_dir,'domain_shp':domain_df,
                               'active_domain':active_domain,'elev_fname':elev_fname,'rchg_fname':rchg_fname,
                               'k_fnames':None,'model_in_dir':model_in_dir,'sea_level':sealevel_elev,
                               'in_proj':in_proj,'use_ll':False}
                start_time = time.time()               
                m_domain = cpt.Model_domain_driver(**domain_dict)
                m_domain.run_make_domain(load_vk=False,save_nc_grids=save_nc_grids,
                                         load_nc_grids_bool=load_nc_grids_bool)
                print('Grid import took {0:4.1f} min'.format((time.time()-start_time)/60.))
    
                if not hasattr(m_domain,'density') or rerun_sl_dens:
                    # Make density and sea level grids
                    print('Loading sea level and seawater density data...')
                    g_dict = {'xi':(m_domain.cc_proj[0],m_domain.cc_proj[1]),'method':'nearest'}
                    
    #                if active_domain in [0]: 
    #                    g_dict['method'] = 'nearest'
    #                else:
    #                    g_dict['method'] = 'linear'
                        
                    buffer0 = 8e3 # m buffer around model
                    temp_extent = [m_domain.cc_proj[0].min(),m_domain.cc_proj[0].max(),
                                   m_domain.cc_proj[1].min(),m_domain.cc_proj[1].max()]
                    inpts = (sl_data_proj[:,:1]<=temp_extent[1]+buffer0) & (sl_data_proj[:,:1]>=temp_extent[0]-buffer0) \
                            & (sl_data_proj[:,1:2]<=temp_extent[3]+buffer0) & (sl_data_proj[:,1:2]>=temp_extent[2]-buffer0)
                    
                    m_domain.sea_level = cru.griddata(sl_data_proj[inpts.ravel(),:2],sl_data_proj[inpts.ravel(),2:],**g_dict).squeeze()+sealevel_elev
                    #m_domain.sea_level = np.median(sl_data_proj[inpts.ravel(),2:])+sealevel_elev
    #                g_dict['method'] = 'nearest'
                    inpts = (sal_data_proj[:,:1]<=temp_extent[1]+buffer0) & (sal_data_proj[:,:1]>=temp_extent[0]-buffer0) \
                            & (sal_data_proj[:,1:2]<=temp_extent[3]+buffer0) & (sal_data_proj[:,1:2]>=temp_extent[2]-buffer0)
                    m_domain.density = cru.griddata(sal_data_proj[inpts.ravel(),:2],sal_data_proj[inpts.ravel(),2:],**g_dict).squeeze()
                    #m_domain.density = np.median(sal_data_proj[inpts.ravel(),2:])
                
                # Assign cell types
                assign_dict = {'domain_obj':m_domain,'ws_shp':None}
                m_assignment = cpt.Assign_cell_types(**assign_dict)
                m_assignment.run_assignment(assign_wb=False,use_ws=False)
        #%%
                # ----------- Create flopy objects -----------
                # Model information
                m_info_dict = {'workspace':model_in_dir,'model_name':model_name}
                m_info = cmft.Model_info(**m_info_dict)
                
                # Develop discretization in space and time       
                m_dis_dict = {'cell_spacing':cell_spacing,'nlay':nlay,'nrow':m_domain.nrow,
                              'ncol':m_domain.ncol,'delv':layer_thick,
                              'zthick_elev_min':elev_thick_min, 'min_elev':min_surface_elev,
                              }
                              
                m_dis = cmft.Model_dis(**m_dis_dict)
                m_dis_time_dict = {'nper':nper,'perlen':perlen,'nstp':nstp,'steady':steady}
                m_dis.time_dis(**m_dis_time_dict)
            
                # Model specific changes to aid convergence
        #        if active_domain in [8]:
        #            # Set upper limit on recharge
        #            max_rchg = 0.2
        #            m_domain.recharge[m_domain.recharge>max_rchg] = max_rchg
            #%%
                # Make flopy package inputs
                m_dicts_in = {'dis_obj':m_dis,'cell_types':m_assignment.cell_types,
                              'elev_array':m_domain.elevation,'rchg_array':m_domain.recharge,
                              'hk':Kh,'k_decay':k_decay,'porosity':porosity,'run_swi':run_swi,
                              'solver':solver,'rho_s':m_domain.density,'sea_level':m_domain.sea_level}
                
                dis_kwargs = {'zthick':None,'min_zthick':min_zthick,
                              'smooth_zbot':True,'nlayer_continuous':1}
                drn_kwargs = {'dampener':1e4,'elev_damp':1e0}
                ghb_kwargs = {'skip_ghb':True}
                rchg_kwargs = {'rchg_units':[2,5],'reduce_lowk_rchg':True,
                               'lowk_rchg_ratio':1,'lowk_val':1e-3,'rchg_ratio':rchg_ratio} # recharge in m/yr
                bcf_kwargs = {'v_ani_ratio':v_ani,'iwdflg':1, 'smooth_k':True,
                              'hk_in':None,
                              'hk_botm':None,
                              'min_hk':min_hk,'nan_lower':False,'propkdown':True,
                              'hk_extent_clip':True}
                bas_kwargs = {'use_fweq_head':False,'set_marine_to_constant':True,
                              'min_thick_calc':None,'ibound_thick_threshold':False,
                              'ibound_minelev_threshold':0.,'check_inactive':False}
                gmg_kwargs = {'rclose':1e-2,'hclose':1e-2,
                              'mxiter':100,'iiter':100,
                              'isc':1,'ism':0} # loose convergance for iterations
                nwt_kwargs = {'options':'COMPLEX','iprnwt':1,'headtol':1e-2,'maxiterout':1000}#,'linmeth':2}
                if solver in ['nwt']:
                    upw_kwargs = bcf_kwargs.copy()
                    del upw_kwargs['iwdflg'] # Remove iwdflg
                    run_all_dicts_inputs = {'drn_kwargs':drn_kwargs,
                                            'ghb_kwargs':ghb_kwargs,
                                            'rchg_kwargs':rchg_kwargs,
                                            'upw_kwargs':upw_kwargs,
                                            'nwt_kwargs':nwt_kwargs,
                                            'dis_kwargs':dis_kwargs,
                                            'bas_kwargs':bas_kwargs}
                else:              
                    run_all_dicts_inputs = {'drn_kwargs':drn_kwargs,
                                            'ghb_kwargs':ghb_kwargs,
                                            'rchg_kwargs':rchg_kwargs,
                                            'bcf_kwargs':bcf_kwargs,
                                            'gmg_kwargs':gmg_kwargs,
                                            'dis_kwargs':dis_kwargs,
                                            'bas_kwargs':bas_kwargs}       
                                            
                m_dicts = cmft.Model_dict(**m_dicts_in)
                m_dicts.run_all(**run_all_dicts_inputs)
           
                # Force layers to no flow
            #    m_dicts.bas_dict['ibound'][1:,:,:] = 0
            #    m_dicts.bas_dict['ibound'][0,(m_dicts.ztop-m_dicts.zbot[0])<10.] = 0        
                                         
                # Make flopy packages
                maker_dict = {'dict_obj':m_dicts,'info_obj':m_info,
                              'external_path':None,'output_path':model_out_dir}
                m_maker = cmft.Model_maker(**maker_dict)
                m_maker.run()
                
                # setting inputs_exist=True will not overwrite inputs, only re-run model
                run_dict = {'model_obj':m_maker,'run_mf':True,'inputs_exist':False}
                m_run_obj = cmft.Model_run(**run_dict)
            #%%
                if force_write_inputs:
                    m_run_obj.inputs_exist=False
                else:
                    # see if files already written, ghb written last
                    if os.path.isfile(os.path.join(model_in_dir,'{}.ghb'.format(model_name))):
                        m_run_obj.inputs_exist=True
                        
                if hasattr(m_dicts,'conversion_mask'):
                    # Need to reduce extent of zones
                    cgu.recursive_applybydtype(m_domain,
                                               func=cgu.shrink_ndarray,
                                               func_args={'mask_in':m_maker.dict_obj.conversion_mask,
                                               'shape_out':m_maker.dict_obj.cell_types.shape})
                    
                    # Re-write grid data if the size of the grids changed
                    m_domain.load_griddata(save_nc_grids=save_nc_grids)
                    
                ref_dict = {'model_info_dict':{'model_ws':model_in_dir,'model_name':model_name,
                                               'xul':m_domain.cc_proj[0].data[0,0],# should be top left corner of top left cell, this is cell center
                                               'yul': m_domain.cc_proj[1].data[0,0],
                                               'length_units':m_run_obj.model_obj.mf_model.dis.lenuni,
                                               'time_units':m_run_obj.model_obj.mf_model.dis.itmuni,
                                               'start_date':0,'start_time':0,
                                               'model_type':solver,
                                               'rotation':m_domain.grid_transform[2],
                                               'proj_type':'proj4',
                                               'proj':m_domain.grid_transform[0]}}
                # Save model reference even when not running model
                m_run_obj.save_model_ref(**ref_dict)
                domain_extent_ll = cgu.get_extent(m_domain.cc_ll)
                domain_corners = [[domain_extent_ll[0],domain_extent_ll[3]],
                                  [domain_extent_ll[2],domain_extent_ll[3]],
                                  [domain_extent_ll[2],domain_extent_ll[1]],
                                  [domain_extent_ll[0],domain_extent_ll[1]]]
                             
                # Run Modflow model via flopy
                m_run_dict = {'mf_silent':True,'mf_exception':True,'save_ref_kwargs':ref_dict,
                              'xycorners':domain_corners,'XY_cc':m_domain.cc,'XY_proj':[np.asarray(m_domain.cc_proj[0].data),
                                                                                        np.asarray(m_domain.cc_proj[1].data)],
                              'inproj':m_domain.grid_transform[0]}  
                
                if run_mf:                
                    m_run_obj.run(**m_run_dict)
                    
                    # Copy nc file into general shared folder
                    if not os.path.isdir(new_nc_folder):
                        os.makedirs(new_nc_folder)
                    
                    store_nc_fname = os.path.join(new_nc_folder,os.path.basename(new_nc_fname))
                    copyfile(new_nc_fname,store_nc_fname)
                    copyfile(os.path.join(model_in_dir,ref_name),os.path.join(new_nc_folder,ref_name))
                    
                    # Save model budget data
                    if m_run_obj is not None:
                        if isinstance(m_run_obj.list_df,czbt.zbu.pd.Series):
                            m_run_obj.list_df = czbt.zbu.pd.DataFrame(m_run_obj.list_df).T
                        if model_name in model_budget_df.index.values:
                            model_budget_df.loc[model_name,:] = m_run_obj.list_df.loc[model_name,:]
                        else:
                            model_budget_df = czbt.zbu.pd.concat([model_budget_df,m_run_obj.list_df],axis=0,sort=True)    
                    else:
                        if isinstance(m_run_obj.list_df,czbt.zbu.pd.Series):
                            m_run_obj.list_df = czbt.zbu.pd.DataFrame(m_run_obj.list_df).T
                            
                        temp_df = czbt.zbu.pd.DataFrame(data=[],columns=model_budget_df.columns.values,index=model_name)
                        temp_df['success'] = False
                        model_budget_df = czbt.zbu.pd.concat([model_budget_df,temp_df],axis=0,sort=True)
                    
                    if not model_budget_df.loc[model_name,'convergence']:
                        print("Model {} failed to converge...moving on to additional model runs.".format(model_name))
            #            break
        #                continue
                
                    # Load head data
        #            fw_head,_ = cmfu.load_hds(model_name,model_out_dir,
        #                                                inactive_head=cgu.np.nan)    
                    #%% ---- Plot results ----
                    if plot_results:
                        
                        plot_dict = {'model_name':m_info.model_name,'workspace':model_out_dir,
                                     'domain_obj':m_domain,'dicts_obj':m_dicts,
                                 'head_vlims':[0,10],'wtdepth_vlims':[0,5],
                                 'run_swi':m_dicts_in['run_swi'],'cc_type':'model',
                                 'save_fig':True,'save_fig_fpath':os.path.join(figs_dir,'{}.png'.format(model_name))}
                                     
                        head_out,wt_layers,zeta_out = cmfu.cgw_plot(**plot_dict)
                    
                    
                    # ----------- ZoneBudget analysis -----------
                    if prep_zb:
                        if hasattr(m_dicts,'conversion_mask'):
                            # Need to reduce extent of zones
                            cgu.recursive_applybydtype(m_assignment,
                                                       func=cgu.shrink_ndarray,
                                                       func_args={'mask_in':m_maker.dict_obj.conversion_mask,
                                                       'shape_out':m_maker.dict_obj.cell_types.shape})  
                        
                        # Update m_assignment cell types
                        m_assignment.cell_types = m_maker.dict_obj.cell_types.copy()
                        zb_assign_dict = {'assign_cwb':True,'assign_ws':True}                               
                        zb_assign = czbt.Assign_zones(cell_type_obj=m_assignment)
                        zb_assign.run_assign_zones(**zb_assign_dict)
                        if (max_zb_layer is not None):
                            zb_assign.add_layers(max_layer=max_zb_layer)
                            zone_layer = np.arange(zb_assign.zones.shape[0])
                        else:
                            zone_layer= None
                        
                
                        zb_run_dict = {'output_ws':model_out_dir,'model_name':m_info.model_name,
                                       'dims':m_run_obj.model_obj.mf_model.nrow_ncol_nlay_nper,
                                       'zones':zb_assign.zones,
                                       'zone_layer':zone_layer}
                                       
                        zb_obj = czbt.Make_ZoneBudget(**zb_run_dict)
                        if run_zb:
                            zb_obj.run_ZoneBudget()    
                        
                            zb_proc_obj = czbt.Process_ZoneBudget(zb_output=zb_obj.zb_output,zone_obj=zb_assign)
                            zb_proc_obj.run_all(save_data=True,save_shp=True)
                       
                    loop_time = time.time()-model_start
                    model_budget_df.loc[model_name,'total_time_s'] = loop_time    
                    # ----------- Model run finishing statements -----------
                    print('Elapsed solution time: {0:4.1f} min'.format((time.time()-model_start)/60.))
                    print('{} finished at {}'.format(model_name,time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
                    print('--------------------------------------------\n')
                    if clear_all:
                        mp_res_obj = None
                        mp_obj = None
                        zb_obj = None
                        zb_proc_obj = None
                        m_run_obj = None
                        m_dicts = None
                        m_domain = None
                        m_assignment = None
                        m_maker = None
                        m_info = None
                    
                    # Not really a success if the model has large budget error
                    model_budget_df.loc[model_budget_df['PERCENT_DISCREPANCY'].abs()>1.,'success']=False
                    model_budget_df['ca_region'] = ca_region
                    model_budget_df[id_col] = cgu.np.array([ind.split('_') for ind in model_budget_df.index.values])[:,1]
                    model_budget_df.to_csv(os.path.join(model_outputs_dir,'{}_budget_summary.csv'.format(ca_region)),
                                           index=True,index_label='model_name')      
#%%
print('Total time for modeling: {0:4.1f} min'.format((time.time()-all_start)/60.))


#%%
copy_results=False
# https://support.microsoft.com/sq-al/help/240268/copy-xcopy-and-move-overwrite-functionality-changes-in-windows
xcopy_txt = 'XCOPY {} {} /Q/D/E/C/I/Y'
robomove_txt = 'robocopy {} {} /MOVE /E'

move_or_copy = 'move'

if copy_results:
    
    # os independent way...doesn't check to see if file exists
#    import shutil, errno
#    
#    def copyanything(src, dst):
#        '''
#        
#        Source: https://stackoverflow.com/a/1994840/8651697
#        '''
#        try:
#            shutil.copytree(src, dst)
#        except OSError as exc: # python >2.5
#            if exc.errno == errno.ENOTDIR:
#                shutil.copy(src, dst)
#            else: raise
#    
    
    if not os.path.isdir(out_main_model_dir):
        os.makedirs(out_main_model_dir)

    for ca_region in ca_regions: # loop through CA regions
        # ----------- Region directory information -----------
        
        region_dir = os.path.join(main_model_dir,ca_region)
        out_region_dir = os.path.join(out_main_model_dir,ca_region)

        if not os.path.isdir(out_region_dir):
            os.makedirs(out_region_dir)
        
        if move_or_copy in ['copy']:
            copy_dict = {'cmd_list':xcopy_txt.format(region_dir,out_region_dir)}
        else:
            copy_dict = {'cmd_list':robomove_txt.format(region_dir,out_region_dir)}
        copy_success,outtxt = cgu.run_cmd(**copy_dict)
#        copyanything(region_dir,out_region_dir)


