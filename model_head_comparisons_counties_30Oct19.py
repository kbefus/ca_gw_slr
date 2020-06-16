# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:40:23 2017

@author: kbefus


Feb 6 19 update - add KDTree to remove pts inside convex hull but with nan model outputs

"""

import sys,os
import numpy as np
import time,glob 
import pandas as pd
import geopandas as gpd

res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
sys.path.insert(1,code_dir)

from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_raster_utils as cru
from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_modflow import cgw_mf_utils as cmfu
from cgw_model.cgw_utils import nwis_utils as cwis

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rc('legend',**{'fontsize':9})
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from scipy.optimize import curve_fit
from scipy.spatial import cKDTree as KDTree

import rasterio
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile

#%%
def xy_from_affine(tform=None,nx=None,ny=None):
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
    return X,Y

def read_geotiff(in_fname,band=0):
    with rasterio.open(in_fname) as src:
        data = src.read()[band]
        data[data==src.nodata]=np.nan
        ny,nx = data.shape
        X,Y = xy_from_affine(src.transform,nx,ny)
    return X,Y,data

def load_tif_and_extractxy(in_fname=None, out_proj=None, xypts=None,
                           resampling = Resampling.bilinear,):
    '''.'''
    if isinstance(out_proj,int):
        out_crs = CRS.from_epsg(out_proj)
    
    vrt_options = {'resampling': resampling,
                'crs':out_crs,
                'nodata':-9999,
                'tolerance':0.01,'all_touched':True,'num_threads':4,
                'sample_grid':'YES','sample_steps':100,
                'source_extra':10}
    
    
    with rasterio.open(in_fname) as src:
        nrows,ncols = src.shape
        transform,width,height = calculate_default_transform(src.crs,out_crs,ncols,nrows,*src.bounds)
        vrt_options.update({'height': height,
                            'width': width,
                            'transform':transform})
        
        with WarpedVRT(src, **vrt_options) as vrt:
            if xypts is None:
                out_rast = vrt.read()[0]
                out_rast[out_rast == vrt.nodata] = np.nan
                X,Y = xy_from_affine(transform,width,height)
                return X,Y, out_rast
            else:
                vals = np.array(list(vrt.sample(xypts)))
                return vals
    

#def reproject_gdal(in_fname=None,band=0,dst_crs='EPSG:4326'):
#    '''
#    
#    EPSG: 4326 is WGS 84
#    '''
#
#    with rasterio.open(in_fname) as src:
#        transform, width, height = calculate_default_transform(
#            src.crs, dst_crs, src.width, src.height, *src.bounds)
#        kwargs = src.meta.copy()
#        kwargs.update({
#            'crs': dst_crs,
#            'transform': transform,
#            'width': width,
#            'height': height
#        })
#        with MemoryFile() as memfile:
#            with memfile.open(**kwargs) as dst:
##                dst.write(np.zeros((1,height,width)))
#                reproject(
#                    source=rasterio.band(src, band),
#                    destination=rasterio.band(dst, band),
#                    src_transform=src.transform,
#                    src_crs=src.crs,
#                    dst_transform=transform,
#                    dst_crs=dst_crs,
#                    resampling=Resampling.nearest)
#                # Return result
#                data = dst.read()[band]
#                data[data==dst.nodata]=np.nan
#                ny,nx = data.shape
#                X,Y = xy_from_affine(dst.transform,nx,ny)
#    
#    return X,Y,data
                
#%%
save_fig = True
save_csv=True
rerun_analysis = True

research_dir_orig = os.path.join(res_dir,'ca_slr')
data_dir_orig = os.path.join(research_dir_orig,'data')
research_dir = r'/mnt/762D83B545968C9F'
output_dir = os.path.join(research_dir,'data','outputs_fill_gdal_29Oct19')

model_types = ['model_lmsl_noghb','model_mhhw_noghb']

# Eq 1, Krause, P., and D. P. Boyle (2005), Advances in Geosciences Comparison 
# of different efficiency criteria for hydrological model assessment, 
# Adv. Geosci., 5(89), 89–97, doi:10.5194/adgeo-5-89-2005.
R2func = lambda obs,mod: (np.sum((obs-np.mean(obs))*(mod-np.mean(mod)))\
                          /(np.sqrt(np.sum((obs-np.mean(obs))**2.))*\
                            np.sqrt(np.sum((mod-np.mean(mod))**2.))))**2
def lin_func(x,a):
    return a*x
                   
ref_name = 'usgs.model.reference'



ca_gw_dir = os.path.join(research_dir_orig,'data','well_data')
ca_gw_fname = os.path.join(ca_gw_dir,'CA_ESI_DWR_wtelev_17Nov19.csv')
ca_df = cfu.pd.read_csv(ca_gw_fname)
e88_col = 'elev_NAVD88_ft'
max_tot_depth = 300 # ft
ca_df = ca_df[ca_df['TOT_DEPTH']<=max_tot_depth]
ca_head_col = 'wt_elev_ft_nanmean'

# Remove unrealistic values
# Minimum head = elevation - total depth
too_low_bool = ca_df[ca_head_col]<(ca_df[e88_col]-ca_df['TOT_DEPTH'])
# Confined when gw level > surface elevation
confined_bool = ca_df[ca_head_col]>ca_df[e88_col]
ca_df = ca_df[~too_low_bool & ~confined_bool]
ca_df = ca_df.drop_duplicates(subset=['UNIQUE_ID'])

ft2m = .3048
out_proj = 4269 # NAD83

county_fname = os.path.join(research_dir_orig,'data','gis','CA_Counties_TIGER2016.shp')
shape_feature = ShapelyFeature(Reader(county_fname).geometries(),
                                ccrs.PlateCarree(), edgecolor='grey')

county_shp_df = gpd.read_file(county_fname)
shp_crs = county_shp_df.crs['init'].split(':')[1]
shp_crs = CRS.from_epsg(shp_crs).wkt
out_list = []

#%%
model_type = 'fw' # 'swi' or 'fw'
sealevel_elev = 0. # only run comparison for present-day sea level
cell_spacing = 10. # meters
max_dist = np.sqrt(2*(cell_spacing**2.))/2.
Kh_vals = [0.1,1.,10.]
plt_errors=True
skip_existing = False   
model_name_fmt = '_{0}_res{1}m_sl{2:3.2f}m_Kh{3:3.2f}'
other_model_name_fmt = '_{0}_res{1}m_sl{2:3.2f}m_Kh{3:3.1f}'   
active_date = 'means_4Nov19'

    
#%%
a1=[]
store_model_data = []

for model_type in model_types:
    datum_type = model_type.split('_')[1].upper()
    scenario_type = '_'.join(model_type.split('_')[1:])
    
    head_dir = os.path.join(output_dir,model_type,'head')
    save_fig_dir = os.path.join(head_dir,'Error_plots',active_date)
    if not os.path.isdir(save_fig_dir):
        os.makedirs(save_fig_dir)
    county_dirs = glob.glob(os.path.join(head_dir,'*'))
    county_dirs = [idir for idir in county_dirs if 'Error' not in idir and '.zip' not in idir]
    
    fig_mng_file = os.path.join(save_fig_dir,'fig_management_{0}_{1}.txt'.format(scenario_type,active_date))
    summary_fname = os.path.join(save_fig_dir,'All_crossplot_data_{0}_{1}.csv'.format(scenario_type,active_date))
    if not rerun_analysis:
        if os.path.isfile(summary_fname):
            summary_df = pd.read_csv(summary_fname)
    
    
    
    for Kh in Kh_vals:
        region_data = []
        kh_dir = 'Kh{0:3.2f}mday'.format(Kh)
        kh_dir=kh_dir.replace('.','p')
        out_all_fname = os.path.join(save_fig_dir,'All_error_data_{}_{}_Kh{}mday.csv'.format(scenario_type,active_date,Kh))
        if not rerun_analysis:
            if os.path.isfile(out_all_fname):
                output_df = pd.read_csv(out_all_fname)
                
        for county_dir in county_dirs:
            ca_county = os.path.basename(county_dir)
            ca_county_name = ' '.join(ca_county.split('_'))
            if Kh==Kh_vals[0]:
                fmt = model_name_fmt
            else:
                fmt = other_model_name_fmt
                
            model_start = time.time()
            head_fname = glob.glob(os.path.join(county_dir,kh_dir,'*slr0p00m.tif'))[0]
            model_name = os.path.basename(os.path.splitext(head_fname)[0])
    
            # Check management file
            if os.path.isfile(fig_mng_file) and not rerun_analysis:
                running_bool,model_list = cgu.update_txt(fig_mng_file,model_name)
                
                if running_bool:
                    print('Model already run or running. Moving on to next')
                    print('--------------------------------------------\n')
                    continue
                
            else:
                # Make new file
                running_bool,model_list = cgu.update_txt(fig_mng_file,model_name)
                                
            print('------------Current file: {} ------------'.format(os.path.basename(head_fname)))
            print('Start time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
            # use well locations dl'ed for main model run
            gw_data_dir = os.path.join(county_dir,'observations{}'.format('_v24Jun19'))
            if not os.path.isdir(gw_data_dir):
                os.makedirs(gw_data_dir)
            
            out_fig = os.path.join(save_fig_dir,'{}_means_wlerror.pdf'.format(model_name))
            out_fig2 = os.path.join(save_fig_dir,'{}_means_wlerror.png'.format(model_name))
            if os.path.isfile(out_fig) and os.path.isfile(out_fig2) and skip_existing:
    #            pass
                continue
            
    #        X,Y,wt_head = read_geotiff(head_fname)
    #        proj_in = cru.load_grid_prj(head_fname)
    #        # project to lat, long, NAD83
    #        nrow,ncol,grid_tform = cru.get_rast_info(head_fname)
    #        
    #        transform_dict = {'XY':[X,Y],'xyul':[0,0],'xyul_m':[0]*2,
    #                          'rotation':0,'proj_in':proj_in,'proj_out':'NAD83'}
    #        x_lon,y_lat = cgu.modelcoord_transform(**transform_dict)
            
            # Remove sites outside county
            temp_polys = county_shp_df[county_shp_df['NAME']==ca_county_name].geometry.values
            temp_polys = np.array([ipoly.buffer(0) for ipoly in temp_polys])
            if len(temp_polys.shape)>1:
                temp_polys = temp_polys.squeeze()
            ivalid = [i for i,tempp in enumerate(temp_polys) if tempp.is_valid]
            polys = cfu.unary_union(temp_polys[ivalid])            
            
            if county_shp_df.crs['init'] not in ['epsg:4326','epsg:4269']:
            
                proj_dict= {'proj_in':shp_crs,'proj_out':'NAD83',
                            'xyul_m':[0,0],'xyul':[0,0],
                            'rotation':0}
                try:
                    out_poly = cfu.proj_polys([polys],proj_kwargs=proj_dict)
                except:
                    out_poly = cfu.proj_polys([polys.convex_hull],proj_kwargs=proj_dict)
            else:
                out_poly = [polys]
            
            bbox=out_poly[0].bounds
            
            params_to_dl = [wkey for wkey in cwis.param_dict.keys() if 'wl' in wkey]
            agg_funcs = ['count','mean','median','min','max']
            nwis_dict = {'bbox':bbox,
                         'params':params_to_dl,
                         'agg_funcs':agg_funcs,
                         'save_dict':{'work_dir':gw_data_dir}}
            if not os.path.isfile(os.path.join(gw_data_dir,'site_data.csv')):
    
                print("Downloading site and groundwater data for {}".format(model_name))            
                site_df,data_df = cwis.load_nwis_wq(**nwis_dict)
            else:
                print("Loading site and groundwater data for {}".format(model_name))
                site_df = cwis.pd.read_csv(os.path.join(gw_data_dir,'site_data.csv'))
                data_df = cwis.pd.read_csv(os.path.join(gw_data_dir,'wl_data.csv'))
                if site_df.shape[0]>0:
                    if nwis_dict['agg_funcs'] is not None:
                        data_df = cwis.organize_wq_data(data_df,
                                                        agg_funcs=nwis_dict['agg_funcs'])
                else:
                    site_df = None
                    data_df = None
            
    
            
            buff=(cell_spacing*5)/1.1e5
    #        tree = KDTree(np.c_[x_lon[~np.isnan(wt_head)],y_lat[~np.isnan(wt_head)]])
            if site_df is not None:
    #            nan_inds = np.isnan(wt_head) & (x_lon<(site_df['long'].min()-buff)) &\
    #                    (x_lon>(site_df['long'].max()+buff)) & (y_lat<(site_df['lat'].min()-buff)) &\
    #                    (y_lat>(site_df['lat'].max()+buff))
                # Merge site and data dfs        
                all_df = cwis.pd.merge(site_df,data_df,right_index=True,left_on='site_no')
                all_df = all_df[all_df['wdepth']<=max_tot_depth].copy() # cull deep wells
                in_pt_dict = {'XYpts':[all_df['long'].values,all_df['lat'].values],
                              'in_polygon':out_poly[0]}
                inside_inds = cfu.pt_in_shp(**in_pt_dict)
                if len(np.array(inside_inds).ravel())>0:
                    all_df = all_df.iloc[np.hstack(inside_inds).flatten()]
                    print("Interpolating for USGS current data")
    #                out_data1 = cru.griddata(np.c_[x_lon[~nan_inds],y_lat[~nan_inds]],
    #                                         wt_head[~nan_inds],
    #                                         (all_df['long'].values,all_df['lat'].values))
    #                # Remove points inside hull but nan values
    #                
    #                dist,_ = tree.query(np.c_[all_df['long'].values,all_df['lat'].values])
    #                out_data1[dist>max_dist] = np.nan
                    longlat = np.column_stack((all_df['long'].values,all_df['lat'].values))
                    out_data1 = load_tif_and_extractxy(head_fname,out_proj,xypts=longlat)
                    out_data1[out_data1<0] = np.nan # set negative values to nan
                    
                    n_active_wells = len((~np.isnan(out_data1)).nonzero()[0])
                    if n_active_wells>0:
                        test_confined = lambda x: 'unconfined' not in x.lower() if isinstance(x,str) else False        
                        confined_aqs = all_df['aq_type'].apply(test_confined)
                        
                        # Calculate w.t. elevation using ground surface elevation and water depth
                        if 'wl_dblsdm_mean' in all_df.columns:
                            all_df['wtmdelevm']=all_df['surf_elev'].values*ft2m-all_df['wl_dblsdm_mean'].values
                            all_df['wtmindelevm']=all_df['surf_elev'].values*ft2m-all_df['wl_dblsdm_min'].values
                            all_df['wtmaxdelevm']=all_df['surf_elev'].values*ft2m-all_df['wl_dblsdm_max'].values
    
                        # Convert ft columns to m, only if no values found above
                        if len(all_df['wdepth_units'].shape)==2:
                            ft_rows = np.all((all_df['wdepth_units']=='ft'),axis=1) & all_df['wtmdelevm'].isnull()
                        else:
                            ft_rows = (all_df['wdepth_units']=='ft') & all_df['wtmdelevm'].isnull()
                        if len(ft_rows.shape)>1:
                            ft_rows=ft_rows.T.drop_duplicates().T
                            all_df = all_df.loc[:,~all_df.columns.duplicated()]
                        
                        if not isinstance(ft_rows,cwis.pd.Series):
                            ft_rows = ft_rows['wdepth_units']
                        
                        if 'wl_dlsft_mean' in all_df.columns:
                            all_df.loc[ft_rows,'wtmdelevm'] = (all_df.loc[ft_rows,'surf_elev'].values-all_df.loc[ft_rows,'wl_dlsft_mean'].values)*ft2m
                            all_df.loc[ft_rows,'wtmindelevm']= (all_df.loc[ft_rows,'surf_elev'].values-all_df.loc[ft_rows,'wl_dlsft_min'].values)*ft2m
                            all_df.loc[ft_rows,'wtmaxdelevm']= (all_df.loc[ft_rows,'surf_elev'].values-all_df.loc[ft_rows,'wl_dlsft_max'].values)*ft2m
                        
                        # Only use unconfined aquifers
                        final_df = all_df.loc[~confined_aqs,:].copy()
                        gw_levels = out_data1[np.where(~confined_aqs)]
                        
                        # remove nan heads
                        nan_heads = cgu.np.isnan(gw_levels).squeeze()
                        if len(nan_heads.shape)==0:
                            nan_heads = np.array([nan_heads])
                        gw_levels = gw_levels[np.where(~nan_heads)]
                        final_df = final_df.loc[~nan_heads,:].copy()
                        gw_levels = gw_levels[np.where(~final_df['wtmdelevm'].isnull())]
                        final_df.dropna(subset=['wtmdelevm'],inplace=True,axis=0)
                        final_df['model_headm'] = gw_levels
                        final_df = final_df.dropna(subset=['model_headm'])
                    else:
                        final_df = np.array([])
                else:
                    final_df = np.array([])
                # Calculate fit stats
                if final_df.shape[0]>0:
                    pres_r2 = R2func(final_df['model_headm'],final_df['wtmdelevm'].values)
                    popt, pcov = curve_fit(lin_func,final_df['wtmdelevm'].values.astype(float),final_df['model_headm'],
                                           bounds=(0.,10.))
                    perr = np.sqrt(np.diag(pcov))
                else:
                    gw_levels = []
                    popt = [np.nan]
                    pres_r2 = np.nan
                    final_df=np.array([])
            else:
                gw_levels = []
                popt = [np.nan]
                pres_r2 = np.nan
                final_df=np.array([])
    
            # Load historic gw level data
            # Historic data
            hist_dict = {'url_opts':{'bbox':bbox,'site_no':None},
                 'agg_funcs':agg_funcs,
                 'save_dict':{'work_dir':gw_data_dir}}
            if not os.path.isfile(os.path.join(gw_data_dir,'site_hist_data.csv')):
                print("Downloading historic site and groundwater data for {}".format(model_name))   
                mdf,urls = cwis.load_historicgw_df(**hist_dict)
            else:
                print("Loading historic site and groundwater data for {}".format(model_name))
                site_hist_df = cwis.pd.read_csv(os.path.join(gw_data_dir,'site_hist_data.csv'))
                data_hist_df = cwis.pd.read_csv(os.path.join(gw_data_dir,'wl_hist_data.csv'))
                if site_hist_df.shape[0]>0:
                    org_df = cwis.organize_wq_data(data_hist_df,ind_cols=['site_no'],
                                                   agg_funcs=agg_funcs,
                                                   agg_col='lev_va')    
                    mdf = cwis.pd.merge(site_hist_df,org_df,left_on='site_no',right_index=True)
                    mdf = mdf[mdf['well_depth_va']<=max_tot_depth].copy() # cull deep wells
                    # Fix multi-column names
                    new_cols = cwis.pd.Index(['_'.join(e) if isinstance(e,tuple) else e for e in mdf.columns.tolist()])
                    mdf.columns = new_cols
                else: 
                    mdf = None
            
            if mdf is not None:
                
                
                # Remove sites outside active model domain
                in_pt_dict2 = {'XYpts':[mdf['dec_long_va'].values,mdf['dec_lat_va'].values],
                              'in_polygon':out_poly[0]}
                inside_inds2 = cfu.pt_in_shp(**in_pt_dict2)
                if len(np.array(inside_inds2).ravel())>0:
    #                nan_inds = np.isnan(wt_head) & (x_lon<(mdf['dec_long_va'].values[inside_inds2].min()-buff)) &\
    #                    (x_lon>(mdf['dec_long_va'].values[inside_inds2].max()+buff)) &\
    #                    (y_lat<(mdf['dec_lat_va'].values[inside_inds2].min()-buff)) &\
    #                    (y_lat>(mdf['dec_lat_va'].values[inside_inds2].max()+buff))
                    print("Interpolating for USGS historic data")
    #                out_data2 = cru.griddata(np.c_[x_lon[~nan_inds],y_lat[~nan_inds]],
    #                                         wt_head[~nan_inds],
    #                                         (mdf['dec_long_va'].values[inside_inds2],mdf['dec_lat_va'].values[inside_inds2]))
    #                
    #                # Remove points inside hull but nan values
    #                dist,_ = tree.query(np.c_[mdf['dec_long_va'].values[inside_inds2],mdf['dec_lat_va'].values[inside_inds2]])
    #                out_data2[dist>max_dist] = np.nan
                    
                    longlat = np.column_stack((mdf['dec_long_va'].values[inside_inds2],
                                               mdf['dec_lat_va'].values[inside_inds2]))
                    out_data2 = load_tif_and_extractxy(head_fname,out_proj,xypts=longlat)
                    out_data2[out_data2<0] = np.nan # set negative values to nan
                    
                    n_active_wells_hist = len((~np.isnan(out_data2)).nonzero()[0]) 
                    
                    test_hist_confined = mdf.iloc[np.array(inside_inds2).flatten()]['aqfr_type_cd'].isin(['C','M'])
                    
                    # All historic data in ft
                    mdf['wtmdelevm']=(mdf['alt_va'].values-mdf['lev_va_mean'].values)*ft2m
                    mdf['wtmindelevm']=(mdf['alt_va'].values-mdf['lev_va_min'].values)*ft2m
                    mdf['wtmaxdelevm']=(mdf['alt_va'].values-mdf['lev_va_max'].values)*ft2m
                    
                    hist_final_df = mdf.iloc[np.hstack(inside_inds2).flatten()].loc[~test_hist_confined].copy()
                    hist_gw_levels = out_data2[np.where(~test_hist_confined)]
                                               
                    
                    # remove nan heads
                    nan_hist_heads = cgu.np.isnan(hist_gw_levels)
                    hist_gw_levels = hist_gw_levels[np.where(~nan_hist_heads)]
                    hist_final_df = hist_final_df[~nan_hist_heads].copy()
                    hist_gw_levels = hist_gw_levels[np.where(~hist_final_df['wtmdelevm'].isnull())]
                    hist_final_df.dropna(subset=['wtmdelevm'],inplace=True,axis=0)
                    hist_final_df['model_headm'] = hist_gw_levels
                    hist_final_df = hist_final_df.dropna(subset=['model_headm'])
                else:
                    hist_final_df = np.array([])
                    
                if hist_final_df.shape[0]>0:
                    # Calculate fit stats
                    hist_r2 = R2func(hist_final_df['model_headm'],hist_final_df['wtmdelevm'])
            #        [m_hist,b_hist] = np.polyfit(hist_gw_levels,hist_final_df['wtmdelevm'].values,1)
                    popt_hist, pcov_hist = curve_fit(lin_func,hist_final_df['wtmdelevm'].values.astype(float), hist_final_df['model_headm'],
                                                     bounds=(0.,10.))
                    perr_hist = np.sqrt(np.diag(pcov_hist))
                else:
                    mdf=None
                    hist_gw_levels = []
                    popt_hist = [np.nan]
                    hist_r2 = np.nan
                    hist_final_df = np.array([])
            else:
                hist_gw_levels = []
                popt_hist = [np.nan]
                hist_r2 = np.nan
                hist_final_df = np.array([])
            
            # Extract CA data    
            ca_bool = (ca_df['LONGITUDE']<=bbox[2]) & (ca_df['LONGITUDE']>=bbox[0]) &\
                      (ca_df['LATITUDE']<=bbox[3]) & (ca_df['LATITUDE']>=bbox[1])
            active_ca_df = ca_df.loc[ca_bool,:].copy().dropna(subset=[ca_head_col])
            
            in_pt_dict3 = {'XYpts':[active_ca_df['LONGITUDE'].values,active_ca_df['LATITUDE'].values],
                              'in_polygon':out_poly[0]}
            inside_inds3 = cfu.pt_in_shp(**in_pt_dict3)
            if len(np.array(inside_inds3).ravel())>0:
                
                print("Interpolating for CA data")
                longlat = np.column_stack((active_ca_df['LONGITUDE'].values[inside_inds3],active_ca_df['LATITUDE'].values[inside_inds3]))
                out_data3 = load_tif_and_extractxy(head_fname,out_proj,xypts=longlat)
                out_data3[out_data3<0] = np.nan # set negative values to nan
    #            nan_inds = np.isnan(wt_head) & (x_lon<(active_ca_df['LONGITUDE'].values[inside_inds3].min()-buff)) &\
    #                    (x_lon>(active_ca_df['LONGITUDE'].values[inside_inds3].max()+buff)) &\
    #                    (y_lat<(active_ca_df['LATITUDE'].values[inside_inds3].min()-buff)) &\
    #                    (y_lat>(active_ca_df['LATITUDE'].values[inside_inds3].max()+buff))
    #            out_data3 = cru.griddata(np.c_[x_lon[~nan_inds],y_lat[~nan_inds]],wt_head[~nan_inds],
    #                                     (active_ca_df['LONGITUDE'].values[inside_inds3],active_ca_df['LATITUDE'].values[inside_inds3]))
                # Remove points inside hull but nan values
    #            dist,_ = tree.query(np.c_[active_ca_df['LONGITUDE'].values[inside_inds3],active_ca_df['LATITUDE'].values[inside_inds3]])
    #            out_data3[dist>max_dist] = np.nan
                active_ca_df = active_ca_df.iloc[np.hstack(inside_inds3).flatten()].copy() # remove outside wells
    
                n_active_wells_ca = len((~np.isnan(out_data3)).nonzero()[0])
                ca_gw_levels = out_data3 #  keep here for later culling of wt elevations by depth or other parameters
                active_ca_df['model_headm'] = out_data3
                active_ca_df = active_ca_df.dropna(subset=['model_headm'])
            else:
                n_active_wells_ca=0
                active_ca_df['model_headm'] = np.nan
                active_ca_df = active_ca_df.dropna(subset=['model_headm'])
                
            if n_active_wells_ca>0:
                ca_r2 = R2func(active_ca_df['model_headm'],active_ca_df[ca_head_col]*ft2m)
                popt_ca, pcov_ca = curve_fit(lin_func,active_ca_df[ca_head_col]*ft2m, active_ca_df['model_headm'],
                                                     bounds=(0.,10.))
                perr_ca = np.sqrt(np.diag(pcov_ca))
            else:
                ca_gw_levels = []
                popt_ca = [np.nan]
                ca_r2 = np.nan
            
                  
            # ------------- Prepare data for plotting ------------
            
            len_usgs_current = final_df.shape[0]
            len_usgs_hist = hist_final_df.shape[0]
            len_ca = active_ca_df.shape[0]
            
            row_labels = ['R$^2$','Slope_fit','N_wells']
            col_labels = ['Current','Historic','CA']
            cell_text = [['{0:10.2f}'.format(itemp) for itemp in [pres_r2,hist_r2,ca_r2]],
                         ['{0:10.2f}'.format(itemp) for itemp in [popt[0],popt_hist[0],popt_ca[0]]],
                         [len_usgs_current,len_usgs_hist,len_ca]]
                          
            # save model run information
            out_list.append([ca_county,model_name,
                             len_usgs_current,pres_r2,popt[0],
                             len_usgs_hist,hist_r2,popt_hist[0],
                             len_ca,ca_r2,popt_ca[0]])
            # Plot data
            marker_size = 3.
    #        fig, ax = plt.subplots(1,2,figsize=(8,8))
            fig = plt.figure(figsize=(11,8.5))
            w = [1, 2]
    #        h = [1,1]
            gs = gridspec.GridSpec(1, 2,width_ratios=w)
    #        ax1 = plt.subplot(gs[1])
            inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                                                     subplot_spec=gs[1], wspace=0.1,
                                                     hspace=0.2,height_ratios=[6,1,1])
            ax1a = plt.subplot(inner[0])
            fig.suptitle("{} County".format(' '.join(ca_county.split('_'))), fontsize=14)        
            
            if len_usgs_current > 0 and len_usgs_hist > 0 and len_ca>0:
                max_xy=np.max([final_df['wtmdelevm'].max(),hist_final_df['wtmdelevm'].max(),active_ca_df[ca_head_col].max()*ft2m])
            elif len_usgs_current > 0:
                max_xy=np.max([final_df['wtmdelevm'].max(),final_df['wtmdelevm'].max()])
            elif len_usgs_hist > 0:
                max_xy=np.max([hist_final_df['wtmdelevm'].max(),hist_final_df['wtmdelevm'].max()])
            elif len_ca > 0:
                max_xy=np.max([active_ca_df[ca_head_col].max()*ft2m,active_ca_df[ca_head_col].max()*ft2m])
            else:
                max_xy = 50.
            
            if np.isnan(max_xy) or max_xy<=1:
                max_xy = 50.
            
            max_xy = 20 # force max_xy values (comment if variability wanted)
            out_data = []
            
            # Cross plot
            if plt_errors:
                if len_usgs_current > 0:
                    final_df.loc[final_df['wtmindelevm']<0.,'wtmindelevm'] = np.nan
                    final_df.loc[final_df['wtmaxdelevm']<0.,'wtmaxdelevm'] = np.nan
                    final_df.loc[final_df['wtmdelevm']<0.,'wtmdelevm'] = np.nan
                    error_array = np.vstack([final_df['wtmdelevm'].values-final_df[['wtmindelevm','wtmaxdelevm']].min(axis=1).values,
                                             final_df[['wtmindelevm','wtmaxdelevm']].max(axis=1).values-final_df['wtmdelevm'].values])
                    error_array[error_array==0] = np.nan
                    ax1a.errorbar(final_df['wtmdelevm'],final_df['model_headm'],xerr=error_array,
                                    markersize=marker_size,fmt='b.',label='Current min to max',alpha=0.2)
                    # Organize outputs for saving to csv
                    out_data.extend(list(zip([Kh]*len_usgs_current,[ca_county]*len_usgs_current,
                                             [model_name]*len_usgs_current,
                                             final_df['long'].values,final_df['lat'].values,final_df['surf_elev']*ft2m,
                                             ['USGS_Current']*len_usgs_current,final_df['site_no'].values,final_df['wtmdelevm'].values,
                                             final_df['wtmdelevm'].values-error_array[0],
                                             final_df['wtmdelevm'].values+error_array[1],final_df['model_headm'].values)))
                    
                if len_usgs_hist > 0:
                    hist_final_df.loc[hist_final_df['wtmindelevm']<0.,'wtmindelevm'] = np.nan
                    hist_final_df.loc[hist_final_df['wtmaxdelevm']<0.,'wtmaxdelevm'] = np.nan
                    hist_final_df.loc[hist_final_df['wtmdelevm']<0.,'wtmdelevm'] = np.nan
                    hist_error_array = np.vstack([hist_final_df['wtmdelevm'].values-hist_final_df[['wtmindelevm','wtmaxdelevm']].min(axis=1).values,
                                             hist_final_df[['wtmindelevm','wtmaxdelevm']].max(axis=1).values-hist_final_df['wtmdelevm'].values])
                    hist_error_array[hist_error_array==0] = np.nan
                    hist_final_df.loc[hist_final_df['wtmindelevm']<0.,'wtmindelevm'] = np.nan
                    ax1a.errorbar(hist_final_df['wtmdelevm'],hist_final_df['model_headm'],
                                    xerr=hist_error_array,
                                    markersize=marker_size,fmt='g.',label='Historic min to max',alpha=0.2)
                    # Organize outputs for saving to csv
                    out_data.extend(list(zip([Kh]*len_usgs_hist,[ca_county]*len_usgs_hist,[model_name]*len_usgs_hist,
                                             hist_final_df['dec_long_va'],hist_final_df['dec_lat_va'],hist_final_df['alt_va']*ft2m,
                                                     ['USGS_Historic']*len_usgs_hist,hist_final_df['site_no'].values,hist_final_df['wtmdelevm'].values,
                                                     hist_final_df['wtmdelevm']-hist_error_array[0],
                                                     hist_final_df['wtmdelevm']+hist_error_array[1],hist_final_df['model_headm'])))
                    
                if n_active_wells_ca>0:
                    for icol in ['wt_elev_ft_nanmin','wtmaxfprod_elev_ft_nanmin','wtfprod_elev_ft_nanmin',
                                 'wtmaxfprod_elev_ft_nanmax','wt_elev_ft_nanmax','wtfprod_elev_ft_nanmax',
                                 'model_headm']:
                        active_ca_df.loc[active_ca_df[icol]<0.,icol] = np.nan
    
                    ca_error_array = np.vstack([active_ca_df[ca_head_col].values-np.nanmin(active_ca_df[['wt_elev_ft_nanmin','wtmaxfprod_elev_ft_nanmin']].values,axis=1),
                                             np.nanmax(active_ca_df[['wt_elev_ft_nanmax','wtmaxfprod_elev_ft_nanmax']].values,axis=1)-active_ca_df[ca_head_col].values])*ft2m
                    ca_error_array[ca_error_array==0] = np.nan
                    ax1a.errorbar(active_ca_df[ca_head_col]*ft2m,active_ca_df['model_headm'],
                                    xerr=ca_error_array,
                                    markersize=marker_size,fmt='r.',label='CA min to max',alpha=0.2)
    
                    # Organize outputs for saving to csv
                    out_data.extend(list(zip([Kh]*len_ca,[ca_county]*len_ca,[model_name]*len_ca,
                                             active_ca_df['LONGITUDE'],active_ca_df['LATITUDE'],active_ca_df['elev_NAVD88_ft']*ft2m,
                                             ['CA_ESI_DWR']*len_ca,active_ca_df['UNIQUE_ID'].values,active_ca_df[ca_head_col].values*ft2m,
                                             (active_ca_df[ca_head_col].values-ca_error_array[0])*ft2m,
                                             (active_ca_df[ca_head_col].values+ca_error_array[1])*ft2m,
                                             active_ca_df['model_headm'].values)))
            if save_fig:        
                xy_array = np.array([0,max_xy])
        
                # Current water level plots
                if len_usgs_current > 0:
                    
                    ax1a.plot(final_df['wtmdelevm'],final_df['model_headm'],'bx',label='Current groundwater levels',
                                markersize=marker_size)
                    ax1a.plot(xy_array,lin_func(xy_array,*popt),'b-',label='Current fit')
                    
                # Historic water level plots
                if len_usgs_hist > 0:
                    ax1a.plot(hist_final_df['wtmdelevm'],hist_final_df['model_headm'],'g.',label='Historic groundwater levels',
                                markersize=marker_size)
                    ax1a.plot(xy_array,lin_func(xy_array,*popt_hist),'g-',label='Historic fit')
                
                if n_active_wells_ca>0:
                    ax1a.plot(active_ca_df[ca_head_col]*ft2m,active_ca_df['model_headm'],'r.',label='CA gw levels')
                    ax1a.plot(xy_array,lin_func(xy_array,*popt_ca),'r-',label='CA fit')
                
        
                ax1a.set_xlim(xy_array)
                ax1a.set_ylim(xy_array)
                ax1a.plot(xy_array,xy_array,'k--',label='1:1')
                ax1a.set_xlabel('Measured head [m]')
                ax1a.set_ylabel('Modeled head [m]')
                ax1a.set_aspect('equal')
                
                # Plot legend
                handles, labels = ax1a.get_legend_handles_labels()
                ax1b = plt.subplot(inner[1])
                legend=ax1b.legend(handles,labels,numpoints=1,scatterpoints=1,ncol=2,
                                   loc='upper center')
                ax1b.set_axis_off()
                
                # Plot table
                ax1c = plt.subplot(inner[2])
                ax1c.table(cellText=cell_text,rowLabels=row_labels,colLabels=col_labels)
                ax1c.set_axis_off()
        #            bbox=[0.1, -.65, .8, .4])
                
                # Plot location data second
                ax0 = plt.subplot(gs[0],projection=ccrs.PlateCarree())
                ax0.coastlines(resolution="10m")
                
                # Plot county boundaries
                ax0.add_feature(shape_feature)
                
                plot_opts = {'ax':ax0,'facecolor':'none'}
                cfu.plot_shp(out_poly,**plot_opts)
                
                scat_opts = {'facecolor':'none',
                             'vmax':5.,'vmin':-5.,'cmap':'coolwarm','s':15}
                if len_usgs_current > 0:
                    c1 = ax0.scatter(final_df['long'],final_df['lat'],c=final_df['wtmdelevm']-final_df['model_headm'],**scat_opts)
                    c1.set_facecolor('none')
                if len_usgs_hist > 0:
                    c2 = ax0.scatter(hist_final_df['dec_long_va'],hist_final_df['dec_lat_va'],c=hist_final_df['wtmdelevm']-hist_final_df['model_headm'],**scat_opts)
                    c2.set_facecolor('none')
                
                if n_active_wells_ca > 0:
                    c3 = ax0.scatter(active_ca_df['LONGITUDE'],active_ca_df['LATITUDE'],c=active_ca_df[ca_head_col]*ft2m-active_ca_df['model_headm'],**scat_opts)
                    c3.set_facecolor('none')
                    
                ax0.set_aspect('equal')
                ax0.set_xlabel('Longitude [dd]')
                ax0.tick_params(axis='x', rotation=-45)
                ax0.set_ylabel('Latitude [dd]')
                if mdf is not None:
                    cbar = plt.colorbar(c2,ax=ax0,fraction=0.046, pad=0.07,
                                        orientation='horizontal',extend='both')
                    cbar.ax.set_xlabel('Absolute error (Observation-Model) [m]')
                
    #                    axpos = ax0.get_position()
    #                    pos1 = cbar.ax.get_position()
    #                    cbar.ax.set_position([pos1.x0,pos1.y0+axpos.height,pos1.width,pos1.height])
                    cbar.ax.xaxis.set_ticks_position('top')
                    cbar.ax.xaxis.set_label_position('top')
                # setup axis labels
                ngrid_lines = 3.
                xlocs = np.arange(bbox[0],bbox[2]+(bbox[2]-bbox[0])/ngrid_lines,(bbox[2]-bbox[0])/ngrid_lines)
                ylocs = np.arange(bbox[1],bbox[3]+(bbox[3]-bbox[1])/ngrid_lines,(bbox[3]-bbox[1])/ngrid_lines)
                ax0.set_xticks(xlocs)
                ax0.set_yticks(ylocs)
                lon_formatter = LongitudeFormatter(zero_direction_label=True)
                lat_formatter = LatitudeFormatter()
                ax0.xaxis.set_major_formatter(lon_formatter)
                ax0.yaxis.set_major_formatter(lat_formatter)
                gl=ax0.gridlines(xlocs=xlocs,ylocs=ylocs)
                ax0.set_ylim(np.min(ylocs),np.max(ylocs))
                ax0.set_xlim(np.min(xlocs),np.max(xlocs))
    
                fig.savefig(out_fig,format='pdf',papertype='letter',orientation='landscape')
                fig.savefig(out_fig2,dpi=300,format='png',papertype='letter',orientation='landscape')
            plt.close(fig)
            # ----------- Model run finishing statements -----------
            print('Elapsed time: {0:4.1f} min'.format((time.time()-model_start)/60.))
            print('Model finished at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
            print('--------------------------------------------\n')
            region_data.extend(out_data)
      
        store_model_data.extend(region_data)
    
        
        if save_csv:
            header_cols = ['ca_county','run_name',
                               'C_nwell','C_R2','C_slope',
                               'H_nwell','H_R2','H_slope',
                               'CA_nwell','CA_R2','CA_slope']
            if not os.path.isfile(out_all_fname):
                
                with open(out_all_fname,'w') as f:
                    # Write header
                    f.write('{} \n'.format(','.join(header_cols)))
                    for iline in out_list:
                        f.write('{0:s},{1:s},{2:8.0f},{3:6.4f},{4:6.4f},{5:8.0f},{6:6.4f},{7:6.4f},{8:8.0f},{9:6.4f},{10:6.4f} \n'.format(*iline))
            else:
                add_df = pd.DataFrame(out_list,columns=header_cols)
                output_df = pd.concat([output_df,add_df],ignore_index=True)
                output_df.to_csv(out_all_fname,index=False)
                
    # Save all data
    if save_csv:
        out_cols = ['Kh','ca_county','run_name','longitude','latitude','gs_elev_m','obs_type','id','obs_wl','obs_wl_min','obs_wl_max','model_wl']
    
        out_df = pd.DataFrame(store_model_data,columns=out_cols)
        if os.path.isfile(summary_fname):
            out_df = pd.concat([summary_df,out_df],ignore_index=True,sort=True)
            
        out_df.to_csv(summary_fname,index=False)