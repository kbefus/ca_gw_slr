# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:38:53 2018

@author: kbefus
"""
import sys,os
import numpy as np
import time 

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
plt.rc('legend',**{'fontsize':9})

res_dir = r'/mnt/data2/CloudStation'
code_dir = os.path.join(res_dir,r'ca_slr/scripts')
#code_dir = r'C:\Users\kbefus\OneDrive - University of Wyoming\ca_slr\scripts'
sys.path.insert(1,code_dir)

from cgw_model.cgw_utils import cgw_general_utils as cgu
from cgw_model.cgw_utils import cgw_feature_utils as cfu
from cgw_model.cgw_utils import cgw_raster_utils as cru

#from scipy.stats import logistic
from scipy.special import erf
import geopandas as gpd
import rasterio
from rasterio import mask
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.io import MemoryFile

import affine
from shapely.ops import shared_paths,snap,linemerge
#%%

# Blend/merge options
mid_loc = 0.5 # center logistic curve on 0.5
scale=6 # size of transition in proportion to distance of intersect (10=86% of transition within .1 of .5)
locvals = np.linspace(0,1,100) # number of discritization points to set original logistic curve, interpolate between
blend_func = lambda x: (erf((x-mid_loc)*scale)+1)/2.
#cgu.plt.plot(locvals,blend_func(locvals))
#print(blend_func(locvals[0]),blend_func(locvals[-1]))

utm10n = 3717
utm11n = 3718

def get_meta(in_fname=None):
    with rasterio.open(in_fname) as src:
        meta=src.meta.copy()
    return meta

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
        

def raster_clip_poly(in_fname=None,poly=None,all_touched=True,crop=False,filled=True):
    with rasterio.open(in_fname) as src:
        out_rast,tform = mask.mask(src,[poly],crop=crop,
                                            all_touched=all_touched,filled=filled)
        out_rast[out_rast==src.nodata] = np.nan
    
    if crop:
        # Need to reload
        with rasterio.open(in_fname) as src:
            out_rast_temp,tform2 = mask.mask(src,[poly],crop=False,
                                                all_touched=all_touched,filled=False)
        mask_out = ~out_rast_temp.mask # flip boolean
    elif filled:
        mask_out = ~out_rast.mask
    else:
        # Not ideal mask, since doesn't include non-nan areas inside of poly
        mask_out = np.zeros_like(out_rast,dtype=bool)
        mask_out[~np.isnan(out_rast)] = True
    
    out_rast = out_rast.squeeze()     
    ny,nx = out_rast.shape   
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
        
    return [X,Y,out_rast],mask_out.squeeze()

def merge_tifs(model_dict1=None,model_dict2=None,intersect_poly=None,
               ibuff=100.,ndistpts=100,snap_dist=50,cell_spacing=10,
               wt_fmt='{}_wtdepth.tif',head_fmt='{}_head.tif',
               cell_fmt='{}_celltypes.tif',out_dir=None,buff_intersection_poly=False,
               force_new=[False]*2,small_value=0,extend_m=1e3,
               set_marine_mask=False,new_intersect=False,
               marine_mask_val=-500, marine_cell_type=-2.,save_fig=True):
    #%%
#    ibuff=100
#    ndistpts=100
#    extend_m=1e3
#    out_dir=slr_dir
#    new_intersect=True
#    marine_cell_type=-2.
#    marine_mask_val=-500
#    force_new=[False]*2
#    buff_intersection_poly=False
    
    nonnan_mat1 = np.array([False])
    
    # Make head filenames
    head_fname1 = os.path.join(model_dict1['in_dir'],model_dict1['model_name'],
                               head_fmt.format(model_dict1['model_name']))
    head_fname2 = os.path.join(model_dict2['in_dir'],model_dict2['model_name'],
                               head_fmt.format(model_dict2['model_name']))
    fname1 = os.path.splitext(head_fname1)[0][:-5]
    fname2 = os.path.splitext(head_fname2)[0][:-5]
    
    # Set marine mask
    ctype_fname1 = os.path.join(model_dict1['in_dir'],model_dict1['model_name'],
                           cell_fmt.format(model_dict1['model_name']))
    ctype_fname2 = os.path.join(model_dict2['in_dir'],model_dict2['model_name'],
                           cell_fmt.format(model_dict2['model_name']))
    
    if out_dir is None:
        new_wt_fname1 = wt_fmt.format('_'.join([fname1,'merged']))
        new_wt_fname2 = wt_fmt.format('_'.join([fname2,'merged']))
        new_head_fname1=head_fmt.format('_'.join([fname1,'merged']))
        new_head_fname2=head_fmt.format('_'.join([fname2,'merged']))
    else:
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
            
        new_wt_fname1 = os.path.join(out_dir,wt_fmt.format('_'.join([os.path.basename(fname1),'merged'])))
        new_wt_fname2 = os.path.join(out_dir,wt_fmt.format('_'.join([os.path.basename(fname2),'merged'])))
        new_head_fname1=os.path.join(out_dir,head_fmt.format('_'.join([os.path.basename(fname1),'merged'])))
        new_head_fname2=os.path.join(out_dir,head_fmt.format('_'.join([os.path.basename(fname2),'merged'])))

    
    
    # Check to see if merged head tif's already made, use those instead 
    if not force_new[0]:
        if os.path.isfile(new_head_fname1):
            head_fname1 = new_head_fname1
    if not force_new[1]:    
        if os.path.isfile(new_head_fname2):
            head_fname2 = new_head_fname2
    
    print(" {} to {}".format(os.path.basename(head_fname1),os.path.basename(head_fname2)))
    with rasterio.open(head_fname1) as f1:
        temp_profile1 = f1.profile
    
    with rasterio.open(head_fname2) as f2:
        temp_profile2 = f2.profile
#    temp_meta1 = get_meta(head_fname1)
#    temp_meta2 = get_meta(head_fname2)
    
#    new_intersect = False
#    # Reproject input domains to head file crs's
#    if model_dict1['df'].crs != temp_meta1['crs']:
#        model_dict1['df']=model_dict1['df'].to_crs(temp_meta1['crs'].data)
#        new_intersect=True
#        
#    if model_dict2['df'].crs != temp_meta2['crs']:
#        model_dict2['df']=model_dict2['df'].to_crs(temp_meta2['crs'].data)
#        new_intersect = True
#        
    if new_intersect:
        intersect_poly = model_dict1['df'].geometry.values[0].intersection(model_dict2['df'].geometry.values[0])
        
#    print(head_fname1)
#    print(temp_meta1['crs'].to_string())
#    print(head_fname2)
#    print(temp_meta2['crs'].to_string())
    if model_dict1['df']['ca_region'].values[0] != model_dict2['df']['ca_region'].values[0] and \
       model_dict2['df']['ca_region'].values[0] in ['soca']:
            
        # Need to make intersect_poly in consistent crs 
        temp_df2 = model_dict2['df'].copy()
        temp_df2 = temp_df2.to_crs(model_dict1['df'].crs) # transform to other crs
        domain2 = temp_df2.iloc[0].geometry
        domain2 = snap(domain2,model_dict1['df'].iloc[0].geometry,snap_dist)
        intersect_poly = model_dict1['df'].iloc[0].geometry.intersection(domain2)
        
        # And make other intersection polygon in 2nd domain's crs
        temp_df1 = model_dict1['df'].copy()
        temp_df1 = temp_df1.to_crs(model_dict2['df'].crs) # transform to other crs
        domain1_temp = temp_df1.iloc[0].geometry
        domain1_temp = snap(domain1_temp,model_dict2['df'].iloc[0].geometry,snap_dist)
        intersect_poly2 = model_dict2['df'].iloc[0].geometry.intersection(domain1_temp)
#        crs1 = CRS.from_epsg(utm10n)
#        crs2 = CRS.from_epsg(utm11n)
    else:
        domain2 = model_dict2['df'].geometry.values[0]
#        if model_dict1['region'] in ['soca']:
#            crs1 = CRS.from_epsg(utm11n)
#            crs2 = CRS.from_epsg(utm11n)
#        else:
#            crs1 = CRS.from_epsg(utm10n)
#            crs2 = CRS.from_epsg(utm10n)
        
    
    if intersect_poly.geom_type != 'Polygon':
        intersect_poly =  [i for i in intersect_poly.geoms if i.geom_type == 'Polygon'][0]
    
    # Can apply an interior buffer for the intersection polygon area
    ipoly = intersect_poly.buffer(ibuff,cap_style=1,resolution=1,join_style=2)
    if buff_intersection_poly:
        try:
            buff_domain1 = model_dict1['df'].geometry.values[0].buffer(ibuff,cap_style=1,resolution=1,join_style=2)
            buff_domain1 = snap(buff_domain1,ipoly,ibuff)
            buff_domain2 = domain2.buffer(ibuff,cap_style=1,resolution=1,join_style=2)
            buff_domain2 = snap(buff_domain2,ipoly,ibuff)
        
            spath1,_ = shared_paths(buff_domain1.boundary,ipoly.boundary)
            spath2,_ = shared_paths(buff_domain2.boundary,ipoly.boundary)
        except:
            spath1,_ = shared_paths(model_dict1['df'].geometry.values[0].boundary,intersect_poly.boundary)
            spath2,_ = shared_paths(domain2.boundary,intersect_poly.boundary)
    else:
        spath1,_ = shared_paths(model_dict1['df'].geometry.values[0].boundary,intersect_poly.boundary)
        spath2,_ = shared_paths(domain2.boundary,intersect_poly.boundary)
        
    line1 = linemerge([tempbound for tempbound in spath1.geoms if not tempbound in spath2.geoms])
    line2 = linemerge([tempbound for tempbound in spath2.geoms if not tempbound in spath1.geoms]) # only need one

    # need to make more points than just the ends of the line with ndistpts
    dist_line_xy = np.vstack([np.array(line1.interpolate(i,normalized=True).xy).T for i in np.linspace(0,1,ndistpts)])
    other_line_xy = np.vstack([np.array(line2.interpolate(i,normalized=True).xy).T for i in np.linspace(0,1,ndistpts)])
    
    # Extend boundary line used for caclulating distance from boundary
    if extend_m > 0:
        last_n_pts = 5
        if dist_line_xy.shape[0]<=last_n_pts:
            last_n_pts = dist_line_xy.shape[0]-1
            
        end_vect1 = dist_line_xy[last_n_pts]-dist_line_xy[0]
        end_vect2 = dist_line_xy[-last_n_pts]-dist_line_xy[-1]
        end_vect1_norm = end_vect1/np.linalg.norm(end_vect1)
        end_vect2_norm = end_vect2/np.linalg.norm(end_vect2)
        # add new points to beginning and end of line
        dist_line_xy = np.vstack([dist_line_xy[0]+[-end_vect1_norm*extend_m],dist_line_xy,dist_line_xy[-1]+[-end_vect2_norm*extend_m]])
    
    line_length = dist_line_xy.shape[0]
    blend_switch = True

    
#%%
	# Extract overlapping area of models
    clip_dict1 = {'in_fname':head_fname1,'poly':ipoly,'crop':True}
    [x1,y1,h1],m1 = raster_clip_poly(**clip_dict1)
    # For cell types too
    clip_dict1 = {'in_fname':ctype_fname1,'poly':ipoly,'crop':True}
    [_,_,ct1],_ = raster_clip_poly(**clip_dict1)
    
    if model_dict1['df']['ca_region'].values[0] != model_dict2['df']['ca_region'].values[0] and \
        model_dict2['df']['ca_region'].values[0] in ['soca']:
            
        # need to project the soca model file, should only ever be the second one
        ipoly2 = intersect_poly2.buffer(ibuff,cap_style=1,resolution=1,join_style=2)
        clip_dict2 = {'in_fname':head_fname2,'poly':ipoly2,'crop':True}
        [x2,y2,h2],m2 = raster_clip_poly(**clip_dict2)
#        x2,y2 = cgu.osr_transform(XY=[x2,y2],proj_in=utm11n,proj_out=utm10n)
        clip_dict2 = {'in_fname':ctype_fname2,'poly':ipoly2,'crop':True}
        [_,_,ct2],_ = raster_clip_poly(**clip_dict2)
    else:
        clip_dict2 = {'in_fname':head_fname2,'poly':ipoly,'crop':True}
        [x2,y2,h2],m2 = raster_clip_poly(**clip_dict2)
        clip_dict2 = {'in_fname':ctype_fname2,'poly':ipoly,'crop':True}
        [_,_,ct2],_ = raster_clip_poly(**clip_dict2)

    # Make new grid extents and transform
    left, bottom, right, top = ipoly.bounds
    xres = cell_spacing
    yres = cell_spacing
    overlap_transform = affine.Affine(xres, 0.0, left,
                                      0.0, -yres, top)
    overlap_height = np.round((top-bottom)/yres)
    overlap_width = np.round((right-left)/xres)
    
    
    vrt_options = {'resampling': Resampling.bilinear,
                    'transform': overlap_transform,
                    'crs':temp_profile1['crs'],
                    'height': overlap_height,
                    'width': overlap_width,'nodata':-200,
                    'tolerance':0.01,'all_touched':True,'num_threads':4,
                    'sample_grid':'YES','sample_steps':100,
                    'source_extra':10}

    # Make new spatial grid
    newX,newY = xy_from_affine(overlap_transform,overlap_width,overlap_height)
    
#    new_x=np.arange(max_extent[0],max_extent[2]+cell_spacing,cell_spacing)
#    new_y=np.arange(max_extent[1],max_extent[3]+cell_spacing,cell_spacing)
#    newX,newY = np.meshgrid(new_x,new_y)

    # Set marine values to nan
    h1[ct1==marine_cell_type] = np.nan
    h2[ct2==marine_cell_type] = np.nan

    nonnan1 = ~np.isnan(h1)
    nonnan2 = ~np.isnan(h2)
    if len(nonnan1.nonzero()[0])>0 and len(nonnan2.nonzero()[0])>0:
        
        vrt_options_ct = vrt_options.copy()
        vrt_options_ct.update({'resampling':Resampling.nearest})
        
        # Need to load head in with nodata value AND marine cells as nans
        # This will make sure the bilinear interp doesn't spread marine_value on land
        with rasterio.open(head_fname1) as src:
            head_temp = src.read()[0]
#            head_temp[head_temp==src.nodata]=np.nan
            # Also read in cell_types
            with rasterio.open(ctype_fname1) as ctsrc:
                ct1 = ctsrc.read()[0]
                head_temp[ct1==marine_cell_type] = np.nan
                
            # Save head as memory file
            with MemoryFile() as memfile:
                with memfile.open(**src.profile) as dataset:
                    dataset.write(np.expand_dims(head_temp,0))
                    with WarpedVRT(dataset, **vrt_options) as vrt:
                        new_h1 = vrt.read()[0]
                        new_h1[new_h1 == vrt.nodata] = np.nan
                    # Need to fill in edges that got messed up by the interpolation
                    new_vrt_options=vrt_options.copy()
                    new_vrt_options['resampling'] = Resampling.nearest
                    with WarpedVRT(dataset, **new_vrt_options) as vrt:
                        new_h1b = vrt.read()[0]
                        new_h1b[new_h1b == vrt.nodata] = np.nan
                    new_h1[np.isnan(new_h1) & ~np.isnan(new_h1b)] = new_h1b[np.isnan(new_h1) & ~np.isnan(new_h1b)]
            memfile.close()
        with rasterio.open(head_fname2) as src:
            head_temp = src.read()[0]
#            head_temp[head_temp==src.nodata]=np.nan
            # Also read in cell_types
            with rasterio.open(ctype_fname2) as ctsrc:
                ct2 = ctsrc.read()[0]
                head_temp[ct2==marine_cell_type] = np.nan
                
            # Save head as memory file
            with MemoryFile() as memfile:
                with memfile.open(**src.profile) as dataset:
                    dataset.write(np.expand_dims(head_temp,0))
                    with WarpedVRT(dataset, **vrt_options) as vrt:
                        new_h2 = vrt.read()[0]
                        new_h2[new_h2 == vrt.nodata] = np.nan
                    new_vrt_options=vrt_options.copy()
                    new_vrt_options['resampling'] = Resampling.nearest
                    with WarpedVRT(dataset, **new_vrt_options) as vrt:
                        new_h2b = vrt.read()[0]
                        new_h2b[new_h2b == vrt.nodata] = np.nan
                    new_h2[np.isnan(new_h2) & ~np.isnan(new_h2b)] = new_h2b[np.isnan(new_h2) & ~np.isnan(new_h2b)]
            memfile.close()
            
        with rasterio.open(ctype_fname1) as src:
            with WarpedVRT(src, **vrt_options_ct) as vrt:
                ct1 = vrt.read()[0]
                ct1[ct1 == vrt.nodata] = np.nan
                
        with rasterio.open(ctype_fname2) as src:
            with WarpedVRT(src, **vrt_options_ct) as vrt:
                ct2 = vrt.read()[0]
                ct2[ct2 == vrt.nodata] = np.nan
#%%        
        # Rasterio interpolation method; model results to new grid
#        with rasterio.open(head_fname1) as src:
#            with WarpedVRT(src, **vrt_options) as vrt:
#                new_h1 = vrt.read()[0]
#                new_h1[new_h1 == vrt.nodata] = np.nan
#                
#        with rasterio.open(head_fname2) as src:
#            with WarpedVRT(src, **vrt_options) as vrt:
#                new_h2 = vrt.read()[0]
#                new_h2[new_h2 == vrt.nodata] = np.nan
                
        # Interpolate model results to new grid
#        new_h1 = cru.griddata(np.c_[x1[nonnan1],y1[nonnan1]],
#                                    h1[nonnan1],(newX,newY))
#        new_h2 = cru.griddata(np.c_[x2[nonnan2],y2[nonnan2]],
#                                    h2[nonnan2],(newX,newY))
        with np.errstate(invalid='ignore'):
            new_h1[new_h1<small_value] = np.nan
            new_h2[new_h2<small_value] = np.nan
        

          
#        new_h1[ct1==marine_cell_type] = np.nan
#        new_h2[ct2==marine_cell_type] = np.nan
        
        
        # Calculate distance from overlapping grid locations to each domain centroid
        nonnan_mat1 = ~np.isnan(new_h1) & ~np.isnan(new_h2)
    else:
        nonnan_mat1 = np.zeros_like(newX,dtype=bool) # all false
    
    # Load original rasters
    x1_orig,y1_orig,h1_orig = read_geotiff(head_fname1)
    x2_orig,y2_orig,h2_orig = read_geotiff(head_fname2)
    if len(nonnan_mat1.nonzero()[0])>0 and line_length>0:
        dmat1a = cgu.calc_dist(dist_line_xy,list(zip(newX[nonnan_mat1].ravel(),newY[nonnan_mat1].ravel())))
        dmat1a = np.min(dmat1a,axis=0) # find closest values
        
        # Calculate distance between model bounds
        dist1 = np.min(cgu.calc_dist(dist_line_xy,other_line_xy),axis=0)
        
#        dmat1a = cgu.calc_dist([[c1x,c1y]],list(zip(newX[nonnan_mat1].ravel(),newY[nonnan_mat1].ravel())))
        #dmat1b = cgu.calc_dist([[c2x,c2y]],list(zip(newX[nonnan_mat1].ravel(),newY[nonnan_mat1].ravel())))
        
        # Normalize blending by distance between unique model lines
        dmat1a_norm = dmat1a/np.nanmedian(dist1)
        dmat1a_norm = 1.-(dmat1a_norm-0.)/(np.nanmax(dmat1a_norm)-0.)
#        dmat1a_mat = dmat1a_norm.reshape(newX.shape) # if not using nonnan_mat1
#        dmat1a_norm[dmat1a_norm<0] = 0
#        dmat1a_norm[dmat1a_norm>1] = 1
        dmat1a_mat = np.nan*np.zeros_like(newX)
        dmat1a_mat[nonnan_mat1] = dmat1a_norm.ravel()
        
        # Old way
#        dmat1a_norm = dmat1a/np.mean(dmat1a)
#        dmat1a_norm = 1.-(dmat1a_norm-np.min(dmat1a_norm))/(np.max(dmat1a_norm)-np.min(dmat1a_norm))
#        #dmat1b_norm= dmat1b/np.mean(dmat1b)
#        #dmat1b_norm = 1.-(dmat1b_norm-np.min(dmat1b_norm))/(np.max(dmat1b_norm)-np.min(dmat1b_norm))
#    
#        dmat1a_mat = np.nan*np.zeros_like(newX)
#        #dmat1b_mat = np.nan*np.zeros_like(newX)
#        dmat1a_mat[nonnan_mat1] = dmat1a_norm.ravel()
        
        #Extrapolate weights by nearest neighbor
        dmatnans = np.isnan(dmat1a_mat)   
        if len(dmat1a_mat[~dmatnans])>0:
            xy = np.c_[newX[~dmatnans],newY[~dmatnans]]
            xy2 = (newX,newY)                
            
            dmat2 = cru.griddata(xy,dmat1a_mat[~dmatnans],
                                xy2,method='nearest')
            dmat2[~dmatnans] = dmat1a_mat.copy()[~dmatnans]
        else:
            dmat2 = dmat1a_mat.copy()
            plt.pcolormesh(newX,newY,dmat1a_mat)
            cfu.plot_shp([ipoly],facecolor='none')
    
        # Use error function with center at 0.5
        blend_m1 = np.round(blend_func(dmat2),decimals=5)
        with np.errstate(invalid='ignore'):
            blend_m1[(new_h1<0.) | (new_h2<0.)] = np.nan
#        blend_m1[~nonnan_mat1] = np.nan
        
        if save_fig:
            fig,ax = plt.subplots()
            blend_m2 = blend_m1.copy()
            blend_m2[(ct1==marine_cell_type) | (ct2==marine_cell_type)] = np.nan
            blend_m2[~nonnan_mat1] = np.nan
            c1 = ax.pcolormesh(newX,newY,np.ma.masked_invalid(blend_m2))
            cbar = plt.colorbar(c1,ax=ax)
            cbar.set_label('Blend ratio')
            
            cfu.plot_shp([model_dict1['df'].geometry.values[0],domain2,ipoly],
                         facecolor='none',ax=ax)
            
            ax.set_ylabel('UTM y coordinates [m]')
            ax.set_xlabel('UTM x coordinates [m]')
            
            fig_dir = os.path.join(out_dir,'figs')
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            
            mod1name = '_'.join([model_dict1['df']['ca_region'].values[0],
                                 str(model_dict1['df']['Id'].values[0])])
            mod2name = '_'.join([model_dict2['df']['ca_region'].values[0],
                                 str(model_dict2['df']['Id'].values[0])])
            ax.set_title('{} to {}'.format(mod1name,mod2name))
            fig_name = 'blend_{}to{}.png'.format(os.path.basename(fname1),os.path.basename(fname2))
            out_fig = os.path.join(fig_dir,fig_name)
            plt.tight_layout()
            fig.savefig(out_fig,dpi=300,format='png')
            plt.close(fig)
        
        if not blend_switch:
            out_h = blend_m1*new_h1 + (1.-blend_m1)*new_h2
        else:
            out_h = blend_m1*new_h2 + (1.-blend_m1)*new_h1
        
        # Insert head values from original rasters so long as nan values are near that raster's "good" area
        out_h[np.isnan(out_h) & ~np.isnan(new_h1) & (blend_m1<0.5)] = new_h1[np.isnan(out_h) & ~np.isnan(new_h1) & (blend_m1<0.5)]
        out_h[np.isnan(out_h) & ~np.isnan(new_h2) & (blend_m1>0.5)] = new_h2[np.isnan(out_h) & ~np.isnan(new_h2) & (blend_m1>0.5)]
        
        # Areas that should be filled by other raster fill in with gaussian or interp
        out_h_nans = np.isnan(out_h)
        in_h_nonnans1 = out_h_nans & ~np.isnan(new_h2)
#        out_h_fill1 = fillnodata(out_h,mask=~in_h_nonnans1,max_search_distance=ibuff)
        out_h_fill1 = cru.griddata(np.c_[newX[~out_h_nans],newY[~out_h_nans]],out_h[~out_h_nans],
                        (newX[in_h_nonnans1],newY[in_h_nonnans1]),method='nearest')
        
        out_h[in_h_nonnans1] = out_h_fill1
        out_h_nans = np.isnan(out_h)
        in_h_nonnans2 = out_h_nans & ~np.isnan(new_h1)
#        out_h_fill2 = fillnodata(out_h,mask=~in_h_nonnans2,max_search_distance=ibuff)
        out_h_fill2 = cru.griddata(np.c_[newX[~out_h_nans],newY[~out_h_nans]],out_h[~out_h_nans],
                        (newX[in_h_nonnans2],newY[in_h_nonnans2]),method='nearest')
#        out_h2 = out_h.copy()
        #[in_h_nonnans1]
        out_h[in_h_nonnans2] = out_h_fill2#[in_h_nonnans2]
        
#        cgu.quick_plot(np.array([out_h_fill1,out_h_fill2,out_h,out_h2,in_h_nonnans1.astype(int),in_h_nonnans2.astype(int)]),ncols=2,vmin=0,vmax=20)

#        out_h_fill1[in_h_nonnans1] = out_h_fill1
        
        # --- Insert merged data into each head array ---
    	# Extract x,y values of original model results using mask
#        xtemp,ytemp = x1_orig[m1==1].reshape((-1,1)),y1_orig[m1==1].reshape((-1,1))
#        xtemp2,ytemp2 = x2_orig[m2==1].reshape((-1,1)),y2_orig[m2==1].reshape((-1,1))
    
    	# Interpolate from merged heads back to original model grid
        vrt_options_toh1 = vrt_options.copy()
        vrt_options_toh1.update({'transform':temp_profile1['transform'],
                                 'height': temp_profile1['height'],
                                 'width': temp_profile1['width']})

        vrt_options_toh2 = vrt_options.copy()
        vrt_options_toh2.update({'transform':temp_profile2['transform'],
                                 'height': temp_profile2['height'],
                                 'width': temp_profile2['width'],
                                 'crs':temp_profile2['crs']})
        
        overlap_meta = temp_profile1.copy()
        overlap_meta.update({'driver':'GTiff','height':overlap_height,
                          'transform':overlap_transform,'width':overlap_width,
                          'dtype':dmat1a_mat.dtype})

        # Create new grid from blended data
        with MemoryFile() as memfile:
            with memfile.open(**overlap_meta) as dataset:
                dataset.write(np.expand_dims(out_h,0))
                with WarpedVRT(dataset, **vrt_options_toh1) as vrt:
                    htemp1 = vrt.read()[0]
                    htemp1[htemp1==vrt.nodata] = np.nan
                
                vrt_options_toh1b=vrt_options_toh1.copy()
                vrt_options_toh1b['resampling'] = Resampling.nearest
                with WarpedVRT(dataset, **vrt_options_toh1b) as vrt:
                    htemp1b = vrt.read()[0]
                    htemp1b[htemp1b==vrt.nodata] = np.nan
                    htemp1[np.isnan(htemp1) & ~np.isnan(htemp1b)] = htemp1b[np.isnan(htemp1) & ~np.isnan(htemp1b)]
        memfile.close()
        with MemoryFile() as memfile:
            with memfile.open(**overlap_meta) as dataset:
                dataset.write(np.expand_dims(out_h,0))
                with WarpedVRT(dataset, **vrt_options_toh2) as vrt:
                    htemp2 = vrt.read()[0]
                    htemp2[htemp2==vrt.nodata] = np.nan
                vrt_options_toh2b=vrt_options_toh2.copy()
                vrt_options_toh2b['resampling'] = Resampling.nearest
                with WarpedVRT(dataset, **vrt_options_toh2b) as vrt:
                    htemp2b = vrt.read()[0]
                    htemp2b[htemp2b==vrt.nodata] = np.nan
                    htemp2[np.isnan(htemp2) & ~np.isnan(htemp2b)] = htemp2b[np.isnan(htemp2) & ~np.isnan(htemp2b)]
        memfile.close()
#        htemp1 = cru.subsection_griddata([newX,newY],out_h,(xtemp,ytemp),nsections=1)
#        htemp2 = cru.subsection_griddata([newX,newY],out_h,(xtemp2,ytemp2),nsections=1)
    
        # Insert new values into original model grid
        new_h3 = h1_orig.copy()
        with np.errstate(invalid='ignore'):
            new_h3[~np.isnan(new_h3) & ~np.isnan(htemp1)] = htemp1[~np.isnan(new_h3) & ~np.isnan(htemp1)]#.ravel()
            new_h3[htemp1<0] = h1_orig[htemp1<0]
        # Fill any nans adopted from htemp1 with original head values
#        new_h3[np.isnan(new_h3) & ~np.isnan(h1_orig)] = h1_orig[np.isnan(new_h3) & ~np.isnan(h1_orig)]
    
        new_h4 = h2_orig.copy()
        with np.errstate(invalid='ignore'):
            new_h4[~np.isnan(new_h4) & ~np.isnan(htemp2)] = htemp2[~np.isnan(new_h4) & ~np.isnan(htemp2)]#.ravel()
            new_h4[htemp2<0] = h2_orig[htemp2<0]
#        new_h4[np.isnan(new_h4) & ~np.isnan(h2_orig)] = h2_orig[np.isnan(new_h4) & ~np.isnan(h2_orig)]
    else:
        # Use original data with no merging
        print('*Using input data as merged copy since no overlap found.*')
        print('# of nonnans = {}'.format(len(nonnan_mat1.nonzero()[0])))
        print('Line length =  {}'.format(line_length))
        
        if save_fig:
            fig,ax = plt.subplots()
            
            cfu.plot_shp([model_dict1['df'].geometry.values[0],domain2,ipoly],
                         facecolor='none',ax=ax)
            
            ax.set_ylabel('UTM y coordinates [m]')
            ax.set_xlabel('UTM x coordinates [m]')
            
            fig_dir = os.path.join(out_dir,'figs')
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            
            mod1name = '_'.join([model_dict1['df']['ca_region'].values[0],
                                 str(model_dict1['df']['Id'].values[0])])
            mod2name = '_'.join([model_dict2['df']['ca_region'].values[0],
                                 str(model_dict2['df']['Id'].values[0])])
            ax.set_title('{} to {}'.format(mod1name,mod2name))
            fig_name = 'blend_{}to{}.png'.format(os.path.basename(fname1),os.path.basename(fname2))
            out_fig = os.path.join(fig_dir,fig_name)
            plt.tight_layout()
            fig.savefig(out_fig,dpi=300,format='png')
            plt.close(fig)
        
        new_h3 = h1_orig.copy()
        new_h4 = h2_orig.copy()
    
    # Calculate new water table depths and save new files
#    wkt1,gt1 = cru.load_grid_prj(head_fname1,gt_out=True)
#    wkt2,gt2 = cru.load_grid_prj(head_fname2,gt_out=True)
    meta1 = temp_profile1.copy()
    meta2 = temp_profile2.copy()
    meta1.update({'dtype':rasterio.float32})
    meta2.update({'dtype':rasterio.float32})
    # Elevation = wt_depth+head
	# New water table depth = Elevation-new_head
    
    # Model 1
    wt1 = rasterio.open(wt_fmt.format(fname1)).read().squeeze()
    wt1[wt1==meta1['nodata']]=np.nan
    wt1[np.isnan(h1_orig)] = np.nan # set values outside model domain to nan
    ct1 = rasterio.open(ctype_fname1).read().squeeze()
    out_wt1 = (wt1+h1_orig)-new_h3
    # Set marine mask
    if set_marine_mask:
        out_wt1[ct1==marine_cell_type] = marine_mask_val
        new_h3[ct1==marine_cell_type] = marine_mask_val
        
#    with np.errstate(invalid='ignore'):    
#        out_wt1[(out_wt1<-1.) & (out_wt1>marine_mask_val) & ~np.isnan(out_wt1)] = np.nan # Remove artifacts from out of bounds topography
    
    new_h3[h1_orig==meta1['nodata']]=np.nan
    new_h3[np.isnan(h1_orig)] = np.nan # set values outside model domain to nan
    
    with rasterio.open(new_wt_fname1,'w',**meta1) as dest:
        dest.write(np.expand_dims(out_wt1,axis=0))
    
    with rasterio.open(new_head_fname1,'w',**meta1) as dest:
        dest.write(np.expand_dims(new_h3,axis=0))
    
#    cru.write_gdaltif(new_wt_fname1,x1_orig,y1_orig,out_wt1,proj_wkt=wkt1,geodata=gt1)    
#    cru.write_gdaltif(new_head_fname1,x1_orig,y1_orig,new_h3,proj_wkt=wkt1,geodata=gt1)    
    
    # Model 2
    wt2 = rasterio.open(wt_fmt.format(fname2)).read().squeeze()
    wt2[wt2==meta2['nodata']]=np.nan
    wt2[np.isnan(h2_orig)] = np.nan # set values outside model domain to nan
    ct2 = rasterio.open(ctype_fname2).read().squeeze()
    out_wt2 = (wt2+h2_orig)-new_h4
    # Set marine mask
    if set_marine_mask:
        out_wt2[ct2==marine_cell_type] = marine_mask_val
        new_h4[ct2==marine_cell_type] = marine_mask_val
        
#    with np.errstate(invalid='ignore'):    
#        out_wt2[(out_wt2<-1.) & (out_wt2>marine_mask_val) & ~np.isnan(out_wt2)] = np.nan # Remove artifacts from out of bounds topography
    new_h4[h2_orig==meta2['nodata']]=np.nan
    new_h4[np.isnan(h2_orig)] = np.nan # set values outside model domain to nan
    
    with rasterio.open(new_wt_fname2,'w',**meta2) as dest:
        dest.write(np.expand_dims(out_wt2,axis=0))
    
    with rasterio.open(new_head_fname2,'w',**meta2) as dest:
        dest.write(np.expand_dims(new_h4,axis=0))
#    cru.write_gdaltif(new_wt_fname2,x2_orig,y2_orig,out_wt2,proj_wkt=wkt2,geodata=gt2)
#    cru.write_gdaltif(new_head_fname2,x2_orig,y2_orig,new_h4,proj_wkt=wkt2,geodata=gt2)


#%%

# 1) Need to load model heads
# 2) Apply merge
# 3) Load topography and calculate merged gw depths


ca_regions = ['norca','paca','sfbay','cenca','soca']
research_dir = r'/mnt/762D83B545968C9F'
#research_dir = r'C:\Users\kbefus\OneDrive - University of Wyoming\ca_slr'
data_dir = os.path.join(research_dir,'data')
model_types = ['model_lmsl_noghb','model_mhhw_noghb'][:1]
model_dirs = [os.path.join(research_dir,mtype) for mtype in model_types]
active_date = '27Dec19'
out_dir = os.path.join(data_dir,'masked_wt_depths_gdal_{}'.format(active_date))

for tempdir in [out_dir,data_dir]:
    if not os.path.isdir(tempdir):
        os.makedirs(tempdir)

id_col = 'Id'
sealevel_elevs = np.hstack([np.arange(0,2.25,.25),2.5,3.,5.])[3:]# m
#datum_type = 'MHHW'
cell_spacing = 10. # meters
marine_mask_val = -500.
Kh_vals = [0.1,1.,10.][1:2]
#research_dir = r'C:\research\kbefus\ca_slr'

#model_name_fmt = '{0:s}_{1:d}_{2}_slr{3:3.2f}m_Kh{4:3.2f}_{5:.0f}m'
other_model_name_fmt = '{0:s}_{1:d}_{2}_slr{3:3.2f}m_Kh{4:3.1f}_{5:.0f}m' 
dirname_fmt = 'slr_{0:3.2f}_m_Kh{1:3.1f}mday' # all outputs to this dir format

#indirname_fmt = 'output_{0}_res{1}m_sl{2:3.2f}m_Kh{3:3.2f}'
inother_dirname_fmt = 'output_{0}_res{1}m_sl{2:3.2f}m_Kh{3:3.1f}' 
 
wt_fmt = '{}_wtdepth.tif'
head_fmt = '{}_head.tif'
cell_fmt = '{}_celltypes.tif' # also assign marine mask where cell_type==-2

shp_date = '11Feb19'
nmodel_domains_shp = os.path.join(res_dir,'ca_slr','data','ca_{}_slr_gw_domains_{}.shp'.format('n',shp_date))
ndomain_df = gpd.read_file(nmodel_domains_shp)
#ndomain_df.crs = {'init': 'epsg:{}'.format(utm10n)} # UTM 10N

smodel_domains_shp = os.path.join(res_dir,'ca_slr','data','ca_{}_slr_gw_domains_{}.shp'.format('s',shp_date))
sdomain_df = gpd.read_file(smodel_domains_shp)
#sdomain_df.crs = {'init': 'epsg:{}'.format(utm11n)} # UTM 11N
#sdomain_df = sdomain_df.to_crs(ndomain_df.crs) # project to to UTM 10N

ncrs =  {'init': 'epsg:{}'.format(utm10n)}
scrs = {'init': 'epsg:{}'.format(utm11n)}
crs_dict = {'norca':ncrs,'paca':ncrs,'sfbay':ncrs,
            'cenca':ncrs,'soca':scrs}

all_modeldomains_df = cfu.pd.concat([ndomain_df,sdomain_df],ignore_index=True)
#all_modeldomains_df = all_modeldomains_df.iloc[40:42] # test only some if uncommented

# Snap nodes where smaller than snap_dist
snap_dist = 50 # m
snapped_polys = [snap(ipoly1[1].geometry,ipoly2[1].geometry,snap_dist) for ipoly1,ipoly2 in zip(all_modeldomains_df.iloc[:-1].iterrows(),all_modeldomains_df.iloc[1:].iterrows())]
snapped_polys.append(all_modeldomains_df.iloc[-1].geometry)
all_modeldomains_df['geometry'] = snapped_polys

intersect_polys = [ipoly1[1].geometry.intersection(ipoly2[1].geometry) for ipoly1,ipoly2 in zip(all_modeldomains_df.iloc[:-1].iterrows(),all_modeldomains_df.iloc[1:].iterrows())]

# Load county data
county_fname = os.path.join(res_dir,'ca_slr','data','gis','CA_coastal_Counties_TIGER2016_UTM10N.shp')
ind_joined_df = gpd.read_file(county_fname)

model_county_df = gpd.sjoin(ind_joined_df,all_modeldomains_df,how='inner',op='intersects')

redo_models = []#['cenca_27','soca_52','norca_6','cenca_32','cenca_39','paca_10']


# Auto run code \/\/\/\/
auto_run = False
if auto_run:
    final_model = other_model_name_fmt.format('soca',56,model_types[0].split('_')[1].upper(),
                                                           sealevel_elevs[-1],Kh_vals[-1],cell_spacing)
    final_file = os.path.join(model_dirs[0],'soca',inother_dirname_fmt.format(model_types[0].split('_')[1].upper(),
                                                                              cell_spacing,sealevel_elevs[-1],Kh_vals[-1]),
                               final_model,head_fmt.format(final_model))
    sleep_time = 60*30 # wait 30 min between checks
    while not os.path.isfile(final_file):
        time.sleep(sleep_time)
    
    time.sleep(sleep_time*2) # wait an extra hr to let straggling models wrap up.


#%%

ibuff = 100. # meters in (-) or out (+) from overlapping model edges
fover = [False,False] # force new merge from original file
active_cell_val = 1
all_start = time.time()
for model_dir,model_type in list(zip(model_dirs,model_types)):
    print('---------- {} --------------'.format(model_type))
    datum_type = model_type.split('_')[1].upper()
    for Kh in Kh_vals:
        print('------------ Kh = {} ---------------'.format(Kh))
#        if model_type in ['model_lmsl']:
#            if Kh==Kh_vals[0]:
#                fmt = model_name_fmt
#                indirfmt = indirname_fmt
#            else:
        fmt = other_model_name_fmt
        indirfmt = inother_dirname_fmt
#        else:
#            fmt = other_model_name_fmt
#            indirfmt = inother_dirname_fmt
            
        for sealevel_elev in sealevel_elevs:
            print('--- SL = {} ----'.format(sealevel_elev))
            slr_dir = os.path.join(out_dir,model_type,
                                   dirname_fmt.format(sealevel_elev,Kh))
        
            # Loop through neighboring models
            for [intersect_poly,[model_ind1,model_info1],[model_ind2,model_info2]] in \
            zip(intersect_polys,all_modeldomains_df.iloc[:-1].iterrows(),all_modeldomains_df.iloc[1:].iterrows()):
                model_start = time.time()
                # Collect raster information
                model_name1 = fmt.format(model_info1['ca_region'],
                                            model_info1['Id'],datum_type,
                                            sealevel_elev,Kh,cell_spacing)
                model_name2 = fmt.format(model_info2['ca_region'],
                                            model_info2['Id'],datum_type,
                                            sealevel_elev,Kh,cell_spacing)
                
                outputs_dir = indirfmt.format(datum_type,cell_spacing,sealevel_elev,Kh)
                
                model_out_dir1 = os.path.join(model_dir,model_info1['ca_region'],outputs_dir)
                model_out_dir2 = os.path.join(model_dir,model_info2['ca_region'],outputs_dir)
                    
                # Load active_bounds shapefiles
        #            shp1 = os.path.join(model_dir,model_info1['ca_region'],'active_bounds','{}.shp'.format(model_name1))
        #            shp2 = os.path.join(model_dir,model_info2['ca_region'],'active_bounds','{}.shp'.format(model_name2))
        #            shp1_df = gpd.read_file(shp1)
        #            shp2_df = gpd.read_file(shp2)
        #            
        #            if model_info1['ca_region'] != model_info2['ca_region'] and model_info2['ca_region'] in ['soca']:
        #                # need to project the soca model file, should only ever be the second one
        #                if not hasattr(shp2_df,'crs'):
        #                    shp2_df.crs = {'init': 'epsg:3718'} # UTM 11N
        #                
        #                shp2_df.to_crs(ndomain_df.crs)
        #            
        #            # Find overlap of main active features
        #            shp1_df = shp1_df[shp1_df['ID']==active_cell_val]
        #            shp2_df = shp2_df[shp2_df['ID']==active_cell_val]
        #            
        #            active_domain1 = shp1_df.loc[shp1_df.area.idxmax()].geometry
        #            active_domain2 = shp2_df.loc[shp2_df.area.idxmax()].geometry
        #            intersect_poly = active_domain1.intersection(active_domain2)
                
                # Convert iterrows back to geoseries
                model_df1 = gpd.GeoDataFrame(model_info1.to_frame().T,
                                             geometry=[model_info1.geometry],
                                             crs = crs_dict[model_info1['ca_region']])
                model_df2 = gpd.GeoDataFrame(model_info2.to_frame().T,
                                             geometry=[model_info2.geometry],
                                             crs = crs_dict[model_info2['ca_region']])
                
                # Run merge code
                model_dict1 = {'model_name':model_name1,'in_dir':model_out_dir1,
                               'df':model_df1}
                model_dict2 = {'model_name':model_name2,'in_dir':model_out_dir2,
                               'df':model_df2}
                
                
                merge_dict = {'model_dict1':model_dict1,'model_dict2':model_dict2,
                                  'intersect_poly':intersect_poly,'force_new':fover,
                                  'ibuff':ibuff,'out_dir':slr_dir,'new_intersect':True,
                                  'set_marine_mask':True}
                print("------ Merging model pair: {} to {} ------".format('_'.join([model_info1['ca_region'],
                                                                                   str(model_info1['Id'])]),
                                                                          '_'.join([model_info2['ca_region'],
                                                                                   str(model_info2['Id'])])))
                merge_tifs(**merge_dict)
                loop_time = time.time()-model_start
                print('Elapsed merge time: {0:4.1f} min'.format((time.time()-model_start)/60.))
                print('Merge finished at {}'.format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
                time.sleep(1) # pause
    print('Total time for merging: {0:4.1f} min'.format((time.time()-all_start)/60.))
            