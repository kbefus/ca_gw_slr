# -*- coding: utf-8 -*-
"""
Created on Wed May 11 14:14:05 2016

@author: kbefus
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from distutils.dir_util import copy_tree
import glob,os,warnings

from osgeo import osr,gdal

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

def gdal_error_handler(err_class, err_num, err_msg):
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print('Error Number: %s' % (err_num))
    print('Error Type: %s' % (err_class))
    print('Error Message: %s' % (err_msg))

grid_type_dict = {'active': 1.,
                  'inactive': 0.,
                  'noflow_boundary':-1.,
                  'nearshore':-2.,
                  'coastline': -5.,
                  'river': -10.,
                  'waterbody': -15.}
default_globe = ccrs.Globe('NAD83')
s_per_day = 86400.
m_per_ft = .3048

length_unit_dict = {0:'undefined',1:'foot',2:'meter',3:'centimeter'}
time_unit_dict = {0:'undefined',1:'second',2:'minute',3:'hour',4:'day',5:'year'}

def dd_to_m(dd,km_per_dd=110.):
    return dd*km_per_dd*1e3

def m_to_dd(m,km_per_dd=110.):
    return m/km_per_dd/1e3

def find_key(value,in_dict):
    try:
        return in_dict.keys()[in_dict.values().index(value)]
    except:
        return None

def perm_to_kh(perm_m2,density=1e3,viscosity=1e-3,gravity=9.80665):
    return (perm_m2*density*gravity)/viscosity # m/s

def kh_to_perm(kh,density=1e3,viscosity=1e-3,gravity=9.80665):
    return (kh*viscosity)/(density*gravity) # m**2

def calc_fw_head(phead,ehead,rho_f=1e3,rho_s=1025.):
    return ((phead*rho_s)/rho_f)+ehead


def vcont_func(dv_top,dv_bot,vk_top,vk_bot,dvcb=0.,vkcb=1):
    '''
    Calculate vertical conductance between layers/cells using
    Eq 5-40 in Modflow documentation. Defaults to non-confining bed.
    '''
    vcont =  1./(((0.5*dv_top)/vk_top)+ \
            ((0.5*dv_bot)/vk_bot) + \
            (dvcb/vkcb))
    return vcont

def xrot(x,y,theta):
    return x*np.cos(theta)-y*np.sin(theta)
    
def yrot(x,y,theta):
    return x*np.sin(theta)+y*np.cos(theta)

def shrink_ndarray(array_in,mask_in,shape_out):
    
    array_out = array_in.copy()

    if array_out.shape[-2:]!=shape_out[-2:]:
        if array_out.shape==mask_in.shape:
            # ndim mask = ndim array_in
            array_out = array_out[mask_in].reshape(shape_out)
        elif len(array_out.shape)==len(mask_in.shape)+1 and array_out.shape[-2:]==mask_in.shape[-2:]:
            # mask ndim+1 = ndim array_in
            if len(array_out.shape)==len(shape_out) and (array_out.shape != shape_out):
                array_out = array_out[:,mask_in].reshape(shape_out)
            elif (array_out.shape != shape_out):
                array_out = array_out[:,mask_in].reshape(np.hstack([array_out.shape[0],shape_out]))
            
    return array_out

def match_grid_size(mainXY=None, new_xyul=None, new_shape=None):
    """Convert to reduced size grid from upper left corner and array shape."""
    
    # Find xy position of new_xyul
    main_xy = np.array(list(zip(mainXY[0].flatten(),mainXY[1].flatten())))
    dif1 = main_xy-new_xyul
    abs_dif = np.abs(dif1[:,0]+1j*dif1[:,1]) # calculate distance
    min_ind = np.unravel_index(np.argmin(abs_dif),mainXY[0].shape) # find min row,col
        
    mask_out = np.zeros(mainXY[0].shape,dtype=bool)
    mask_out[min_ind[0]:new_shape[0]+min_ind[0],min_ind[1]:new_shape[1]+min_ind[1]]=True
    
    return mask_out
    
def recursive_applybydtype(obj,func,func_args=None,dtype=np.ndarray,skip_vals=[None]):
    run_apply=True
    obj_dict,obj_keys=dict_contents(obj)
    if obj_dict is None:
        run_apply=False
        
    if run_apply:
        for obj_key in obj_keys:
            if isinstance(obj_dict[obj_key],dtype) and obj_key not in skip_vals:
                try:
                    obj_dict[obj_key] = func(obj_dict[obj_key],**func_args)
                except:
                    pass
    #                print obj_key, obj_dict[obj_key].shape
            elif isinstance(obj_dict[obj_key],list) and obj_key not in skip_vals:
                temp_objs=[]
                for obj2 in obj_dict[obj_key]:
                    if isinstance(obj2,dtype):
                        try:
                            obj2 = func(obj2,**func_args)
                        except:
    #                        print obj_key, obj2.shape
                            pass
                    temp_objs.append(obj2)
                obj_dict[obj_key] = temp_objs

def dict_contents(obj):
    if isinstance(obj,dict):
        obj_dict = obj
        obj_keys = obj.keys()
    elif hasattr(obj,'__dict__'):
        obj_dict = obj.__dict__
        obj_keys = obj_dict.keys()
    else:
        obj_dict = None
        obj_keys = None
    return obj_dict,obj_keys
    

def run_cmd(cmd_list=None,cwd='./',async1=False,
            silent=False, pause=False, report=False,
            normal_msg=None,failed_words = ["fail", "error"]):
    '''Run command in DOS.
    
    This function will run the cmd_list using subprocess.Popen.  It
    communicates with the process' stdout asynchronously/syncrhonously and reports
    progress to the screen with timestamps
    
    Parameters
    ----------
    cmd_list : list
        List of [Executable name (with path, if necessary), other args].
    cwd : str
        current working directory, where inputs are stored. (default is the
        current working directory - './')
    silent : boolean
        Echo run information to screen (default is True).
    pause : boolean, optional
        Pause upon completion (default is False).
    report : boolean, optional
        Save stdout lines to a list (buff) which is returned
        by the method . (default is False).
    normal_msg : str
        Normal termination message used to determine if the
        run terminated normally. (default is None)
    failed_words: list
        List of words to search for that indicates problem with running
        command. (default is ["fail","error"])
    async : boolean
        asynchonously read model stdout and report with timestamps.  good for
        models that take long time to run.  not good for models that run
        really fast
    Returns
    -------
    (success, buff)
    success : boolean
    buff : list of lines of executable output (i.e., stdout)
    
    
    Source: after flopy.mbase.run_model'''
    
    # Load libraries
    from datetime import datetime
    import subprocess as sp
    import threading
    if sys.version_info > (3, 0):
        import queue as Queue
    else:
        import Queue
    
    success = False
    buff = []

    # simple function for the thread to target
    def q_output(output, q):
        for line in iter(output.readline, b''):
            q.put(line)
            # time.sleep(1)
            # output.close()

    proc = sp.Popen(cmd_list,
                    stdout=sp.PIPE, stderr=sp.STDOUT, cwd=cwd)

    # Run executable and handle all output at once
    if not async1:
        while True:
            line = proc.stdout.readline()
            c = line.decode('utf-8')
            if c != '' and c is not None:
                if normal_msg is not None and normal_msg in c.lower():
                    success = True
                c = c.rstrip('\r\n')
                if not silent:
                    print('{}'.format(c))
                if report == True:
                    buff.append(c)
            else:
                break
        return success, buff
    
    # ------- Run exe and collect output while still running -------------
    # some tricks for the async stdout reading
    q = Queue.Queue()
    thread = threading.Thread(target=q_output, args=(proc.stdout, q))
    thread.daemon = True
    thread.start()

    last = datetime.now()
    lastsec = 0.
    while True:
        try:
            line = q.get_nowait()
        except Queue.Empty:
            pass
        else:
            if line == '':
                break
            line = line.decode().lower().strip()
            if line != '':
                now = datetime.now()
                dt = now - last
                tsecs = dt.total_seconds() - lastsec
                line = "(elapsed:{0})-->{1}".format(tsecs, line)
                lastsec = tsecs + lastsec
                buff.append(line)
                if not silent:
                    print(line)
                for fword in failed_words:
                    if fword in line:
                        success = False
                        break
        if proc.poll() is not None:
            break
    proc.wait()
    thread.join(timeout=1)
    buff.extend(proc.stdout.readlines())
    proc.stdout.close()

    # Look for success message in output if one provided
    if normal_msg is not None:
        for line in buff:
            if normal_msg in line:
                print("success")
                success = True
                break

    if pause:
        input('Press Enter to continue...')
    return success, buff    
    
def match_keys(obj,in_token=None):
    obj_dict,obj_keys=dict_contents(obj)
    match_keys = [key for key in obj_keys if in_token in key]
    return match_keys
        
def sort_xy_cw(x=None,y=None):  
    '''Sort xy coordinates counterclockwise.
    
    Note: Only works when centroid is within the polygon specified by the input pts    
    
    Source: http://stackoverflow.com/a/13935419
    '''
    
    # Find centroids
    centroidx = np.mean(x)
    centroidy = np.mean(y)
    
    # Calculate angles from centroid
    angles_from_centroid = np.arctan2(y-centroidy,x-centroidx)
    
    sort_order = np.argsort(angles_from_centroid)
    
    return np.array(x)[sort_order],np.array(y)[sort_order]

def to_nD(in_val,to_dims):
    '''
    Check dtype of in_val and output to to_dims dimensions
    '''
    # Make sure to_dims is in shape format
    to_dims = np.ones(to_dims).shape
    
    ndims_out = len(to_dims)
    dim_ones = np.ones(ndims_out-1,dtype=np.int)
    
    if isinstance(in_val,(int,float)):
        out_val = in_val*np.ones(to_dims)
    else:
        in_val = np.array(in_val).squeeze()             
        dims_in = in_val.shape
        
        if dims_in==to_dims:
            out_val = in_val.copy() # already the desired shape
        
        elif dims_in==to_dims[1:]:
            # no layers
            out_val = np.tile(in_val,np.hstack((to_dims[0],dim_ones)))
        
        elif dims_in == to_dims[:1]:
            # only layer information, apply to nrow,ncols
            out_val = in_val.reshape(np.hstack((-1,dim_ones)))*np.ones(to_dims)
        elif dims_in in ((1,),()):
            # one entry
            out_val = in_val*np.ones(to_dims)
            
        else:
            if hasattr(in_val,'shape'):
                raise ValueError('Could not transform array {} to shape {}'.format(in_val.shape,to_dims))
            else:
                raise ValueError('Could not transform array {} to shape {}'.format(None,to_dims))
    return out_val
    
    
def unique_rows(a,sort=True,return_inverse=False):
    '''
    Find unique rows and return indexes of unique rows
    '''
    a = np.ascontiguousarray(a)
    unique_a,uind,uinv = np.unique(a.view([('', a.dtype)]*a.shape[1]),return_index=True,return_inverse=True)
    if sort:    
        uord = [(uind==utemp).nonzero()[0][0] for utemp in np.sort(uind)]
        outorder = uind[uord]
    else:
        outorder = uind
    if return_inverse:
        return unique_a,uind,uinv
    else:
        return outorder

def calc_dist(xy1,xy2,**kwargs):
    from scipy.spatial import distance
    
    dist_mat = distance.cdist(xy1,xy2,**kwargs)
    return dist_mat

def get_extent(XY):
    return [np.nanmin(XY[0]),np.nanmin(XY[1]),np.nanmax(XY[0]),np.nanmax(XY[1])]

def reduce_extent(in_extent,inx,iny, buffer_size=0,fill_mask_bool=True):
    '''
    Select portions of x,y cooridnates within in_extent
    
    in_extent = [minx,miny,maxx,maxy]
    
    '''
    if fill_mask_bool:
        inx = fill_mask(inx,np.inf)
        iny = fill_mask(iny,np.inf)
        
    inpts = (inx>=in_extent[0]-buffer_size) & (inx<=in_extent[2]+buffer_size) &\
            (iny>=in_extent[1]-buffer_size) & (iny<=in_extent[3]+buffer_size) 
    if len(inpts) > 0:
        return inx[inpts],iny[inpts],inpts
    else:
        return [],[],[]

        
def define_mask(cc_XY,active_indexes=None):
    bool_out = np.zeros_like(cc_XY[0],dtype=bool)
    bool_out[active_indexes[0],active_indexes[1]] = True
    return bool_out

def fill_mask(in_array,fill_value=np.nan):
    if hasattr(in_array,'mask'):
        out_array = np.ma.filled(in_array.copy(),fill_value)
    else:
        out_array = in_array.copy()
        
    if ~np.isnan(fill_value):
        out_array[np.isnan(out_array)]=fill_value
        
    return out_array

def to_model_coord(XYnodes,grid_transform,
            from_proj=ccrs.Geodetic(globe=default_globe),reverse=False):
    to_proj,xyshift,rot_angle = grid_transform
    if not reverse:
        # Transform to new coordinate system, from_proj to to_proj
        proj_xy = to_proj.transform_points(from_proj,XYnodes[0],XYnodes[1])
        
        # apply rotation and shift
        rotshift_x,rotshift_y = rot_shift([proj_xy[:,:,0],proj_xy[:,:,1]],
                                          xyshift=xyshift,rad_angle=rot_angle,
                                          reverse=reverse)
    if reverse:
        # apply rotation and shift
        rotshift_x,rotshift_y = rot_shift(XYnodes,
                                          xyshift=xyshift,rad_angle=rot_angle,
                                          reverse=reverse)
                                          
        # Transform to new coordinate system, to_proj to from_proj
        if (from_proj is not None) and (from_proj != to_proj):
            # Convert from "to_proj" to "from_proj"
            proj_xy = from_proj.transform_points(to_proj,rotshift_x,rotshift_y)                            
            if len(proj_xy.shape)==3:
                rotshift_x,rotshift_y = proj_xy[:,:,0].reshape(XYnodes[0].shape),proj_xy[:,:,1].reshape(XYnodes[0].shape)
            else:
                rotshift_x,rotshift_y = proj_xy[:,0].reshape(XYnodes[0].shape),proj_xy[:,1].reshape(XYnodes[0].shape)

    return rotshift_x,rotshift_y 

def modelcoord_transform(XY=None,mf_model=None,xyul_m=None, xyul=[0,0],rotation=0.,
                         proj_in=None,proj_out=None):
    '''Convert from model coordinates to other coordinate system.
    
    
    Note: xyul is not the xyshift from defining the domain, but the upper left 
            [x,y] pt of the model in a projected (e.g., UTM) coordinate system.
    
    '''
    if xyul_m is None:
        # Load node cooridnates
        y,x,z = mf_model.dis.get_node_coordinates()
        
        # Select upper left node coordinates
        xul_m,yul_m = x[0],y[0]
    else:
        xul_m,yul_m = xyul_m
    
    X_out = xrot(XY[0]-xul_m,XY[1]-yul_m,rotation)+xyul[0]
    Y_out = yrot(XY[0]-xul_m,XY[1]-yul_m,rotation)+xyul[1]

    if (proj_out is not None) and (proj_in is not None):
        # Project to new coordinate system
        if hasattr(proj_in,'proj4_init') and hasattr(proj_out,'proj4_init'):  #ccrs
            X_out,Y_out = ccrs_transform(XY=[X_out,Y_out],proj_in=proj_in,proj_out=proj_out)
        else:
            if hasattr(proj_in,'proj4_init'):
                proj_in = proj_in.proj4_init
            if hasattr(proj_out,'proj4_init'):
                proj_out = proj_out.proj4_init
                
            X_out,Y_out = osr_transform(XY=[X_out,Y_out],proj_in=proj_in,proj_out=proj_out)

    
    return X_out, Y_out    
    
def ccrs_transform(XY=None, proj_in=None, proj_out=None):
    X_out,Y_out = XY
    xy_temp = proj_out.transform_points(proj_in,X_out,Y_out)
    if len(xy_temp.shape)==3:
        X_out,Y_out = xy_temp[:,:,0].reshape(X_out.shape),xy_temp[:,:,1].reshape(X_out.shape)
    else:
        X_out,Y_out = xy_temp[:,0].reshape(X_out.shape),xy_temp[:,1].reshape(X_out.shape)
    return X_out,Y_out

def osr_transform(XY=None, proj_in=None, proj_out=None,param_opts=None):
    
    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)
    
    X_out,Y_out = XY
    xy_source = np.array([X_out,Y_out])
    
    
    if len(xy_source.shape)==3:
        shape=xy_source[0,:,:].shape
        size=xy_source[0,:,:].size
    else:

        shape = xy_source[:1,:].shape
        size = xy_source[:1,:].size

    # Define coordinate systems
    if hasattr(proj_in,'ExportToProj4'):
        src_proj = proj_in
    else:
        src_proj = osr.SpatialReference()
        if isinstance(proj_in,(int,float)):
            src_proj.ImportFromEPSG(proj_in)
        elif 'PROJCS' in proj_in or 'GEOGCS' in proj_in or '+' in proj_in:
            src_proj.ImportFromWkt(proj_in)
        else:
            src_proj.SetWellKnownGeogCS(proj_in) # assume geographic coordinates
    
    if hasattr(proj_out,'ExportToProj4'):
        dest_proj = proj_out
    else:
        dest_proj = osr.SpatialReference()            
        if isinstance(proj_out,(int,float)):
            dest_proj.ImportFromEPSG(proj_out)
        elif 'PROJCS' in proj_out or 'GEOGCS' in proj_out or '+' in proj_out:
            dest_proj.ImportFromWkt(proj_out)
        else:
            dest_proj.SetWellKnownGeogCS(proj_out) # assume geographic coordinates
    
    # Check for changes to coordinate systems
    if param_opts is not None:
        if 'proj_in' in param_opts.keys():
            if 'Params' in param_opts['proj_in'].keys():
                for ikey in param_opts['proj_in']['Params'].keys():
                    src_proj.SetProjParm(ikey,param_opts['proj_in']['Params'][ikey])
            if 'LinearUnits' in param_opts['proj_in'].keys():
                ikey = param_opts['proj_in']['LinearUnits'].keys()[0]
                src_proj.SetLinearUnits(ikey,param_opts['proj_in']['LinearUnits'][ikey])
            
        if 'proj_out' in param_opts.keys():
            if 'Params' in param_opts['proj_out'].keys():
                for ikey in param_opts['proj_out']['Params'].keys():
                    dest_proj.SetProjParm(ikey,param_opts['proj_out']['Params'][ikey])
            if 'LinearUnits' in param_opts['proj_out'].keys():
                ikey = param_opts['proj_out']['LinearUnits'].keys()[0]
                dest_proj.SetLinearUnits(ikey,param_opts['proj_out']['LinearUnits'][ikey])
    # Project to new coordinate system
    # the ct object takes and returns pairs of x,y, not 2d grids
    # so the the grid needs to be reshaped (flattened) and back.
    ct = osr.CoordinateTransformation(src_proj, dest_proj)
    xy_target = np.array(ct.TransformPoints(xy_source.reshape(2, size).T))
    X_out = xy_target[:,0].reshape(shape)
    Y_out = xy_target[:,1].reshape(shape)
    return X_out, Y_out
    

def rot_shift(XY,xyshift=[0,0],rad_angle=0, reverse=False):
    if reverse:
        # Shift and then rotate
        rotshift_x = xrot(XY[0]+xyshift[0],XY[1]+xyshift[1],-rad_angle)
        rotshift_y = yrot(XY[0]+xyshift[0],XY[1]+xyshift[1],-rad_angle)
    else:
        # Rotate and then shift
        rotshift_x = xrot(XY[0],XY[1],rad_angle)-xyshift[0]
        rotshift_y = yrot(XY[0],XY[1],rad_angle)-xyshift[1]
    
    return rotshift_x,rotshift_y

def write_grid_transform(gtransform,fname,model_info=None):
    coord_sys,xyshift,rot_angle = gtransform
    coord_sys_dict = coord_sys.proj4_params
    with open(fname,'w') as fout:
        # Header info
        fout.write('# Model: {}\n'.format(model_info))
        # Coordinate system information
        for key in coord_sys_dict.keys():
            fout.write('{},{}\n'.format(key,coord_sys_dict[key]))

        fout.write('x_shift,{}\n'.format(xyshift[0]))
        fout.write('y_shift,{}\n'.format(xyshift[1]))
        fout.write('rot_radians,{}\n'.format(rot_angle))

def read_grid_transform(fname):
    with open(fname,'r') as f_in:
        out_data={}
        for iline in f_in:
            if iline[0] in ['#']:
                continue
            
            iline = iline.strip('\n')
            pieces = iline.split(',')
            try:
                pieces[1] = int(pieces[1])
            except:
                try:
                    pieces[1] = float(pieces[1])
                except:
                    pass
            
            out_data.update([pieces])
    return out_data

def make_grid_transform(in_dict,s_hemi=False,from_ref=False, use_osr=True):
    if from_ref:
        if 'proj4' in in_dict.keys():
            # Convert from proj4 string to dict entries
            proj4_entries = in_dict['proj4'].split('+')
            proj4_entries_clean = [entry.strip().split('=') for entry in proj4_entries if len(entry)>0]
            proj4_entries_cull = [entry for entry in proj4_entries_clean if len(entry)>1]
            in_dict.update(proj4_entries_cull)
            in_dict['x_shift'],in_dict['y_shift'] = in_dict['xul'],in_dict['yul']
            in_dict['rot_radians'] = np.deg2rad(in_dict['rotation']) # new flopy is +angle is counterclockwise
    if use_osr:
        proj_in = in_dict['proj4']
        proj_out = osr.SpatialReference()
        if isinstance(proj_in,(float,int)):
            proj_out.ImportFromEPSG(proj_in)
        elif '+' in proj_in:
            proj_out.ImportFromProj4(proj_in)
        elif 'PROJCS' in proj_in or 'GEOGCS' in proj_in:
            proj_out.ImportFromWkt(proj_in)
        elif hasattr(proj_in,'ImportFromWkt'):
            proj_out = proj_in # already an osr sr
        else:
            # Assume outproj is geographic sr
            proj_out.SetWellKnownGeogCS(proj_in) 

    else:        
        globe = ccrs.Globe(datum=in_dict['datum'],ellipse=in_dict['ellps'])
        if in_dict['proj'] in ['utm']:
            proj_out = ccrs.UTM(in_dict['zone'],globe=globe,southern_hemisphere=s_hemi)
        elif in_dict['proj'] in ['aea']:
            proj_out = ccrs.AlbersEqualArea(float(in_dict['lon_0']),float(in_dict['lat_0']),
                                            standard_parallels=(float(in_dict['lat_1']),float(in_dict['lat_2'])),
                                            globe=ccrs.Globe(datum=in_dict['datum'],ellipse=in_dict['ellps']))
    grid_transform = [proj_out,[in_dict['x_shift'],in_dict['y_shift']],in_dict['rot_radians']]    
    return grid_transform

def write_model_ref(model_info_dict=None):
    '''Export usgs.model.reference file.
    '''
    fname = os.path.join(model_info_dict['model_ws'],r'usgs.model.reference')
    with open(fname,'w') as fout:
        # Header info
        fout.write('# Model reference data for model {}\n'.format(model_info_dict['model_name']))
        fout.write('xul {0:.5f}\n'.format(model_info_dict['xul']))
        fout.write('yul {0:.5f}\n'.format(model_info_dict['yul']))
        fout.write('rotation {}\n'.format(np.rad2deg(model_info_dict['rotation'])))
        
        # Write length units
        if isinstance(model_info_dict['length_units'],(float,int)):
            fout.write('length_units {}\n'.format(length_unit_dict[int(model_info_dict['length_units'])]))
        else:
            fout.write('length_units {}\n'.format(model_info_dict['length_units']))
        
        # Write time units
        if isinstance(model_info_dict['time_units'],(float,int)):
            fout.write('time_units {}\n'.format(time_unit_dict[int(model_info_dict['time_units'])]))
        else:
            fout.write('time_units {}\n'.format(model_info_dict['time_units'])) 
        
        fout.write('start_date {}\n'.format(model_info_dict['start_date']))
        fout.write('start_time {}\n'.format(model_info_dict['start_time']))
        
        if model_info_dict['model_type'].lower() in ['nwt']:
            fout.write('model {}\n'.format('MODFLOW-NWT_1.0.9'))
            
        elif model_info_dict['model_type'].lower() in ['mf2005','gmg','pcg']:
            fout.write('model {}\n'.format('MF2005.1_11'))
            
        
        if model_info_dict['proj_type'] in ['espg']:
            fout.write('espg {}\n'.format(model_info_dict['proj']))
            fout.write('# epsg code')
        elif model_info_dict['proj_type'] in ['proj4']:
            fout.write('proj4 {}\n'.format(model_info_dict['proj']))
            fout.write('# proj4 string')
        
        fout.close()
    return fname

def read_model_ref(fname,comment_symbols=['#']):
    inum=0
    with open(fname,'r') as f_in:
        out_data={}
        for iline in f_in:
            if iline.strip()[0] in comment_symbols:
                pieces = ['comment_{}'.format(inum),iline[1:].strip('\n')]
                inum+=1
                out_data.update([pieces])
                continue
            
            iline = iline.strip('\n')
            pieces = iline.split(' ')
            if len(pieces)>2:
                pieces = [pieces[0],' '.join(pieces[1:])]
            try:
                pieces[1] = int(pieces[1])
            except:
                try:
                    pieces[1] = float(pieces[1])
                except:
                    pass
        
            out_data.update([pieces])
    return out_data

def quick_plot(mat,ncols=3,**kwargs):
    
    if len(mat.shape)==3 and mat.shape[0]>1:
        nplots = mat.shape[0]
        if nplots <=3:
            nrows,ncols = 1,nplots
        else:
            nrows = np.int(np.ceil(nplots/float(ncols))) 
        fig,ax  = plt.subplots(nrows,ncols)
        ax = ax.ravel()
        for ilay in np.arange(mat.shape[0]):
            im1=ax[ilay].imshow(np.ma.masked_invalid(mat[ilay,:,:]),
                                        interpolation='none',**kwargs)
            plt.colorbar(im1,ax=ax[ilay],orientation='horizontal')
            ax[ilay].set_title('Layer {}'.format(ilay))
    else:
        fig,ax  = plt.subplots()
        im1=ax.imshow(np.ma.masked_invalid(np.squeeze(mat)),interpolation='none',
                      **kwargs)
        plt.colorbar(im1,ax=ax,orientation='horizontal')

    plt.show()
    
    return ax

def read_txt(fname=None):
    out_list=[]
    with open(fname,'r') as fin:
        for iline in fin:
            out_list.append(iline.strip('\n'))
    return out_list

def save_txt(fname=None,vals=None):
    with open(fname,'w') as fout:
        for ival in vals:
            fout.write(''.join([ival,'\n']))
            
def update_txt(fname=None,new_val=None):
    if os.path.isfile(fname):
        list1 = read_txt(fname=fname)
    else:
        list1=[]
        
    if new_val in list1:
        return True,list1
    else:
        list1.append(new_val)
        save_txt(fname,list1)
        return False,list1
    

# File utils
        
def copy_rename_folder(in_folder,out_folder,fbase,exclude_list=['chk'],overwrite=True):
    ''' Copy contents from one folder to another and rename
    '''
    copy_tree(in_folder,out_folder)
    fnames = glob.glob(os.path.join(out_folder,'*.*'))
    for fname in fnames:
        ext_temp = os.path.basename(fname).split('.')[-1]
        if ext_temp not in exclude_list:
            newfname = os.path.join(out_folder,'{}.{}'.format(fbase,ext_temp))
            if os.path.isfile(newfname) and overwrite\
                  and newfname not in [os.path.join(out_folder,fname)]:
                os.remove(newfname)
                os.rename(os.path.join(out_folder,fname),newfname)
            elif os.path.isfile(newfname) and not overwrite:
                continue
            elif newfname in [os.path.join(out_folder,fname)] and not overwrite:
                continue
            else:
                os.rename(os.path.join(out_folder,fname),newfname)
    