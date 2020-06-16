# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:53:40 2016

@author: kbefus
"""
from __future__ import print_function
import os,sys,shutil
from osgeo import osr, ogr
import numpy as np
import shapefile
from shapely.geometry import Polygon,MultiPoint,LineString,MultiPolygon#,MultiLineString
from shapely.geometry.polygon import orient
from shapely.geometry import box as shpbox
from shapely.prepared import prep
from shapely.ops import unary_union,cascaded_union
from shapely.affinity import rotate
import cartopy.crs as ccrs
from shapely import speedups
#import re
import pandas as pd
import geopandas as gpd


from . import cgw_general_utils as cgu
from . import cgw_raster_utils as cru
gdal = cru.gdal

speedups.enable()

default_globe = cgu.default_globe

def buffer_invalid(poly_in,buffer_size=0.):
    if not poly_in.is_valid:
        poly_in = poly_in.buffer(buffer_size)
    return poly_in

def sort_xy_cw(x=None,y=None):  
    '''Sort xy coordinates counterclockwise.
    
    Note: Only works when centroid is located within the polygon
            defined by the x,y coordinates.
            
    Source: http://stackoverflow.com/a/13935419
    '''
    
    # Find centroids
    centroidx = np.mean(x)
    centroidy = np.mean(y)
    
    # Calculate angles from centroid
    angles_from_centroid = np.arctan2(y-centroidy,x-centroidx)
    
    sort_order = np.argsort(angles_from_centroid)
    
    return x[sort_order],y[sort_order]

def project_xy(xy,cproj_in,cproj_out):
    '''
    Project x,y pairs from cartopy projection cproj_in to projection cproj_out
    '''
    # extent1 = shapefile bounding box [xll,yll,xur,yur] 
    x,y=zip(*xy)
    xyp = cproj_out.transform_points(cproj_in,np.array(x),np.array(y))
    return xyp[:,:2]

def define_UTM_zone(input_extent,use_negative=True):
    ''' Find UTM zone of domain center.
    
    Inputs
    --------
    input_extent: list,np.ndarray
        input bounds of feature for defining UTM zone, [xmin,ymin,xmax,ymax]
    '''
    meanx = np.mean(input_extent[::2])
    if meanx < 0. and use_negative:
        utm_edges=np.arange(-180.,186.,6.)
    else:
        utm_edges=np.arange(0.,366.,6.)
    utm_zones=np.arange(1,utm_edges.shape[0])
    west_boundary = utm_edges[:-1]
    east_boundary = utm_edges[1:]
    UTM_ind = ((meanx>=west_boundary) & (meanx<=east_boundary)).nonzero()[0][0]
    output_zone = utm_zones[UTM_ind]
    return output_zone

def proj_polys_old(polys=None,proj_kwargs=None):
    '''Project list of polygons to new projection.
    
    reverse_bool sets order of transformations:
        if True, then rotation and shift before reprojection (i.e., model to utm)
        if False, rotation and shift after reprojection (i.e., utm to model)
    '''    
    projected_polys = []             
    for poly in polys:
        if hasattr(poly,'area'):
            if poly.geom_type in ['Polygon']:
                shp_xy = np.array(poly.exterior.xy).T
                
                
                shp_xy_proj = cgu.modelcoord_transform(XY=[shp_xy[:,0],shp_xy[:,1]],**proj_kwargs)
                
                projected_polys.append(Polygon(np.array(shp_xy_proj).T))
            elif poly.geom_type in ['MultiPolygon']:
                # Loop through polygon parts
                poly_parts = []
                for ipoly in poly:
                    shp_xy = np.array(ipoly.exterior.xy).T
                    shp_xy_proj = cgu.modelcoord_transform(XY=[shp_xy[:,0],shp_xy[:,1]],**proj_kwargs)
                    poly_parts.append(Polygon(np.array(shp_xy_proj).T))
                    
                projected_polys.append(MultiPolygon(poly_parts))
        else:
            projected_polys.append(poly) # if not a shapely shape, then return as is
            
    return projected_polys

def proj_polys(polys=None,proj_in=None,proj_out=None,proj_kwargs=None,model_coords=False):
    '''Project list of polygons to new projection.
    
    reverse_bool sets order of transformations:
        if True, then rotation and shift before reprojection (i.e., model to utm)
        if False, rotation and shift after reprojection (i.e., utm to model)
    '''               

    if proj_in is None:
        proj_in = proj_kwargs['proj_in']
    if proj_out is None:
        proj_out = proj_kwargs['proj_out']

    # Load all xy pairs into one matrix, keep track of indexes
    new_shape_inds = []
    new_shape_internal_inds = []
    all_xy = []
    all_xy_internal=[]
    istart,iend = 0,0
    istart_internal,iend_internal = 0,0
    
    projected_polys = []
    for poly in polys:
        if hasattr(poly,'area'):
            if poly.geom_type in ['Polygon']:
                shp_xy = np.array(poly.exterior.xy).T
                all_xy.append(shp_xy)
                iend += int(shp_xy.shape[0])
                new_shape_inds.append([istart,iend])
                istart += int(shp_xy.shape[0])
                
                # For internal features separately
                if len(poly.interiors) > 0:
                    int_xy = []
                    int_inds = []
                    for poly1 in poly.interiors:
                        shp_int_xy = np.array(poly1.xy).T.reshape((-1,2))
                        int_xy.append(shp_int_xy)
                        iend_internal += int(shp_int_xy.shape[0])
                        int_inds.append([istart_internal,iend_internal])
                        istart_internal += int(shp_int_xy.shape[0])
                    
                    new_shape_internal_inds.append(int_inds)    
                    all_xy_internal.append(np.vstack(int_xy))
                else:
                    new_shape_internal_inds.append([[None,None]])    

                
            elif poly.geom_type in ['MultiPolygon']:
                    # Loop through polygon parts
                    multi_inds = []
                    poly_parts = []
                    multi_shape_internal_inds = []
                    multi_xy_internal = []
                    for ipoly in poly:
                        shp_xy_temp = np.array(ipoly.exterior.xy).T
                        poly_parts.append(shp_xy_temp)
                        iend += int(shp_xy_temp.shape[0])
                        multi_inds.append([istart,iend])
                        istart += int(shp_xy_temp.shape[0])
                        
                        # For internal features separately
                        if len(ipoly.interiors) > 0:
                            int_xy = []
                            int_inds = []
                            for poly1 in ipoly.interiors:
                                shp_int_xy = np.array(poly1.xy).T.reshape((-1,2))
                                int_xy.append(shp_int_xy)
                                iend_internal += int(shp_int_xy.shape[0])
                                int_inds.append([istart_internal,iend_internal])
                                istart_internal += int(shp_int_xy.shape[0])
                            
                            multi_shape_internal_inds.append(int_inds)    
                            multi_xy_internal.append(np.vstack(int_xy))
                        else:
                            multi_shape_internal_inds.append([[None,None]])
                    
                    new_shape_internal_inds.append(multi_shape_internal_inds)
                    if len(multi_xy_internal)>0:
                        all_xy_internal.append(np.vstack(multi_xy_internal))
                    
                    shp_xy = np.vstack(poly_parts)    
                    all_xy.append(shp_xy)
                    
                    new_shape_inds.append(multi_inds)
            
    # Project data once
    all_xy_array = np.vstack(all_xy)
    if model_coords:
        shp_xy_proj = np.array(cgu.modelcoord_transform(XY=[all_xy_array[:,0],all_xy_array[:,1]],**proj_kwargs)).T
    else:
        shp_xy_proj = np.array(cgu.osr_transform(XY=[all_xy_array[:,0],all_xy_array[:,1]],
                                                 proj_in=proj_in,
                                                 proj_out=proj_out)).T
        
    shp_xy_proj = shp_xy_proj.squeeze()
    
    if len(all_xy_internal) > 0:
        all_xy_internal_array = np.vstack(all_xy_internal)
        if model_coords:
            shp_xy_proj_internal = np.array(cgu.modelcoord_transform(XY=[all_xy_internal_array[:,0],all_xy_internal_array[:,1]],**proj_kwargs)).T
        else:
            shp_xy_proj_internal = np.array(cgu.osr_transform(XY=[all_xy_internal_array[:,0],all_xy_internal_array[:,1]],
                                                 proj_in=proj_in,
                                                 proj_out=proj_out)).T
    # Parse xy pairs back into polygons
     
    for ipoly_startend,iinternal_se in zip(new_shape_inds,new_shape_internal_inds):
        if isinstance(ipoly_startend[0],int) or isinstance(ipoly_startend[0],float):# float changed from long, might have issues with python 2.x
            # Only one shape to make
            istart,iend = ipoly_startend
            # remake internal features
            internal_polys = []
            for istart_int,iend_int in iinternal_se:
                if istart_int is not None:
#                    print(shp_xy_proj_internal[istart_int:iend_int,:])
                    internal_polys.append(Polygon(shp_xy_proj_internal[istart_int:iend_int,:].squeeze()))
            internal_polys = cascaded_union(internal_polys)    
            if hasattr(internal_polys,'geom_type'):    
                projected_polys.append(Polygon(shp_xy_proj[istart:iend,:]).difference(internal_polys))
            else:
                projected_polys.append(Polygon(shp_xy_proj[istart:iend,:]))
        else:
            # Multiple shapes
            poly_parts = []
#                print ipoly_startend
            for (istart,iend),istartend_ints in zip(ipoly_startend,iinternal_se):
                # remake internal features
                internal_polys = []
                for istart_int,iend_int in istartend_ints:
                    if istart_int is not None:
                        internal_polys.append(Polygon(shp_xy_proj_internal[istart_int:iend_int,:].squeeze()))
                internal_polys = cascaded_union(internal_polys)
                if hasattr(internal_polys,'geom_type'): 
                    poly_parts.append(Polygon(shp_xy_proj[istart:iend,:]).difference(internal_polys))
                else:
                    poly_parts.append(Polygon(shp_xy_proj[istart:iend,:]))
                
            projected_polys.append(MultiPolygon(poly_parts))
    
    return projected_polys    
    
    
def shp_to_grid(shp,cell_spacing=None,pts_to_decimate=1,
                in_proj = None,out_proj=None,
                active_feature = None):
    '''Construct regular grid from polygon feature.
    
    Calculates smallest bounding rectangle for active_feature in shapefile fname.
    Converts shapefile to UTM prior to rotating and shifting. Output grid is
    in a Modflow-like coordinate system with the origin at row,col=0,0.
    
    in_proj and out_proj should be EPSG, Proj4, or WKT
    '''
        
    if ~hasattr(shp,'geom_type'):
        # shp is not a shapely feature
        shp,npolys = shp_to_polygon(shp)
        if len(shp) > 1:
            if active_feature is None:
                areas = [Polygon(shape.points).area for shape in shp]
                feature_index = np.argmax(areas) # use feature with largest area
            else:
                feature_index = active_feature # use chosen feature
                
            shp = shp[feature_index]
        else:
            shp=shp[0]
            
#    extent = shp.bounds
#    UTM_zone = define_UTM_zone(extent)
#    cproj_out = ccrs.UTM(UTM_zone,globe=globe_type)
#    shp_xy = project_xy(np.array(shp.exterior.xy).T,in_proj,cproj_out)
#    shp_pts = np.array(shp_xy)
    shp_pts = np.array(shp.exterior.xy).T
    if in_proj is not None and out_proj is not None:
        if in_proj != out_proj:
            # Reproject shp coordinates to out_proj
            proj_dict = {'xy_source':shp_pts,
                         'inproj':in_proj,'outproj':out_proj}
            shp_pts = cru.projectXY(**proj_dict)
    elif in_proj is not None:
        out_proj = in_proj

    shp_pts_dec = shp_pts[::pts_to_decimate]
    
    # Calculate convex hull of feauture
    shp_dec = Polygon(list(zip(shp_pts_dec[:,0],shp_pts_dec[:,1])))
    shp_hull = shp_dec.convex_hull
    shp_hull_xy = np.array(shp_hull.exterior.coords.xy).T
    ncoords = shp_hull_xy.shape[0]
    
    rect_area = lambda bounds: (bounds[2]-bounds[0])*(bounds[3]-bounds[1])
    
    out_bounds = []
    angle_area_array = []
    rot_shps = []
    for icoord in list(range(ncoords-1)):
        dx,dy = np.diff(shp_hull_xy[icoord:icoord+2,:],axis=0)[0]
        angle_theta = np.arctan2(dy,dx) # positive angle = counter-clockwise
        rot_shp = rotate(shp_hull,angle_theta,origin=(0.,0.),use_radians=True)
        out_bounds.append(rot_shp.bounds)
        angle_area_array.append([angle_theta,rect_area(rot_shp.bounds)])
        rot_shps.append(rot_shp)
        
    angle_area_array = np.array(angle_area_array)
    min_area_index = np.argmin(angle_area_array[:,1])
    min_angle = angle_area_array[min_area_index,0]    # radians
    
    # Constrain minimum angle in 1st and 4th quandrants to keep from flipping unnecessarily
    if min_angle < np.pi and min_angle > np.pi/2.:
        min_angle = np.pi+min_angle # 2nd quadrant to 4th
    elif min_angle >= np.pi and min_angle <= 3.*np.pi/2.:
        min_angle = min_angle-np.pi # 3rd to 1st
    
    shp_dec_rot = rotate(shp_dec,min_angle,origin=(0.,0.),use_radians=True)
    shp_dec_rot_xy = np.array(shp_dec_rot.exterior.coords.xy).T
    domain_extent = shp_dec_rot.bounds
    
    bounds1=np.array(domain_extent).reshape((2,2))
    xshift,yshift = bounds1[0,:] # set bottom left corner to 0,0
    rect_bounds = np.vstack((bounds1,np.hstack((bounds1[:,0],np.roll(bounds1[:,1],1,axis=0))).reshape((2,2)).T))
    rect_bounds = rect_bounds[np.argsort(rect_bounds[:,0]),:]

    # Apply x and y shifts to translate domain to origin    
    rect_bounds[:,0] = rect_bounds[:,0]-xshift
    rect_bounds[:,1] = rect_bounds[:,1]-yshift    
    shp_dec_rot_xy[:,0]=shp_dec_rot_xy[:,0]-xshift
    shp_dec_rot_xy[:,1]=shp_dec_rot_xy[:,1]-yshift
    shp_dec_rot_trans = Polygon(shp_dec_rot_xy.tolist())
    
    max_x,max_y = np.max(rect_bounds, axis=0)
    x_vect,y_vect = np.arange(-cell_spacing, max_x + 2.*cell_spacing, cell_spacing), np.arange(-cell_spacing, max_y + 2.*cell_spacing, cell_spacing)
    
    # Node locations
    X_nodes,Y_nodes = np.meshgrid(x_vect, y_vect[::-1]) # y decreases down increasing rows, x increases by columns to right
    # Return X,Y of nodes, rotated polygon, projection,rotation, and translation info
    return [X_nodes,Y_nodes],shp_dec_rot_trans,[out_proj,[xshift,yshift],min_angle]

def shp_to_polygon(in_polygon):
    '''Convert a shapefile-like item to a shapely polygon.
    
    Inputs
    --------
    
    in_polygon: str, pyshp obj, or shapely obj
        input polygon that can be one of the above formats that will be
        converted to a list of shapely polygons.
        
    Returns
    --------
    
    out_polygon: list
        list of shapely polygons.
        
    nshapes: int
        Number of shapes in list.
    
    '''
    if isinstance(in_polygon,str):
        shp1 = shapefile.Reader(in_polygon)
        in_polygon=[]
        for shape in shp1.shapes():
            parts = shape.parts
            points = shape.points
            if len(parts) > 2:
                for part1,part2 in zip(parts[:-1],parts[1:]):
                    in_polygon.append(Polygon(points[part1:part2]))
                else:
                    in_polygon.append(Polygon(points[part2:]))
            else:
                in_polygon.append(Polygon(points))
#        in_polygon = [Polygon(shape.points) for shape in shp1.shapes() #old way, can't handle holes
    
    if hasattr(in_polygon,'bbox'):
        if len(in_polygon.parts)>1:
            pts = in_polygon.points
            parts = np.hstack([in_polygon.parts,len(pts)])
            out_polygon = [Polygon(pts[parts[ipt]:parts[ipt+1]]) for ipt in range(len(parts)-1)]
        else:
            out_polygon = [Polygon(in_polygon.points)]
            
        nshapes = len(out_polygon) 
    else:
        if isinstance(in_polygon,list):
            nshapes = len(in_polygon)
            out_polygon=in_polygon
        elif isinstance(in_polygon,np.ndarray):
            nshapes = len(in_polygon)
            out_polygon=in_polygon
        else:
            if in_polygon.geom_type is 'MultiPolygon':
                out_polygon = in_polygon.geoms
                nshapes = len(out_polygon)
            else:
                out_polygon = [in_polygon] # put single shape into a list
                nshapes = len(out_polygon)
    return out_polygon,nshapes

def pt_in_shp(in_polygon,XYpts,grid_buffer=None):
    speedups.enable()
    in_polygon,nshapes=shp_to_polygon(in_polygon)
      
    if hasattr(XYpts,'geom_type'):
        pts = XYpts
    elif isinstance(XYpts,np.ndarray):
        pts = MultiPoint(XYpts)
    else:
        x,y = XYpts
        pts = MultiPoint(list(zip(x.ravel(),y.ravel())))
    
    if grid_buffer is None:
        iout=[]
        for ipoly,loop_poly in enumerate(in_polygon):
            prepared_polygon = prep(loop_poly)
            iout.extend([ishp for ishp,pt in enumerate(pts) if prepared_polygon.contains(pt)])
    else:
        try:
            new_polys = [shpbox(pt.xy[0]-grid_buffer,pt.xy[1]-grid_buffer,pt.xy[0]+grid_buffer,pt.xy[1]+grid_buffer) for pt in pts]
        except:
            new_polys = [shpbox(pt.xy[0][0]-grid_buffer,pt.xy[1][0]-grid_buffer,pt.xy[0][0]+grid_buffer,pt.xy[1][0]+grid_buffer) for pt in pts]             
        iout=[]
        for ipoly,loop_poly in enumerate(in_polygon):
            prepared_polygon = prep(loop_poly)
            iout.extend([ishp for ishp,poly in enumerate(new_polys) if prepared_polygon.intersects(poly)])
    return iout

#def pt_in_shp_slow(XY=None,in_polygon=None,cell_spacing=None):
#    '''
#    
#    Source: after http://stackoverflow.com/a/14804366
#    '''
#    from rtree import index
#    if isinstance(cell_spacing,int) or isinstance(cell_spacing,float):
#        cell_spacing = [cell_spacing,cell_spacing]
#    
#    idx = index.Index()
#    XYnew = zip(XY[0].ravel(),XY[1].ravel()) 
#    bounds=[[x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.] for x,y in XYnew]
#    # Populate R-tree index with bounds of grid cells
#    for pos, cell in enumerate(bounds):
#        # assuming cell is a shapely object
#        idx.insert(pos, cell)
#    
#    iout = [pos for pos in idx.intersection(in_polygon.bounds)]
#    return iout
    
    
def gridpts_in_shp(in_polygon,XYpts, print_no_match = False, reduce_extent=True):
    '''
    Find row and column index for grid cell centers within in_polygon
    '''
    speedups.enable()
    in_polygon,nshapes=shp_to_polygon(in_polygon)
            
    all_y_cols,all_x_cols = [],[]
    collect_polys = []
    grid_centers_x,grid_centers_y = XYpts
    if len(in_polygon)>0:
        for ipoly,loop_poly in enumerate(in_polygon):
            poly_extent = loop_poly.bounds# minx,miny,maxx,maxy 
            if reduce_extent:
                pt_spacing = np.abs(np.diff(grid_centers_x,axis=1).mean()+1j*np.diff(grid_centers_y,axis=0).mean())/2. # mean diagonal/2 for cells
                gx,gy,inpts = cgu.reduce_extent(poly_extent,grid_centers_x,grid_centers_y,
                                          buffer_size = pt_spacing)
            else:
                gx,gy = grid_centers_x,grid_centers_y
                inpts = (grid_centers_x == grid_centers_x) & (grid_centers_y == grid_centers_y)
            
            inpts_rows_cols = np.array(inpts.nonzero()).T
            pts = MultiPoint(list(zip(gx,gy)))
            prepared_polygon = prep(loop_poly)
            iout=[ishp for ishp,pt in enumerate(pts) if prepared_polygon.contains(pt)]
            
            if len(iout)>0:
                # Convert back to coordinate grid indexes
                loop_y_rows,loop_x_cols = inpts_rows_cols[iout,:].T
                all_y_cols.extend(loop_y_rows)
                all_x_cols.extend(loop_x_cols)
                collect_polys.extend((np.float(ipoly)*np.ones(len(iout))).tolist())
        
        if len(all_x_cols)>0:
            if nshapes > 1:
                out_unq_ind_order = cgu.unique_rows(np.c_[all_y_cols,all_x_cols])
                out_rows = np.array(all_y_cols)[out_unq_ind_order]
                out_cols = np.array(all_x_cols)[out_unq_ind_order]
                out_poly = np.array(collect_polys)[out_unq_ind_order]
                return out_rows.ravel(),out_cols.ravel(),out_poly.ravel()
            else:
                return loop_y_rows.ravel(),loop_x_cols.ravel(),np.array(collect_polys).ravel()
        else:
            if print_no_match:
                print('No matches found, all points outside domain:')
                print('polygon extent: {}'.format(np.around(poly_extent,decimals=1)))
                print('points extent: {}'.format(np.around([grid_centers_x.min(),grid_centers_y.min(),grid_centers_x.max(),grid_centers_y.max()],decimals=1)))
            return [],[],[]
    else:
        return [],[],[]

def min_dist(in_polygon,XYpts,exterior=True):
    '''Find minimum distance between feature and xy points.
    '''
    speedups.enable()
    in_polygon,nshapes=shp_to_polygon(in_polygon)
    if hasattr(XYpts,'geom_type'):
        pts = XYpts
    else:
        x,y = XYpts
        pts = MultiPoint(list(zip(x,y)))
    iout=[]
    for ipoly,loop_poly in enumerate(in_polygon):
        if exterior:
            iout.append([loop_poly.exterior.distance(pt) for pt in pts])
        else:
            iout.append([loop_poly.distance(pt) for pt in pts])
    min_out = np.min(iout,axis=0)
    return min_out

def shp_to_pd(shp_fname,save_shps=False,shp_col = 'shape',
              save_polys=False,poly_col='poly'):
    '''Load shapefile to pandas dataframe.
    '''
    in_shp = shapefile.Reader(shp_fname)
    shp_df = pd.DataFrame(np.array(in_shp.records()),columns=np.array(in_shp.fields[1:])[:,0])
    shp_fieldtypes = np.array(in_shp.fields[1:])[:,1:]
    if save_shps:
        shp_df[shp_col] = in_shp.shapes()
        if save_polys:
            try: # assume all shapes are polygons
                out_polys = []
                for ishp in shp_df[shp_col].values.tolist():
                    if len(ishp.parts)==1:
                        out_polys.append(Polygon(ishp.points))
                    else:
                        # Multiple parts to the polygon, need to make a multipolygon
                        shp_temp = []
                        shp_pts = ishp.points
                        shp_parts = ishp.parts
                        for ipart in list(range(len(shp_parts))):
                            if ipart<len(shp_parts)-1:
                                shp_temp.append(Polygon(shp_pts[shp_parts[ipart]:shp_parts[ipart+1]]))
                            else:
                                shp_temp.append(Polygon(shp_pts[shp_parts[ipart]:])) # remaining points are part of the last part
                        
                        out_polys.append(MultiPolygon(shp_temp))
                    
                shp_df[poly_col] = out_polys
            except:
                poly_list = []
                for ishp in in_shp.shapes():
                    if ishp.shapeType == 1:
                        poly_list.append(MultiPoint(ishp.points))
                    elif ishp.shapeType == 2:
                        poly_list.append(LineString(ishp.points))
                    else:
                        poly_list.append(Polygon(ishp.points))
                shp_df[poly_col] = poly_list
                        
    for col_name,ftype in zip(shp_df.columns,shp_fieldtypes):
        if ftype[0]=='N':
            if int(ftype[-1]) != 0: # float column
                shp_df[col_name] = shp_df[col_name].astype(np.float)
            else: # integer column
                try:
                    shp_df[col_name] = shp_df[col_name].astype(np.int)
                except:
                    shp_df[col_name] = pd.to_numeric(shp_df[col_name])
    
    return shp_df

def nodes_to_cc(XYnodes,grid_transform,globe_type = default_globe):
    '''
    
    Returns:
        Variables:  [cc_x,cc_y], [cc_x_proj,cc_y_proj],     [cc_x_latlong, cc_y_latlong]
        Dimensions: [[ny-1,nx-1],[ny-1,nx-1]]      , [[ny-1,nx-1],[ny-1,nx-1]], [[ny-1,nx-1],[ny-1,nx-1]]
        where ny,nx = Xcorners.shape
        
    Dependencies:
        Packages: Numpy, Cartopy
        Functions: xrot, yrot
    '''
    Xcorners,Ycorners = XYnodes
    from_proj,xyshift,rot_angle = grid_transform
    # Calculate cell centers from nodes (grid corners)
    cc_x = Xcorners[:-1,:-1]+np.diff(Xcorners[:-1,:],axis=1)/2.
    cc_y = Ycorners[:-1,:-1]+np.diff(Ycorners[:,:-1],axis=0)/2.
    
    # Unrotate cell_centers to get cell centers in projected coordinate system
    cc_x_proj = cgu.xrot(cc_x+xyshift[0],cc_y+xyshift[1],-rot_angle)
    cc_y_proj = cgu.yrot(cc_x+xyshift[0],cc_y+xyshift[1],-rot_angle)
    
    # Unproject cell centers to geographic coordinate system (NAD83)
    if from_proj is not None:
        cc_x_latlong,cc_y_latlong = cru.projectXY(np.array([cc_x_proj,cc_y_proj]),from_proj) # to NAD83
#        cc_in_latlong = ccrs.Geodetic(globe=globe_type).transform_points(from_proj,cc_x_proj,cc_y_proj)
#        cc_x_latlong, cc_y_latlong = cc_in_latlong[:,:,0], cc_in_latlong[:,:,1]
    else:
        cc_x_latlong, cc_y_latlong = [],[]
    return [cc_x,cc_y],[cc_x_proj,cc_y_proj],[cc_x_latlong, cc_y_latlong]

def df_field_dict(in_df,n_dig=19,n_dec=4,ntxt=100,
                  col_names=None,col_types=None):
    '''Create pyshp field dictionary from pandas dataframe.
    '''
    if col_names is None:
        col_names = in_df.columns.values
    if col_types is None:
        col_types = in_df.dtypes.values 

    out_field_dict={}
    for i,(col_name,col_type) in enumerate(zip(col_names,col_types)):
        if col_type in ['int','int32','int64',np.int,np.int64,np.int32]:
            out_field_dict.update({col_name:{'fieldType':"N",'size':n_dig,'decimal':0}})
        elif col_type in ['float','double','float32','float64',np.float,np.float64,np.float32]:
            out_field_dict.update({col_name:{'fieldType':"N",'size':n_dig,'decimal':n_dec}})
        else:
            out_field_dict.update({col_name:{'fieldType':"C",'size':ntxt,'decimal':0}})
    return out_field_dict

def write_model_bound_shp(xycorners,data=None,out_fname=None,
                          col_name_order=None):
    
    temp_df = pd.DataFrame(data,columns=col_name_order)
    gdf = gpd.GeoDataFrame(temp_df,geometry=[Polygon(xycorners)])
    gdf.to_file(out_fname)
    
#    poly_out = Polygon(xycorners)
#    write_shp(polys=[poly_out],data=[data],out_fname=out_fname,field_dict=field_dict,
#              col_name_order=col_name_order,write_prj_file=True)
    
            
def write_shp(polys=None,data=None,out_fname=None,
              field_dict=None,inproj=None,write_prj_file=False,
              col_name_order=None):
    '''Write shapefile.
    '''
    w = shapefile.Writer()
    
    # Add new fields
    if (col_name_order is None) and (field_dict is not None):
        for f in field_dict:
            w.field(f, **field_dict[f])
    elif col_name_order is not None:
        for f in col_name_order:
            w.field(f, **field_dict[f])

    for i,shp in enumerate(polys):
        if not hasattr(shp,'geom_type'):
            raise ValueError('Input features are not shapley geometries')
        elif shp.geom_type == 'Polygon':
            xtemp,ytemp = shp.exterior.xy
            main_poly = np.array(list(zip(xtemp,ytemp))).tolist()
            out_parts = []
            out_parts.append(main_poly)
            # check for interior holes
            if len(shp.interiors)>0:
                internal_polys = [np.array(inpoly.xy).T.tolist() for inpoly in shp.interiors]
                out_parts.extend(internal_polys)
            
            w.poly(parts = out_parts)
        elif shp.geom_type == 'MultiPolygon':
            all_parts = []
            for shp_temp in shp:
                out_parts = []
                xtemp,ytemp = shp_temp.exterior.xy
                out_parts.append(np.array(list(zip(xtemp,ytemp))).tolist())
                # check for interior holes
                if len(shp_temp.interiors)>0:
                    internal_polys = [np.array(inpoly.xy).T.tolist() for inpoly in shp_temp.interiors]
                    out_parts.extend(internal_polys)
                    
                all_parts.extend(out_parts)
            w.poly(parts = all_parts)
        else:
            raise ValueError('Currently only implemented for polygons')
    
    if data is not None:
        w.records.extend(data)    
    w.save(out_fname)
    
    
    if write_prj_file:
        prj_name = os.path.basename(out_fname).split('.')[0]
        dir_name = os.path.dirname(out_fname)
        if inproj is None:
            inproj = ccrs.Geodetic(globe=default_globe)
        write_prj(inproj,os.path.join(dir_name,prj_name))

def write_prj(inproj,fname):
    '''Write projection file for shapefile.
    '''
    shp_dir,shp_name =  os.path.split(fname)
    if hasattr(inproj,'proj4_params'): # ccrs projection information
        if inproj.proj4_params['proj']=='lonlat':
            # Geographic datum
            osr_wkt = osr.GetWellKnownGeogCSAsWKT(inproj.proj4_params['datum'])
        else:
            osr_proj = osr.SpatialReference()
            osr_proj.ImportFromProj4(inproj.proj4_init)
            osr_proj.MorphToESRI()
            osr_wkt = osr_proj.ExportToWkt()
    elif hasattr(inproj,'ImportFromProj4'): # osr projection object
        osr_wkt = inproj.ExportToWkt()
        
    else:
        # Assume inproj is osr_wkt
        osr_wkt = inproj
        
    prj_file = open("{}.prj".format(os.path.join(shp_dir,os.path.splitext(shp_name)[0])), "w")
    prj_file.write(osr_wkt)
    prj_file.close()

def copy_prj(projshp=None,unprojshp=None):
    
    prjshp_dir,prjshp_name =  os.path.split(projshp)
    prjname,_ = os.path.splitext(prjshp_name)
    prjfname = os.path.join(prjshp_dir,'{}.prj'.format(prjname))    
    
    unprjshp_dir,unprjshp_name =  os.path.split(unprojshp)
    unprjname,_ = os.path.splitext(unprjshp_name)
    newprjfname = os.path.join(unprjshp_dir,'{}.prj'.format(unprjname)) 
    
    if os.path.isfile(prjfname):
        
        shutil.copy(prjfname,newprjfname)
    else:
        print("{} does not have a .prj file, choose another shapefile.".format(prjname))
    
def load_prj(prjpath=None):
    prj_list = []
    
    if 'shp' in prjpath:
        # use filename to get prj
        prjpath = '{}.prj'.format(os.path.splitext(prjpath)[0])
    
    with open(prjpath,'r') as prj:
        for iline in prj:
            prj_list.append(iline)
    
    sr_out = osr.SpatialReference()
    sr_out.ImportFromWkt(prj_list[0])
    return sr_out
    
    
def shp_to_patchcollection(in_polys=None,in_shp=None,radius=500.):
    '''Shapefile plotting.
    
    Source: modified from flopy.plot.plotutil.shapefile_to_patch_collection
    '''

    from matplotlib.patches import Polygon as mPolygon
    from matplotlib.patches import Circle as mCircle
    from matplotlib.patches import Path as mPath
    from matplotlib.patches import PathPatch as mPathPatch
    from matplotlib.collections import PatchCollection
    
    in_polygons = []
    # If shapefile path input:
    if (in_shp is not None):
        in_polygons,_ = shp_to_polygon(in_shp)
    
    if (in_polys is not None):
        in_polygons.extend(in_polys)
    
    blanks = []
    ptchs = []
    for poly in in_polys:
        st = poly.geom_type.lower()
        if st in ['point']:
            #points
            for p in poly.coords:
                ptchs.append(mCircle( (p[0], p[1]), radius=radius))
        elif st in ['linestring','linearring']:
            #line
            vertices = np.array(poly.coords)
            path = mPath(vertices)
            ptchs.append(mPathPatch(path, fill=False))
        elif st in ['polygon']:
            #polygons
            pts = np.array(poly.exterior.xy).T
            ptchs.append(mPolygon(pts))
            blanks.extend([mPolygon(np.array(poly1.xy).T) for poly1 in poly.interiors])
        elif st in ['multipolygon']:
            for ipoly in poly.geoms:
                ptchs.append(mPolygon(np.array(ipoly.exterior.xy).T))
                blanks.extend([mPolygon(np.array(ipoly1.xy).T) for ipoly1 in ipoly.interiors])
                
    pc = PatchCollection(ptchs)
    if len(blanks) > 0:
        bpc = PatchCollection(blanks)
    else:
        bpc = None
        
    return pc,bpc

def poly_bound_to_extent(in_polys):
    bound_list=[]
    for in_poly in in_polys:
        if in_poly.geom_type in ['Polygon']:
            bound_list.append(in_poly.bounds)
        else:
            for poly2 in in_poly:
                bound_list.append(poly2.bounds)

    bound_array = np.array(bound_list)
    # find maximum limits and rearrange to [xmin,xmax,ymin,ymax]
    max_bounds = np.array([bound_array[:,0].min(),bound_array[:,2].max(),
                           bound_array[:,1].min(),bound_array[:,3].max()])
    return max_bounds

def raster_to_polygon_slow(XY=None,Z=None,cell_spacing=None,unq_Z_vals=True):
    '''Convert raster area that is not nan to single polygon.
    '''
    
    from shapely.geometry import box as shpbox
    from shapely.ops import cascaded_union
    speedups.enable()
    
     # Create grid cells using cell centers and cell spacing
    if isinstance(cell_spacing,int) or isinstance(cell_spacing,float):
        cell_spacing = [cell_spacing,cell_spacing]
    
    if XY[0].shape != Z.shape:
        Z = Z.T.copy() # Try transposing


    if unq_Z_vals:
        # Create polygon for each unique value
        zvals = np.unique(Z[~np.isnan(Z)])
        out_shape = []
        nvals = []
        for zval in zvals:
            # Select only some of the grid cells depending on where Z is not nan
            XY2 = [[],[]]
            XY2[0]=XY[0][Z==zval].copy()
            XY2[1]=XY[1][Z==zval].copy()
        
            XYnew = list(zip(XY2[0].ravel(),XY2[1].ravel()))
            
            grid_squares = [shpbox(x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.) for x,y in XYnew]
            nvals.append(len(grid_squares))
            out_shape.append(unary_union(grid_squares))
        
    else:
        # Select only some of the grid cells depending on where Z is not nan
        XY2 = [[],[]]
        XY2[0]=XY[0][~np.isnan(Z)].copy()
        XY2[1]=XY[1][~np.isnan(Z)].copy()
        
        XYnew = list(zip(XY2[0].ravel(),XY2[1].ravel()))
        
        grid_squares = [shpbox(x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.) for x,y in XYnew]
        
        out_shape = [cascaded_union(grid_squares)]
        zvals = [1]
        nvals = len(grid_squares)

    return out_shape,zvals,nvals

def find_internal_features(in_poly=None):
    if in_poly.geom_type in ['MultiPolygon']:
        # Find largest feature
        shp_areas = [shp1.area for shp1 in in_poly]
        imax = np.argmax(shp_areas)
        max_shp = in_poly[imax]

        prep_shp= prep(max_shp)
        internal_shps = [shp1 for i,shp1 in enumerate(in_poly) if (i != imax) and prep_shp.contains(shp1)]
        external_shps = [shp1 for i,shp1 in enumerate(in_poly) if (i != imax) and not prep_shp.contains(shp1)]
        out_shp = max_shp.difference(cascaded_union(internal_shps))
        out_shp = out_shp.union(cascaded_union(external_shps))               
        return out_shp,internal_shps,external_shps
    else:
        return in_poly,[],[]                    
                      
    
def calc_edge_indexes(Z):
    from scipy.ndimage.filters import maximum_filter, minimum_filter
    edges = (maximum_filter(Z, size=2) == minimum_filter(Z, size=2))
    r1,c1 = np.array((~edges).nonzero())
    c2,r2 = sort_xy_cw(x=c1,y=r1)    
    return r2,c2

def calc_internal_gridcells(XY=None,Zbool=None,cell_spacing=None,
                            select_largest=True,slow_method=False):
    
    # Create new spatial grids
    XYtrue= [[],[]]
    XYtrue[0]=XY[0][Zbool].copy()
    XYtrue[1]=XY[1][Zbool].copy()
    XYnew = list(zip(XYtrue[0].ravel(),XYtrue[1].ravel())) # list of paired xy coordinates
    ntruecells = len(XYnew)
    
    if slow_method:
        grid_squares = [shpbox(x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.) for x,y in XYnew]
        out_shape=cascaded_union(grid_squares)
    else:
        # Calculate boundary of boolean array
        irows,icols = calc_edge_indexes(Zbool)
        pts = np.c_[XY[0][irows,icols].ravel(),XY[1][irows,icols].ravel()]
    #        pts = cru.raster_outline(XY[0],XY[1],Z) # old way
        linepoly = LineString(pts)
        try:
            poly = Polygon(linepoly).buffer(2.*cell_spacing[0],resolution=1,cap_style=3)
        except:
            # if only one or two cells were found, convert to polygon
            poly = cascaded_union([shpbox(x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.) for x,y in pts])
                    
        ifound = pt_in_shp(poly,[XYtrue[0].ravel(),XYtrue[1].ravel()],grid_buffer=cell_spacing[0]/2.)
        
        # ----- Find additional "true" cells ---------
        if len(ifound[0]) < ntruecells:
            mask=np.ones(ntruecells,dtype=bool)
            mask[ifound[0]] = False
    
            XYmissing = list(zip(XYtrue[0].ravel()[mask].flatten(),XYtrue[1].ravel()[mask].flatten()))
            grid_squares = [shpbox(x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.) for x,y in XYmissing]
    
            
            grid_squares.append(poly) # add main polygon to list
            out_shape = cascaded_union(grid_squares)
        else:
            out_shape = poly
               
        # ----- Remove "incorrect" true assigments --------
        XYfalse = [[],[]]
        XYfalse[0]=XY[0][~Zbool].copy()
        XYfalse[1]=XY[1][~Zbool].copy()
        nanfound = pt_in_shp(out_shape,[XYfalse[0].ravel(),XYfalse[1].ravel()],grid_buffer=cell_spacing[0]/2.)
        nanfound = np.hstack([inan for inan in nanfound if len(inan)>0])
        XYnan = list(zip(XYfalse[0].ravel()[nanfound].flatten(),XYfalse[1].ravel()[nanfound].flatten()))
        nangrid_squares = cascaded_union([shpbox(x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.) for x,y in XYnan])
    
        out_shape = out_shape.difference(nangrid_squares)
        
        # Remove polygons smaller than gridcell size
        if out_shape.geom_type in ['MultiPolygon']:
            out_shape = MultiPolygon([shp1 for shp1 in out_shape if shp1.area >= (cell_spacing[0]*cell_spacing[1])])
        
        # Select only largest feature
        while select_largest:
            if out_shape.geom_type in ['MultiPolygon']:
                outareas = [shp1.area for shp1 in out_shape]
                out_shape=out_shape[np.argmax(outareas)]
                
            else:
                select_largest=False
    
        
        # Clip to original extent
        bbox_array = [[XY[0][0,0],XY[1][0,0]],[XY[0][-1,0],XY[1][-1,0]],
                      [XY[0][-1,-1],XY[1][-1,-1]],[XY[0][0,-1],XY[1][0,-1]],
                      [XY[0][0,0],XY[1][0,0]]]
        bbox = Polygon(bbox_array)
        out_shape = out_shape.intersection(bbox)
    
    return out_shape,ntruecells

def raster_to_polygon_gdal(Z=None,XY=None,in_proj=None,out_shp=None,gt=None,gdal_dfmt=gdal.GDT_Int32,
                           nan_val=-9999,unq_Z_vals=True,layername="polygonized",
                           field_name='ID',field_fmt=ogr.OFTInteger,delete_shp=True):
    
    if XY is not None:
        if XY[0].shape != Z.shape:
            Z = Z.T.copy() # Try transposing
    
    nrows,ncols = Z.shape
    
    # make gdal raster in memory from np array
    dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal_dfmt)
    dst_ds.SetGeoTransform(gt)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(nan_val) #initialise raster with nans
    dst_rb.SetNoDataValue(nan_val)
    
    if unq_Z_vals:
        # Create polygon for each unique value
        make_shp_array=Z.copy()
    else:
        make_shp_array=np.zeros_like(Z,dtype=int)
        make_shp_array[~np.isnan(Z) & Z!=0] = 1
        
    dst_rb.WriteArray(make_shp_array)
    dst_ds.FlushCache()
    
    # Coordinate system management
    srs = osr.SpatialReference()
    srs.ImportFromWkt(in_proj)
    
    if not os.path.exists(os.path.dirname(out_shp)):
        os.makedirs(os.path.dirname(out_shp))
    
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(out_shp):
        if delete_shp:
            shp_driver.DeleteDataSource(out_shp)
            shp_src = shp_driver.CreateDataSource(out_shp)
            shp_layer = shp_src.CreateLayer(layername,srs=srs)
        else:
            shp_src = shp_driver.Open(out_shp,0)
            layernames = [shp_src.GetLayerByIndex(ilay).GetName() for ilay in shp_src.GetLayerCount()]
            if layername in layernames:
                shp_layer = shp_src.GetLayerByName(layername)
            else:
                shp_layer = shp_src.CreateLayer(layername,srs=srs)
    else:
        shp_driver.DeleteDataSource(out_shp)
        shp_src = shp_driver.CreateDataSource(out_shp)
        shp_layer = shp_src.CreateLayer(layername,srs=srs)
        
    new_field = ogr.FieldDefn(field_name,field_fmt)
    shp_layer.CreateField(new_field)
    
    _ = gdal.Polygonize(dst_rb,None,shp_layer,0,callback=None,options=['q'])
    
    shp_src.Destroy()
    dst_ds=None
    
def raster_to_polygon(XY=None,Z=None,cell_spacing=None,unq_Z_vals=True,
                      select_largest=True,slow_method=False):
    '''Convert raster area that is not nan to single polygon.
    '''

#    from cgw_model.cgw_utils import cgw_raster_utils as cru
    
     # Create grid cells using cell centers and cell spacing
    if isinstance(cell_spacing,int) or isinstance(cell_spacing,float):
        cell_spacing = [cell_spacing,cell_spacing]
    
    if XY[0].shape != Z.shape:
        Z = Z.T.copy() # Try transposing
        
    if unq_Z_vals:
        # Create polygon for each unique value
        zvals = np.unique(Z[~np.isnan(Z)])
        out_shape = []
        nvals = []
        for zval in zvals:
            # Select only some of the grid cells depending on where Z is not nan
            Zbool=np.zeros_like(Z,dtype=bool)
            Zbool[Z==zval] = True
            # Calculate edges of value
            out_shape_temp,nvals_temp = calc_internal_gridcells(XY=XY,Zbool=Zbool,
                                                                cell_spacing=cell_spacing,
                                                                select_largest=select_largest,
                                                                slow_method=slow_method)
            nvals.append(nvals_temp)
            out_shape.append(cascaded_union(out_shape_temp))
    else:
        # Select only some of the grid cells depending on where Z is not nan
        
        Zbool=np.ones_like(Z,dtype=bool)
        Zbool[Z==0] = False
        Zbool[np.isnan(Z)] = False
        
        out_shape,nvals = calc_internal_gridcells(XY=XY,Zbool=Zbool,cell_spacing=cell_spacing,
                                                  select_largest=select_largest,
                                                  slow_method=slow_method)
        zvals = [1]
        out_shape = [out_shape]
#        XY2 = [[],[]]
#        XY2[0]=XY[0][Znew].copy()
#        XY2[1]=XY[1][Znew].copy()
#          
#        XYnew = zip(XY2[0].ravel(),XY2[1].ravel())
#        zvals = [1]
#        nvals = len(XYnew)
#        irows,icols = calc_edge_indexes(Znew)
#        pts = np.c_[XY[0][irows,icols].ravel(),XY[1][irows,icols].ravel()]
##        pts = cru.raster_outline(XY[0],XY[1],Z)
#        linepoly = LineString(pts)
#        poly = Polygon(linepoly).buffer(5.*cell_spacing[0],resolution=1,cap_style=3)
#        ifound = pt_in_shp(poly,[XY2[0].ravel(),XY2[1].ravel()])
#        if len(ifound[0]) < len(XYnew):
#            mask=np.ones(len(XYnew),dtype=bool)
#            mask[ifound[0]] = False
##            if cell_spacing[0] != cell_spacing[1]:
#            XYmissing = zip(XY2[0].ravel()[mask].flatten(),XY2[1].ravel()[mask].flatten())
#            grid_squares = [shpbox(x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.) for x,y in XYmissing]
##            else:
##                XYmissing = np.c_[XY2[0].ravel()[mask].flatten(),XY2[1].ravel()[mask].flatten()]
##                grid_squares = [ptemp.buffer(cell_spacing[0]/2.,1.) for ptemp in MultiPoint(XYmissing)]
#            
#            grid_squares.append(poly) # add main polygon to list
#            out_shape = cascaded_union(grid_squares)
#        else:
#            out_shape = poly
#            
#        
#        
#        # subtract incorrect nan assigments
#        XY3 = [[],[]]
#        XY3[0]=XY[0][~Znew].copy()
#        XY3[1]=XY[1][~Znew].copy()
#        nanfound = pt_in_shp(out_shape,[XY3[0].ravel(),XY3[1].ravel()])
##        if cell_spacing[0] != cell_spacing[1]: # fastest way
#        XYnan = zip(XY3[0].ravel()[nanfound].flatten(),XY3[1].ravel()[nanfound].flatten())
#        nangrid_squares = cascaded_union([shpbox(x-cell_spacing[0]/2.,y-cell_spacing[1]/2.,x+cell_spacing[0]/2.,y+cell_spacing[1]/2.) for x,y in XYnan])
##        else:
##            XYnan = np.c_[XY3[0].ravel()[nanfound].flatten(),XY3[1].ravel()[nanfound].flatten()]
##            nangrid_squares = [ptemp.buffer(cell_spacing[0]/2.,1.) for ptemp in MultiPoint(XYnan)]
##            if nangrid_squares.geom_type in ['MultiPolygon']:
##                nangrid_squares = cascaded_union(nangrid_squares)
#
#        out_shape = out_shape.difference(nangrid_squares)
#        
#        # select only largest shape
#        if out_shape.geom_type in ['MultiPolygon']:
#            outareas = [shp1.area for shp1 in out_shape]
#            out_shape=out_shape[np.argmax(outareas)]
#        if out_shape.geom_type in ['MultiPolygon']:
#            # Repeat
#            outareas = [shp1.area for shp1 in out_shape]
#            out_shape=out_shape[np.argmax(outareas)]
#        
#        # clip to original extent
#        bbox_array = [[XY[0][0,0],XY[1][0,0]],[XY[0][-1,0],XY[1][-1,0]],
#                      [XY[0][-1,-1],XY[1][-1,-1]],[XY[0][0,-1],XY[1][0,-1]],
#                      [XY[0][0,0],XY[1][0,0]]]
#        bbox = Polygon(bbox_array)
#        out_shape = out_shape.intersection(bbox)
    return out_shape,zvals,nvals
    
#def merge_shps(in_shp_fnames=None,out_fname=None):
#    '''Merge shapefiles.
#    Only works with python 2.7
#    Source: afte http://geospatialpython.com/2014/06/merging-shapefiles-with-pyshp-and-dbfpy.html'''
#    from dbfpy import dbf
#    w = shapefile.Writer()
#    # Loop through ONLY the shp files and copy their shapes
#    # to a writer object. We avoid opening the dbf files
#    # to prevent any field-parsing errors.
#    for f in in_shp_fnames:
#        shpf = open(f, "rb")
#        r = shapefile.Reader(shp=shpf)
#        w._shapes.extend(r.shapes())
#        shpf.close()
#        
#    # Save only the shp and shx index file to the new
#    # merged shapefile.
#    w.saveShp(out_fname)
#    w.saveShx(out_fname)
#    
#    # Now we come back with dbfpy and merge the dbf files
#    dbf_files = [os.path.join(os.path.dirname(fname),"{}.dbf".format(os.path.basename(fname).split('.')[0])) for fname in in_shp_fnames]
#    
#    # Use the first dbf file as a template
#    template = dbf_files.pop(0)
#    merged_dbf_name = os.path.join(os.path.dirname(out_fname),"{}.dbf".format(os.path.basename(out_fname).split('.')[0]))
#    
#    # Copy the entire template dbf file to the merged file
#    merged_dbf = open(merged_dbf_name, "wb")
#    temp = open(template, "rb")
#    merged_dbf.write(temp.read())
#    merged_dbf.close()
#    temp.close()
#    # Now read each record from the remaining dbf files
#    # and use the contents to create a new record in
#    # the merged dbf file. 
#    db = dbf.Dbf(merged_dbf_name)
#    for f in dbf_files:
#        dba = dbf.Dbf(f)
#        for rec in dba:
#            db_rec = db.newRecord()
#            for k,v in rec.asDict().items():
#                db_rec[k] = v
#            db_rec.store()
#    db.close()
    

    
def plot_shp(in_polys=None,in_shp=None, ax=None,
             extent=None,radius=500., cmap='Dark2',
             edgecolor='scaled', facecolor='scaled',
             a=None, masked_values=None,
             **kwargs):
    """
    Generic function for plotting a shapefile.
    
    
    Parameters
    ----------
    shp : string
        Name of the shapefile to plot.
    radius : float
        Radius of circle for points.  (Default is 500.)
    linewidth : float
        Width of all lines. (default is 1)
    cmap : string
        Name of colormap to use for polygon shading (default is 'Dark2')
    edgecolor : string
        Color name.  (Default is 'scaled' to scale the edge colors.)
    facecolor : string
        Color name.  (Default is 'scaled' to scale the face colors.)
    a : numpy.ndarray
        Array to plot.
    masked_values : iterable of floats, ints
        Values to mask.
    kwargs : dictionary
        Keyword arguments that are passed to PatchCollection.set(``**kwargs``).
        Some common kwargs would be 'linewidths', 'linestyles', 'alpha', etc.

    Returns
    -------
    pc : matplotlib.collections.PatchCollection

    Examples
    --------
    
    Source: modified from flopy.plot.plotutil.plot_shapefile
    """
    import matplotlib.pyplot as plt
    
    if 'vmin' in kwargs:
        vmin = kwargs.pop('vmin')
    else:
        vmin = None

    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        vmax = None

    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    pc,bpc = shp_to_patchcollection(in_polys=in_polys,in_shp=in_shp,radius=radius)
    pc.set(**kwargs)
    if a is None:
        nshp = len(pc.get_paths())
        cccol = cm(1. * np.arange(nshp) / nshp)
        if facecolor == 'scaled':
            pc.set_facecolor(cccol)
        else:
            pc.set_facecolor(facecolor)
        if edgecolor == 'scaled':
            pc.set_edgecolor(cccol)
        else:
            pc.set_edgecolor(edgecolor)
    else:
        pc.set_cmap(cm)
        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)
        if edgecolor == 'scaled':
            pc.set_edgecolor('none')
        else:
            pc.set_edgecolor(edgecolor)
        pc.set_array(a)
        pc.set_clim(vmin=vmin, vmax=vmax)
    # add the patch collection to the axis
    ax.add_collection(pc)
    
    # overlap polygons with white/blank polygons of interior holes
    if bpc is not None:
        bpc.set_edgecolor('none')
        bpc.set_facecolor('w')
        ax.add_collection(bpc)
    
    if (extent is not None):
        ax.axis(extent)
    else:
        ax.axis(poly_bound_to_extent(in_polys))
    plt.show()
    return ax,pc    