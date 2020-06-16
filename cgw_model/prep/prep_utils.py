# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:01:55 2016

@author: kbefus
"""

import arcpy
import os
import numpy as np
from osgeo import gdal, osr
from scipy.interpolate import griddata
import shapefile
from shapely import speedups
from shapely.geometry import Polygon,MultiPoint
from shapely.prepared import prep

def gdal_error_handler(err_class, err_num, err_msg):
    '''
    Capture gdal error and report if needed
    
    Source:
    http://pcjericks.github.io/py-gdalogr-cookbook/gdal_general.html#install-gdal-ogr-error-handler
    '''
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print 'Error Number: %s' % (err_num)
    print 'Error Type: %s' % (err_class)
    print 'Error Message: %s' % (err_msg)



# Overwrite pre-existing files
arcpy.env.overwriteOutput = True
arcpy.env.compression = 'LZW'
#arcpy.env.pyramid = "NONE"

class LicenseError(Exception):
    pass

license_name = 'Spatial'
if arcpy.CheckExtension(license_name) == "Available":
    arcpy.CheckOutExtension(license_name)
else:
    # Raise a custom exception
    #
    raise LicenseError



def collect_NED(ned_dir=None,XY=None,internal_pts=None):
    ned_fnames = []
    for wtemp,ntemp in zip(XY[0][internal_pts].ravel(),XY[1][internal_pts].ravel()):
        fname = os.path.join(ned_dir,'grdn{0:02d}w{1:03d}_1'.format(int(ntemp),int(np.abs(wtemp))))
        fname_test = os.path.isdir(fname)
        if not fname_test: # test for grids named with different convention
            fname = os.path.join(ned_dir,'n{0:02d}w{1:03d}'.format(int(ntemp),int(np.abs(wtemp))))
            fname_test2 = os.path.isdir(fname)
            if fname_test2:
                ned_fnames.append(fname)
        else:
            ned_fnames.append(fname)
            
    return ned_fnames

def mosaic_NED(ned_fnames,out_fname,ndecimate=3.,proj_out=None,
               cell_size=None):
                   
    if cell_size is None:
        cell_size=arcpy.Describe(ned_fnames[0]).meanCellwidth*ndecimate
    
    if proj_out is None:
        proj_out = ned_fnames[0]
    
    arcpy.env.extent = 'MAXOF'
    arcpy.env.compression = 'LZW'
#    arcpy.env.pyramid = "NONE"
    arcpy.env.outputCoordinateSystem = proj_out
    arcpy.env.cellSize = cell_size
    
    arcpy.MosaicToNewRaster_management(";".join(ned_fnames), \
                                       os.path.dirname(out_fname), \
                                       os.path.basename(out_fname), proj_out,\
                                       "32_BIT_FLOAT", cell_size, "1", "LAST","FIRST")


def join_NED_CRM(ned_fname,crm_fname,out_fname,sea_level=0.,
                 max_elev_from_bathy = None):
    ned = arcpy.Raster(ned_fname)
    crm = arcpy.Raster(crm_fname)
    
    arcpy.env.overwriteOutput = True
    arcpy.env.extent = 'MAXOF'
    arcpy.env.compression = 'LZW'
#    arcpy.env.pyramid = "NONE"
    # Project CRM to NED if not done already
    proj_out = ned
    arcpy.env.outputCoordinateSystem = proj_out
    delete_flag = False
    if not arcpy.Describe(crm).SpatialReference.exporttostring()==arcpy.Describe(ned).SpatialReference.exporttostring():
            crm_extract_new = os.path.join(os.path.dirname(crm_fname),'temp.tif')
            arcpy.ProjectRaster_management(in_raster=crm, out_raster= crm_extract_new,
                                           out_coor_system=proj_out, resampling_type="BILINEAR",
                                           cell_size=ned.meanCellHeight)
            crm = arcpy.Raster(crm_extract_new)
            delete_flag = True
    
    
    
    # Assign ned null values to crm                               
    ned2 = arcpy.sa.Con(arcpy.sa.IsNull(ned),crm,ned)
    
    # Assign crm at and above sea level to null
    crm1 = arcpy.sa.SetNull(crm>=sea_level,crm)
        
    # Assign ned where crm is null
    crm2 = arcpy.sa.Con(arcpy.sa.IsNull(crm1),ned2,crm1)
    
    # Assign ned-adjusted crm where ned is less than some small elevation, max_elev_from_bathy
    merged_data = arcpy.sa.Con(ned2<max_elev_from_bathy,crm2,ned2)
    
    # Save data
    arcpy.CopyRaster_management(merged_data,out_fname)

    # Clean up
    arcpy.Delete_management(ned2)
    
    if delete_flag:
        arcpy.Delete_management(crm_extract_new)  
    
    del ned,crm,ned2,crm1,crm2,merged_data
    
def fix_NED_CRM(dem_fname,shp_fname,out_fname=None,
                npt_interpbuffer=0,npt_domainbuffer=50,
                max_elev_to_fix=1.,higher_elev=5.):
    
    # Load DEM
    X,Y,Z=read_griddata(dem_fname)
    
    # Load Shapefile
    shp_shapefile = shapefile.Reader(shp_fname)
    shp_shapes = shp_shapefile.shapes()
    shp_types = np.array(shp_shapefile.records())
    
    orig_fix_level = max_elev_to_fix
    for active_shape,active_type in zip(shp_shapes,shp_types):
        if ('Channel' not in active_type and 'Interp' not in active_type)\
           and ('offshore' in active_type or 'both' in active_type):
            
            if 'higher' in active_type:
                max_elev_to_fix = higher_elev
            else:
                max_elev_to_fix = orig_fix_level
            shp_bbox = active_shape.bbox # minx,miny,maxx,maxy
            # Select subset of elevation data
            locate_extent_x = ((X>=shp_bbox[0]) & (X<=shp_bbox[2])).nonzero()[1]
            locate_extent_y = ((Y>=shp_bbox[1]) & (Y<=shp_bbox[3])).nonzero()[0]
            if len(locate_extent_x)>0 and len(locate_extent_y)>0:
                minx,maxx = np.maximum(0,locate_extent_x.min()-npt_domainbuffer),np.minimum(X.shape[1],locate_extent_x.max()+npt_domainbuffer+1)
                miny,maxy =  np.maximum(0,locate_extent_y.min()-npt_domainbuffer),np.minimum(X.shape[0],locate_extent_y.max()+npt_domainbuffer+1)
                X2,Y2,Z2 = X[miny:maxy,minx:maxx].copy(),Y[miny:maxy,minx:maxx].copy(),Z[miny:maxy,minx:maxx].copy()
                # Find indexes of cells to be replaced within new subset grid
                locate_extent_x2 = ((X2>=shp_bbox[0]) & (X2<=shp_bbox[2])).nonzero()[1]
                locate_extent_y2 = ((Y2>=shp_bbox[1]) & (Y2<=shp_bbox[3])).nonzero()[0]
                minx2,maxx2 = np.maximum(0,locate_extent_x2.min()-npt_interpbuffer),np.minimum(X2.shape[1],locate_extent_x2.max()+npt_interpbuffer+1)
                miny2,maxy2 =  np.maximum(0,locate_extent_y2.min()-npt_interpbuffer),np.minimum(X2.shape[0],locate_extent_y2.max()+npt_interpbuffer+1)
                
                irow,icol,_ = gridpts_in_shp(active_shape,[X2,Y2])
                if len(irow) > 0:
                    Zarea_temp = Z2.copy()
                    Zarea_temp[irow,icol] = np.nan # set cells to replace to nan
                
                    imask = (Z2>=max_elev_to_fix) | np.isnan(Zarea_temp)#| \
                        
                    new_Z = griddata(np.c_[X2[~imask],Y2[~imask]],Z2[~imask],
                                     (X2[miny2:maxy2+1,minx2:maxx2+1],
                                      Y2[miny2:maxy2+1,minx2:maxx2+1]),
                                      method='linear')
                    
                        
                    Z2_new = Z2.copy()
                    
                    Z2_new[miny2:maxy2+1,minx2:maxx2+1][(Z2<max_elev_to_fix)[miny2:maxy2+1,minx2:maxx2+1]]=new_Z[(Z2<max_elev_to_fix)[miny2:maxy2+1,minx2:maxx2+1]]
                    # Replace section in original data
                    Z[miny:maxy,minx:maxx][irow,icol] = Z2_new[irow,icol].copy()
        else:
            continue
    
    # Save corrected grid
    if out_fname is None:
        out_fname = os.path.join(os.path.dirname(dem_fname),
                                 '{}_fixed.tif'.format(os.path.basename(dem_fname).split('.')[0]))

    nan_val = -5e3
    Z[Z<nan_val] = np.nan
    write_gdaltif(out_fname,X,Y,Z.copy())
    
    return out_fname
    
def join_DEM_bathy(dem1_fname,dem2_fname,shp_not_dem1,shp_use_dem2,
                   out_fname=None,
                   fill_dem2=True,elev_threshold=0.):
    
    # arcpy environmental variables
    arcpy.env.overwriteOutput = True
    arcpy.env.compression = 'LZW'
#    arcpy.env.pyramid = "NONE"
    arcpy.env.extent = 'MAXOF'
    arcpy.env.extent = dem1_fname
    arcpy.env.cellSize = dem1_fname
    arcpy.env.outputCoordinateSystem = dem1_fname
    
    # Prepare dem2 to replace dem1
    dem2_r = arcpy.Raster(dem2_fname)
    if fill_dem2:
        dem2_filled = arcpy.sa.Fill(dem2_r)
    else:
        dem2_filled = dem2_r
        
    dem2_use = arcpy.sa.ExtractByMask(dem2_filled,shp_not_dem1)
    dem2_joined = arcpy.sa.Con(arcpy.sa.IsNull(dem2_use),dem2_r,dem2_use)

    # Load dem1
    dem1_r = arcpy.Raster(dem1_fname)
    
    # Add dem1 elevation to doctored dem2
    dem2_nulled = arcpy.sa.SetNull(dem2_joined>=elev_threshold,dem2_joined)
    dem2_null_filled = arcpy.sa.Con(arcpy.sa.IsNull(dem2_nulled),dem1_r,dem2_nulled)
    dem2_for_join = arcpy.sa.ExtractByMask(dem2_null_filled,shp_use_dem2)
    
    # Join dem1 and dem2
    joined_r = arcpy.sa.Con(arcpy.sa.IsNull(dem2_for_join),dem1_r,dem2_for_join)
    
    if out_fname is None:
        out_fname = os.path.join(os.path.dirname(dem1_fname),
                                 '{}_join.tif'.format(os.path.basename(dem1_fname).split('.')[0]))
    
    arcpy.CopyRaster_management(joined_r,out_fname)
    
    del dem1_r,dem2_r,dem2_nulled,\
        dem2_null_filled,dem2_for_join,\
        joined_r,dem2_use,dem2_joined,dem2_filled
    
    return out_fname

# ------ Feature operators ------

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
        in_polygon = [Polygon(shape.points) for shape in shp1.shapes()]
    
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

def gridpts_in_shp(in_polygon,XYpts, print_no_match = False):
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
            pt_spacing = np.abs(np.diff(grid_centers_x,axis=1).mean()+1j*np.diff(grid_centers_y,axis=0).mean())/2. # mean diagonal/2 for cells
            gx,gy,inpts = reduce_extent(poly_extent,grid_centers_x,grid_centers_y,
                                      buffer_size = pt_spacing)
            inpts_rows_cols = np.array(inpts.nonzero()).T
            pts = MultiPoint(zip(gx,gy))
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
                out_unq_ind_order = unique_rows(np.c_[all_y_cols,all_x_cols])
                out_rows = np.array(all_y_cols)[out_unq_ind_order]
                out_cols = np.array(all_x_cols)[out_unq_ind_order]
                out_poly = np.array(collect_polys)[out_unq_ind_order]
                return out_rows.ravel(),out_cols.ravel(),out_poly.ravel()
            else:
                return loop_y_rows.ravel(),loop_x_cols.ravel(),np.array(collect_polys).ravel()
        else:
            if print_no_match:
                print 'No matches found, all points outside domain:'
                print 'polygon extent: {}'.format(np.around(poly_extent,decimals=1))
                print 'points extent: {}'.format(np.around([grid_centers_x.min(),grid_centers_y.min(),grid_centers_x.max(),grid_centers_y.max()],decimals=1))
            return [],[],[]
    else:
        return [],[],[]    



# ------ Grid operators ------
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

def fill_mask(in_array,fill_value=np.nan):
    if hasattr(in_array,'mask'):
        out_array = np.ma.filled(in_array.copy(),fill_value)
    else:
        out_array = in_array.copy()
        
    if ~np.isnan(fill_value):
        out_array[np.isnan(out_array)]=fill_value
        
    return out_array

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

def read_griddata(fname,in_extent = None):
    '''
    Read grid data and export as numpy array
    in_extent = minx,miny,maxx,maxy
    
    '''
    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)
    
    gdal.UseExceptions()
    gdal.AllRegister()
    indataset = gdal.Open(fname)
    
    # Get grid size
    rows = indataset.RasterYSize
    cols = indataset.RasterXSize
    nullVal = indataset.GetRasterBand(1).GetNoDataValue() # 1 for some reason
    # Get georeference info
    proj = indataset.GetProjectionRef()
    inproj = osr.SpatialReference()
    inproj.ImportFromWkt(proj)
     
    gt = indataset.GetGeoTransform()
    # Origin at upper left
    xOrigin = gt[0] # westernmost cell left edge
    yOrigin = gt[3] # northernmost cell top edge
    pixelWidth = gt[1] # positive
    pixelHeight = gt[5] # negative
          
    xmin = xOrigin + pixelWidth * 0.5
    xmax = xmin + (pixelWidth * cols)
    ymax = yOrigin + pixelHeight* 0.5 # grid origin is at the top of cell
    ymin = yOrigin + (pixelHeight * rows) + pixelHeight * 0.5

    # Convert raster to np array
    out_data = indataset.ReadAsArray(0,0,cols,rows).astype(np.float)
    out_data[out_data==nullVal] = np.nan

    xvect = np.arange(xmin,xmax,pixelWidth)       
    yvect = np.arange(ymax,ymin,pixelHeight)
    
    RX,RY = np.meshgrid(xvect,yvect) # X increases in cols, Y decreases in rows
    
    if in_extent is not None:
        locate_extent_x = ((RX>=in_extent[0]) & (RX<=in_extent[2])).nonzero()[1]
        locate_extent_y = ((RY>=in_extent[1]) & (RY<=in_extent[3])).nonzero()[0]
        minx,maxx = locate_extent_x.min(),locate_extent_x.max()+1
        miny,maxy = locate_extent_y.min(),locate_extent_y.max()+1
        
        RX,RY,out_data = RX[miny:maxy,minx:maxx],RY[miny:maxy,minx:maxx],out_data[miny:maxy,minx:maxx]    
    
            
    indataset = None        
    return RX,RY,out_data

def write_gdaltif(fname,X,Y,Vals,rot_xy=0.,
                  proj_wkt=None,set_proj=True,
                  nan_val = -9999.):
    '''
    Write geotiff to file from numpy arrays
    '''
    # Create gtif
    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)
    
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(fname, Vals.shape[1], Vals.shape[0], 1, gdal.GDT_Float32,options = [ 'COMPRESS=LZW' ] )
    # does not take into account rotation in dx,dy yet
    dy = np.mean(np.diff(Y,axis=0))
    dx = np.mean(np.diff(X,axis=1))
    geodata = [X[0,0]-dx/2.,np.cos(rot_xy)*dx,-np.sin(rot_xy)*dx,
               Y[0,0]-(0.5*dy),np.sin(rot_xy)*dy,np.cos(rot_xy)*dy] # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    ds_out.SetGeoTransform(geodata)
    
    Vals_out = Vals.copy()
    Vals_out[np.isnan(Vals_out)]=nan_val

    if set_proj:
        # set the reference info
        srs = osr.SpatialReference()
        if proj_wkt is None:
            srs.SetWellKnownGeogCS("NAD83")
        else:
            srs.ImportFromWkt(proj_wkt)
            
        ds_out.SetProjection(srs.ExportToWkt())
        
    # write the band    
    outband = ds_out.GetRasterBand(1)
    outband.SetNoDataValue(nan_val)
    outband.WriteArray(Vals_out)
    outband.FlushCache()
    ds_out,outband = None, None
    
    
    
    
    
        