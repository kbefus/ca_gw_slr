# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:31:47 2016

cgw_raster_utils.py

Collection of functions to aid Modflow grid creation and manipulation from
spatial datasets

@author: kbefus
"""
from __future__ import print_function
import os,sys
import netCDF4
from osgeo import gdal, osr, gdalnumeric, ogr
from PIL import Image, ImageDraw
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import binned_statistic_2d
#import cartopy.crs as ccrs
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage import measurements
from scipy.special import erf
from . import cgw_general_utils as cgu
from . import cgw_feature_utils as cfu

import rasterio
import affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform


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
    print('Error Number: %s' % (err_num))
    print('Error Type: %s' % (err_class))
    print('Error Message: %s' % (err_msg))

# Grid management
#default_globe = ccrs.Globe('NAD83')

def projectXY(xy_source, inproj=None, outproj="NAD83"):
    '''
    Convert coordinates of raster source
    
    '''
    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)
    
    dest_proj = osr.SpatialReference()
    src_proj = osr.SpatialReference()
    
    
    if isinstance(outproj,(float,int)):
        dest_proj.ImportFromEPSG(outproj)
    elif '+' in outproj:
        dest_proj.ImportFromProj4(outproj)
    elif 'PROJCS' in outproj or 'GEOGCS' in outproj:
        dest_proj.ImportFromWkt(outproj)
    elif hasattr(outproj,'ImportFromWkt'):
        dest_proj = outproj # already an osr sr
    else:
        # Assume outproj is geographic sr
        dest_proj.SetWellKnownGeogCS(outproj)
    
    if isinstance(inproj,(float,int)):
        src_proj.ImportFromEPSG(inproj)
    elif '+' in inproj:
        src_proj.ImportFromProj4(inproj)
    elif 'PROJCS' in inproj or 'GEOGCS' in inproj:
        src_proj.ImportFromWkt(inproj)
    elif hasattr(inproj,'ImportFromWkt'):
        src_proj = inproj # already an osr sr
    else:
        # Assume outproj is geographic sr
        src_proj.SetWellKnownGeogCS(inproj)
        
    # the ct object takes and returns pairs of x,y, not 2d grids
    # so the the grid needs to be reshaped (flattened) and back.
    ct = osr.CoordinateTransformation(src_proj, dest_proj)
    
    if len(xy_source.shape) == 3:
        shape = xy_source[0,:,:].shape
        size = xy_source[0,:,:].size
        xy_source = xy_source.reshape(2, size).T
        xy_target = np.array(ct.TransformPoints(xy_source))
        xx = xy_target[:,0].reshape(shape)
        yy = xy_target[:,1].reshape(shape)
        return xx, yy
    else:
        transposed = False
        if xy_source.shape[0]<xy_source.shape[1]:
            xy_source = xy_source.T # want n x 2 matrix
            transposed = True        

        xy_target = np.array(ct.TransformPoints(xy_source))[:,:2] # only want first 2 columns
    
        if transposed:
            xy_target = xy_target.T
            
        return xy_target


def load_grid_prj(fname=None,gt_out=False):
    
    indataset = gdal.Open(fname)
    proj_wkt = indataset.GetProjectionRef()
    if gt_out:
        gt=indataset.GetGeoTransform()
        indataset = None 
        return proj_wkt,gt
    else:
        indataset=None
        return proj_wkt
    
def get_rast_info(rast):
    indataset = gdal.Open(rast)
    nrows = indataset.RasterYSize
    ncols = indataset.RasterXSize
    gt=indataset.GetGeoTransform()
    indataset=None    
    return nrows,ncols,gt

def read_griddata(fname,in_extent = None, ideal_cell_size = None,
                  force_cell_size_mult = False,
                  shp_in_proj = None,
                  transform_coords=False,ndec=5,load_data=True):
    '''Read gridded spatial data using GDAL.
    
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
    Xp,Yp = np.meshgrid(np.arange(cols),np.arange(rows))        

    Xgeo = lambda gt,xpixel,yline: gt[0] + xpixel*gt[1] + yline*gt[2]
    Ygeo = lambda gt,xpixel,yline: gt[3] + xpixel*gt[4] + yline*gt[5]

    # Calculate for cell centers
    RX = Xgeo(gt,Xp,Yp) + gt[1]/2. + gt[2]/2.
    RY = Ygeo(gt,Xp,Yp) + gt[5]/2. + gt[4]/2.
    
    # Convert raster to np array
    if load_data:
        out_data = indataset.ReadAsArray(0,0,cols,rows).astype(np.float)
        out_data[out_data==nullVal] = np.nan
    else:
        out_data=[]
    
    # Origin at upper left
    #    sig_figs = 10
    #    round_func = lambda x: np.around(x,decimals = sig_figs)
    
    # Old way
#    xOrigin = gt[0] # westernmost cell left edge
#    xRot = gt[2] # grid rotation in horizontal
#    yOrigin = gt[3] # northernmost cell top edge
#    pixelWidth = gt[1] # positive
#    pixelHeight = gt[5] # negative
#    yRot = gt[4]
#    
#    if yRot==0 and xRot==0:   
#        # Grid isn't rotated
#        xmin = xOrigin + pixelWidth * 0.5
#        xmax = xmin + (pixelWidth * cols)
#        ymax = yOrigin + pixelHeight* 0.5 # grid origin is at the top of cell
#        ymin = yOrigin + (pixelHeight * rows) + pixelHeight * 0.5
#    
#    
#        xvect = np.arange(xmin,xmax,pixelWidth)
#        yvect = np.arange(ymax,ymin,pixelHeight)    
#    
#        # Catch inconsistent information in header    
#        if xvect.shape[0] < out_data.shape[1]:
#            xvect = np.arange(xmin,xmax+pixelWidth,pixelWidth)
#            
#        if xvect.shape[0] > out_data.shape[1]:
#            xvect = np.arange(xmin,xmax-pixelWidth,pixelWidth)
#            
#        if yvect.shape[0] < out_data.shape[0]:
#            yvect = np.arange(ymax,ymin+pixelHeight,pixelHeight)
#        
#        if yvect.shape[0] > out_data.shape[0]:
#            yvect = np.arange(ymax,ymin-pixelHeight,pixelHeight)
#        
#    
#        RX,RY = np.meshgrid(xvect,yvect) # X increases in cols, Y decreases in rows
#    else:
        # Grid is rotated

        
    if in_extent is not None:
        locate_extent_x = ((RX>=in_extent[0]) & (RX<=in_extent[2])).nonzero()[1]
        locate_extent_y = ((RY>=in_extent[1]) & (RY<=in_extent[3])).nonzero()[0]
        minx,maxx = locate_extent_x.min(),locate_extent_x.max()+1
        miny,maxy = locate_extent_y.min(),locate_extent_y.max()+1
        
        RX,RY,out_data = RX[miny:maxy,minx:maxx],RY[miny:maxy,minx:maxx],out_data[miny:maxy,minx:maxx]    
    
    if inproj.IsGeographic() != 1 and transform_coords:
        try:
            print('Raster not in long/lat...tryting to project to long/lat')
            osr_proj = osr.SpatialReference()
            osr_proj.ImportFromProj4(shp_in_proj.proj4_init)
#            osr_wkt = osr.GetWellKnownGeogCSAsWKT(shp_in_proj.proj4_params['datum'])
            srsLatLong = inproj.ImportFromWkt(osr_proj.ExportToWkt())
    
            RX,RY = projectXY(np.array([RX,RY]), inproj, srsLatLong)
        except:
            print('Failed: Raster not in long/lat...project to long/lat if desired')
    else:
        # Don't try to change cell size if projected
        if ideal_cell_size is not None:
            dx,dy = np.mean(np.diff(RX,axis=1)),np.mean(np.diff(RY,axis=0))
            cell_ratio = float(ideal_cell_size)/np.nanmax([dx,dy])
            if cell_ratio > 5. and cell_ratio < 1e3:
                # it is worth downsampling the data
                if force_cell_size_mult is True:
                    ntimes_gt_ideal = 1. # force exact cell size
                elif isinstance(force_cell_size_mult,float):
                    ntimes_gt_ideal = force_cell_size_mult # force ratio of cell size
                else:
                    ntimes_gt_ideal = 3. # make spacing 3x larger than ideal to leave info for future interpolation
                    
                new_ncols, new_nrows = np.ceil(cols*(ntimes_gt_ideal/cell_ratio)),np.ceil(rows*(ntimes_gt_ideal/cell_ratio))
                newx,newy = np.linspace(RX[0,0],RX[0,-1],new_ncols),np.linspace(RY[0,0],RY[1,0],new_nrows)
                RX,RY = np.meshgrid(newx,newy[::-1])  # X increases in cols, Y decreases in rows
                interp_func = RegularGridInterpolator((RX[0,:].flatten(),RY[:,0].flatten()),out_data[::-1,:].T)
                interp_out = interp_func(np.c_[RX.ravel(),RY.ravel()])
                out_data = interp_out.reshape(RX.shape)
            
    indataset = None        
    return RX,RY,out_data

def write_gdaltif(fname,X,Y,Vals,rot_xy=0.,
                  proj_wkt=None,set_proj=True,
                  nan_val = -9999.,dxdy=[None,None],
                  geodata=None):
    '''Write geotiff to file from numpy arrays.
    '''
    # Create gtif
    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)
    
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(fname, Vals.shape[1], Vals.shape[0], 1, gdal.GDT_Float32,options = [ 'COMPRESS=LZW' ] )
    
    # Check grid rotation
    if X is not None and Y is not None:
        if rot_xy==0.:
            [X,Y],Vals,rot_xy = grid_rot(XY=[X,Y],val=Vals)
        
        if dxdy[0] is None:
            # Calculate dx,dy taking into account rotation
            dx,dy = calc_dxdy(XY=[X,Y],rot_xy=rot_xy)
        else:
            dx,dy = dxdy
        
    # [top left x, w-e pixel resolution, rotation, 
    #  top left y, rotation, n-s pixel resolution]   
    if geodata is None:
        # need to move origin from cell center to top left node of grid cell
        geodata = [X[0,0]-np.cos(-rot_xy)*dx/2.+np.sin(-rot_xy)*dx/2.,
                   np.cos(-rot_xy)*dx,
                   -np.sin(-rot_xy)*dx,
                   Y[0,0]-np.cos(-rot_xy)*dy/2.-np.sin(-rot_xy)*dy/2.,
                   np.sin(-rot_xy)*dy,
                   np.cos(-rot_xy)*dy] 

#    # Test geotransform - not quite working yet
#    x_ind = np.arange(X.shape[1]) # Xpixels
#    y_ind = np.arange(X.shape[0]) # Yline
#    Xind,Yind = np.meshgrid(x_ind,y_ind)
#    Xp = geodata[0] + Xind*geodata[1] + Yind*geodata[2];
#    Yp = geodata[3] + Xind*geodata[4] + Yind*geodata[5];
#    sumofsquares=np.sum((X-Xp)**2+(Y-Yp)**2)
#    if sumofsquares>0:
#        print "Sum of squares indicates wrong geotransform: {}".format(sumofsquares)

    ds_out.SetGeoTransform(geodata)
    
    Vals_out = Vals.copy()
    Vals_out[np.isnan(Vals_out)]=nan_val

    if set_proj:
        # set the reference info
        srs = osr.SpatialReference()
        if isinstance(proj_wkt,(float,int)):
            srs.ImportFromEPSG(proj_wkt)
        elif '+' in proj_wkt:
            srs.ImportFromProj4(proj_wkt)
        elif 'PROJCS' in proj_wkt or 'GEOGCS' in proj_wkt:
            srs.ImportFromWkt(proj_wkt)
        elif hasattr(proj_wkt,'ImportFromWkt'):
            srs = proj_wkt # already an osr sr
        else:
            # Assume outproj is geographic sr
            srs.SetWellKnownGeogCS(proj_wkt) 
            
        ds_out.SetProjection(srs.ExportToWkt())

    # write the band    
    outband = ds_out.GetRasterBand(1)
    outband.SetNoDataValue(nan_val)
    outband.WriteArray(Vals_out)
    outband.FlushCache()
    ds_out,outband = None, None

def grid_rot(XY=None,val=None):
    '''Rotate matrixes to set origin at top left row,col=0,0.'''
    x,y=XY
    if val is not None:
        h=val.copy()
    else:
        h=None
    # Reorient matrixes based on xy orientation
    xmin_inds = np.unravel_index(np.argmin(x),x.shape)
    ymax_inds = np.unravel_index(np.argmax(y),y.shape)
    
    if xmin_inds[1]==x.shape[0]-1:
        # Flip column axis
        x,y = x[:,::-1],y[:,::-1]
        if val is not None:
            if len(h.shape)==3:
                h = h[:,:,::-1]
            else:
                h = h[:,::-1]

    if ymax_inds[0]!=0:
        # Flip row axis
        x,y = x[::-1,:],y[::-1,:]
        if val is not None:
            if len(h.shape)==3:
                h = h[:,::-1,:]
            else:
                h = h[::-1,:]

    # Calculate grid rotation
    grid_rot = np.arctan2(y[0,1]-y[0,0],x[0,1]-x[0,0])
    
    return [x,y],h,grid_rot

def calc_dxdy(XY=None,rot_xy=0,ndec=10):
    '''Calculate spatial discretization with rotation.'''
    tempX,tempY = XY
    if rot_xy==0:
        dx=tempX[0,1]-tempX[0,0]
        dy=tempY[1,0]-tempY[0,0]
    else:
        x0,y0 = tempX[0,0],tempY[0,0]
        newX = cgu.xrot(tempX-x0,tempY-y0,-rot_xy)
        newY = cgu.yrot(tempX-x0,tempY-y0,-rot_xy)
        dx = np.round(newX[0,1]-newX[0,0],decimals=ndec)
        dy = np.round(newY[1,0]-newY[0,0],decimals=ndec)
    return dx,dy
    
def raster_poly_clip(raster_path=None,in_poly=None,save_bool=False,
                     save_path=None,nodata=-9999.):
    '''Clip raster to polygon extent.
    
    
    '''
    # Load raster as a gdal image to get geotransform
    # (world file) info
    srcImage = gdal.Open(raster_path)
    geoTrans = srcImage.GetGeoTransform()
    srcImage = None
    
    in_dict={'raster_path':raster_path,'in_poly':in_poly,
                 'save_bool':save_bool,'save_path':save_path,
                 'nodata':nodata,'gt':geoTrans}
    
    # Convert the layer extent to image pixel coordinates
    if geoTrans[1]==0. and geoTrans[-1]==0.:
        # Grid is not rotated, use simple solution
        
        X,Y,clip_array,mask = clip_no_rotation(**in_dict)

    else:
        
        X,Y,clip_array,mask = clip_rotation(**in_dict)
        
    return X,Y,clip_array,mask

def clip_rotation(raster_path=None,in_poly=None,save_bool=False,
                     save_path=None,nodata=-9999.,gt=None):
    '''Clip raster with rotation.
    
    '''
    # Load polygon data
    polys,npolys = cfu.shp_to_polygon(in_poly) # handles conversions
    poly=polys[0] # only use first polygon found...plan accordingly
    minX, minY, maxX, maxY = poly.bounds 
    
    # Need to create a mask using full XY grid
    X,Y,srcArray = read_griddata(raster_path)
    inds_found = cfu.pt_in_shp(poly,[X,Y])
    all_inds = np.unravel_index(inds_found,X.shape)
    mask = np.zeros_like(X)
    mask[all_inds[0],all_inds[1]]=1
    mask[np.isnan(srcArray)] = np.nan
    
    ulX,ulY = np.min(all_inds[1]),np.min(all_inds[0])
    lrX,lrY = np.max(all_inds[1]),np.max(all_inds[0])
    
    # Check for multiband raster
    if len(srcArray.shape)==3:
        clip = srcArray[:, ulY:lrY+1, ulX:lrX+1]
    else:
        clip = srcArray[ulY:lrY+1, ulX:lrX+1]
    Xout = X[ulY:lrY+1, ulX:lrX+1]
    Yout = Y[ulY:lrY+1, ulX:lrX+1]
    
    if save_bool:
        proj = load_grid_prj(raster_path)
        write_gdaltif(save_path,Xout,Yout,clip,proj_wkt=proj,nan_val=nodata)
    
    return Xout,Yout,clip,mask
    
    
def clip_no_rotation(raster_path=None,in_poly=None,save_bool=False,
                     save_path=None,nodata=-9999.,save_jpg=False,gt=None):
    '''Clip raster without rotation.
    
    Source: after https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#clip-a-geotiff-with-shapefile
    and http://karthur.org/2015/clipping-rasters-in-python.html'''
    # Load polygon data
    polys,npolys = cfu.shp_to_polygon(in_poly) # handles conversions
    poly=polys[0] # only use first polygon found...plan accordingly
    shp_xy = np.array(poly.exterior.xy).T # extract polygon vertex positions
    minX, minY, maxX, maxY = poly.bounds  
    
    # Load the source data as a gdalnumeric array
    srcArray = gdalnumeric.LoadFile(raster_path)
    ulX, ulY = world2Pixel(gt, minX, maxY)
    lrX, lrY = world2Pixel(gt, maxX, minY)
    
    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
    
    # If the clipping features extend out-of-bounds and ABOVE the raster...
    if gt[3] < maxY:
        # In such a case... ulY ends up being negative--can't have that!
        iY = ulY
        ulY = 0 
    
    # Check for multiband raster
    if len(srcArray.shape)==3:
        clip = srcArray[:, ulY:lrY, ulX:lrX]
    else:
        clip = srcArray[ulY:lrY, ulX:lrX]
    
    # create pixel offset to pass to new image Projection info
    xoffset =  ulX
    yoffset =  ulY
    
    # Create a new geomatrix for the image
    geoTrans2 = list(gt)
    geoTrans2[0] = minX
    geoTrans2[3] = maxY
    
    # Convert polygon to raster using vertex indexes
    # Map points to pixels for drawing the
    # boundary on a blank 8-bit,
    # black and white, mask image.
    pixels = []
    for p in shp_xy:
      pixels.append(world2Pixel(geoTrans2, p[0], p[1]))
      
    rasterPoly = Image.new("L", (pxWidth, pxHeight), 1)
    rasterize = ImageDraw.Draw(rasterPoly)
    rasterize.polygon(pixels, 0)
    
    # If the clipping features extend out-of-bounds and ABOVE the raster...
    if gt[3] < maxY:
        # The clip features were "pushed down" to match the bounds of the
        #   raster; this step "pulls" them back up
        premask = imageToArray(rasterPoly)
        # We slice out the piece of our clip features that are "off the map"
        mask = np.ndarray((premask.shape[-2] - abs(iY), premask.shape[-1]), premask.dtype)
        mask[:] = premask[abs(iY):, :]
        mask.resize(premask.shape) # Then fill in from the bottom

        # Most importantly, push the clipped piece down
        geoTrans2[3] = maxY - (maxY - gt[3])

    else:    
        mask = imageToArray(rasterPoly)
    
    # Clip the image using the mask
    try:
        clip = gdalnumeric.choose(mask, (clip, nodata))

    # If the clipping features extend out-of-bounds and BELOW the raster...
    except ValueError:
        # We have to cut the clipping features to the raster!
        rshp = list(mask.shape)
        if mask.shape[-2] != clip.shape[-2]:
            rshp[0] = clip.shape[-2]

        if mask.shape[-1] != clip.shape[-1]:
            rshp[1] = clip.shape[-1]

        mask.resize(*rshp, refcheck=False)

        clip = gdalnumeric.choose(mask, (clip, nodata))

    if save_bool:
        # Save as new geotiff
        gtiffDriver = gdal.GetDriverByName( 'GTiff' )
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")
        gtiffDriver.CreateCopy(save_path,
            OpenArray(clip, prototype_ds=raster_path, xoff=xoffset, yoff=yoffset)
        )
        if save_jpg:
            # Save as an 8-bit jpeg for an easy, quick preview
            clip = clip.astype(gdalnumeric.uint8)
            gdalnumeric.SaveArray(clip, "{}.jpg".format(os.path.splitext(save_path)[0]),
                                  format="JPEG")
    
        gdal.ErrorReset()
        
    # calculate X,Y positions 
    Xp,Yp = np.meshgrid(np.arange(clip.shape[1]),np.arange(clip.shape[0]))        

    Xgeo = lambda gt,xpixel,yline: gt[0] + xpixel*gt[1] + yline*gt[2]
    Ygeo = lambda gt,xpixel,yline: gt[3] + xpixel*gt[4] + yline*gt[5]

    # Calculate for cell centers
    RX = Xgeo(geoTrans2,Xp,Yp) + geoTrans2[1]/2.
    RY = Ygeo(geoTrans2,Xp,Yp) + geoTrans2[5]/2.
    
    return RX,RY,clip,mask     

def subsection_griddata(orig_xy,orig_val,new_xy,nsections=20.,min_ndxy = 25.,
                        active_method='linear'):
    
    # Unpack inputs
    X_temp,Y_temp = orig_xy
    
    if isinstance(nsections,float) or isinstance(nsections,int):
        nsections = [np.float(nsections),np.float(nsections)] # convert to list
        
    
    # Set up subsection indexes
    ny,nx = new_xy[0].shape
    sections_dy,sections_dx = np.ceil(ny/nsections[1]),np.ceil(nx/nsections[0])
    
    # Want at least min_ndxy number of points per dimension in a subsection
    if sections_dy < min_ndxy:
        sections_dy = min_ndxy
    
    if sections_dx < min_ndxy:
        sections_dx = min_ndxy
        
    sstart_y,sstart_x = np.arange(0,ny,sections_dy,dtype=np.int),np.arange(0,nx,sections_dx,dtype=np.int)
    send_y,send_x = np.roll(sstart_y,-1),np.roll(sstart_x,-1)
    send_y[-1],send_x[-1] = ny,nx
    
    # Initiate output and loop
    try:
        val_mask = new_xy[0].mask # already masked
    except:
        # Make mask
        val_mask = np.isnan(new_xy[0]) # probably none unless previously set to nan
    
    new_val = val_mask*np.nan*np.zeros_like(new_xy[0])
        
    icount = -1
    
    # Loop
    for irow,(rowstart,rowend) in enumerate(zip(sstart_y,send_y)):
        for icol,(colstart,colend) in enumerate(zip(sstart_x,send_x)):
            icount+=1
            in_x = new_xy[0][rowstart:rowend,colstart:colend]
            in_y = new_xy[1][rowstart:rowend,colstart:colend]
            
            if in_y[~val_mask[rowstart:rowend,colstart:colend]].shape[0]==0 or \
                len(in_y)==0 or len(in_x)==0:
#                print '{},{} have no active cells'.format(irow,icol)
                continue
            else:
                
                temp_extent = [in_x.min(),in_x.max(),in_y.min(),in_y.max()]
                buffer0 = 3*np.abs(np.diff(X_temp,axis=1).mean()+1j*np.diff(Y_temp,axis=0).mean()) # mean diagonal for cells
                inpts = (X_temp<=temp_extent[1]+buffer0) & (X_temp>=temp_extent[0]-buffer0) \
                        & (Y_temp<=temp_extent[3]+buffer0) & (Y_temp>=temp_extent[2]-buffer0)
                if len(inpts.nonzero()[0])>0:
                    if len(np.unique(Y_temp[inpts]))>1 and len(np.unique(X_temp[inpts]))>1:
                        new_val_temp = griddata(np.c_[X_temp[inpts],Y_temp[inpts]],orig_val[inpts],(in_x,in_y),method=active_method)
                        new_val[rowstart:rowend,colstart:colend] = new_val_temp.copy()
                    else:
                        # Only one line of unique values, assign nan
#                        new_val_temp = griddata(np.c_[X_temp[inpts],Y_temp[inpts]],orig_val[inpts],(in_x,in_y),method='nearest')
                        new_val[rowstart:rowend,colstart:colend] = np.nan
    
    return new_val

def expand_raster_values(rast_in=None,XY=None,nfilt=3,
                         ncell_pad=50,extrapolate_values=True,
                         nan_val = 0):
    '''Assign missing value cells in raster to nearby values.'''
    
    if isinstance(rast_in,str):
        # Load raster text
        X,Y,rast_in = read_griddata(rast_in)
    else:
        X,Y = XY
    
    # Set nan_value to nan
    rast_in_temp = rast_in.copy()
    if nan_val is not None:
        rast_in_temp[rast_in_temp==nan_val] = np.nan
    
    # Create padded raster
    if ncell_pad > 0:
        dx,dy = np.mean(np.diff(X,axis=1)),np.mean(np.diff(Y,axis=0))
        x2 = np.arange(X[0,0]-ncell_pad*dx,X[0,-1]+(ncell_pad+1)*dx,dx)
        y2 = np.arange(Y[0,0]-ncell_pad*dy,Y[-1,0]+(ncell_pad+1)*dy,dy)
        X2,Y2 = np.meshgrid(x2,y2)
        new_rast = np.nan*np.ones_like(X2)
        new_rast[ncell_pad:-ncell_pad,ncell_pad:-ncell_pad] = rast_in_temp

    else:
        new_rast = rast_in_temp
        X2,Y2=X,Y
        
    # Apply nfilt x nfilt smoothing kernel to extend raster values first
    rast_nans = np.isnan(new_rast)
    g = Gaussian2DKernel(nfilt)
    rast_filt = convolve(new_rast,g,boundary='extend')
    rast_filt[~rast_nans] = new_rast.copy()[~rast_nans] # insert original values where possible
    
    # Extrapolate data to fill all missing values using nearest neighbor extrapolation
    if extrapolate_values:
        rast_nans2 = np.isnan(rast_filt)
        xy = [X2[~rast_nans2],Y2[~rast_nans2]]
        xy2 = (X2,Y2)
        rast_filled = subsection_griddata(xy,rast_filt[~rast_nans2],
                            xy2,active_method='nearest')
        rast_filled[~rast_nans2] = rast_filt.copy()[~rast_nans2] 
    else:
        rast_filled = rast_filt
    
    return [X2,Y2], rast_filled
    
    
def raster_edge(cell_types=None,search_val=-2,invalid_val=0,
                size=3,zsize=20,min_area=None,bool_array=None):
    '''Define a raster edge with additional boolean array.'''
#    mask = np.isnan(Z)
#    X2,Y2 = XY[0].copy(),XY[1].copy()
#    X2[mask],Y2[mask] = np.nan,np.nan # remove XY entries outside of raster data
    edge_bool=minimum_filter(np.abs(cell_types),size=size,mode='nearest') == invalid_val
    
    #Z2 = -np.ma.masked_invalid(Z) # mask and convert depth
    
    # Find where min_Z is the invalid value but original was search_val
    if search_val == -2:
        # Find cells far from land
        offshore_bool = minimum_filter(-cell_types,size=zsize,mode='nearest') >= invalid_val
        bool_out = edge_bool & (cell_types==search_val) & offshore_bool
    elif search_val == 1:
        bool_out = edge_bool & (cell_types==search_val)
    else:
        # find any boundary
        bool_out = edge_bool & (cell_types != invalid_val)

    if bool_array is not None:
        bool_out = bool_out & bool_array
    # Select longest continuous selection
        
    # Distinguish disconnected clusters of active cells in the IBOUND array.
    cluster_array = bool_out.copy().astype(np.int)
    
    array_of_cluster_idx,num = measurements.label(cluster_array)
    
    # Identify the cluster with the most active cells; this is the main active area
    areas = measurements.sum(cluster_array,array_of_cluster_idx,\
                             index=np.arange(array_of_cluster_idx.max()+1))
    
    clean_bool_array = np.zeros_like(bool_out)                         
    if (min_area is None):
        # Use only largest area
        cluster_idx = np.argmax(areas)
        # Activate all cells that belong to primary clusters
        clean_bool_array[array_of_cluster_idx == cluster_idx] = 1
    else:
        cluster_idx = (areas >= min_area).nonzero()[0]
        # Activate all cells that belong to primary clusters
        for idx_active in cluster_idx:
            clean_bool_array[array_of_cluster_idx==idx_active] = 1    

    return clean_bool_array.astype(bool)
    
def cluster_array(in_array=None,min_area=None,mult_val=1.):
    # Distinguish disconnected clusters of active cells in the array.
    cluster_array = (mult_val*in_array).copy().astype(np.int)
    
    array_of_cluster_idx,num = measurements.label(cluster_array)
    
    # Identify the cluster with the most active cells; this is the main active area
    areas = measurements.sum(cluster_array,array_of_cluster_idx,\
                             index=np.arange(array_of_cluster_idx.max()+1))
    
    clean_array = np.zeros_like(in_array)                         
    if (min_area is None):
        # Use only largest area
        cluster_idx = np.argmax(areas)
        # Activate all cells that belong to primary clusters
        clean_array[array_of_cluster_idx == cluster_idx] = 1
    else:
        cluster_idx = (areas >= min_area).nonzero()[0]
        # Activate all cells that belong to primary clusters
        for idx_active in cluster_idx:
            clean_array[array_of_cluster_idx==idx_active] = 1    
    return clean_array
    
def raster_outline(X,Y,Z):
    X,Y,Z = remove_nan_rc(X,Y,Z)
    mask = np.isnan(Z)
    X2 = X.copy()
    X2[mask] = np.nan
    Y2 = Y.copy()
    Y2[mask] = np.nan
    xleft = np.nanmin(X2,axis=1)
    xright = np.nanmax(X2,axis=1)
    ytop = np.nanmax(Y2,axis=0)
    ybot = np.nanmin(Y2,axis=0)
    
    # Collect nans
    xl_nans = np.isnan(xleft)
    xr_nans = np.isnan(xright)
    yt_nans = np.isnan(ytop)
    yb_nans = np.isnan(ybot)
    
    pts = np.vstack([np.c_[X[0,:].ravel()[~yt_nans],ytop[~yt_nans]],
                     np.c_[xright[~xr_nans],Y[:,0].ravel()[~xr_nans]],
                     np.c_[X[0,~yb_nans].ravel()[::-1],ybot[~yb_nans][::-1]],
                     np.c_[xleft[~xl_nans][::-1],Y[~xl_nans,0][::-1].ravel()]])
          
    return pts
    
def remove_nan_rc(X,Y,Z,return_mask=False):
    '''Remove columns and rows with only null values.
    '''
    # Conslidate masks
    mask = np.isnan(Z)
    if hasattr(X,'mask'):
        X2 = np.ma.getdata(X.copy())
    else:
        X2 = X.copy()
    X2[mask] = np.nan
    if hasattr(Y,'mask'):
       Y2 = np.ma.getdata(Y.copy())
    else:
       Y2 = Y.copy()
    
#    Y2[mask] = np.nan
    X2 = np.ma.masked_array(X2,mask=mask)
    Y2 = np.ma.masked_array(Y2,mask=mask)
    xleft = np.nanmin(X2,axis=1)
#    xleft[np.isnan(xleft)] = np.nanmax(xleft)
    xright = np.nanmax(X2,axis=1)
#    xright[np.isnan(xright)] = np.nanmin(xright)
    ytop = np.nanmax(Y2,axis=0)
#    ytop[np.isnan(ytop)] = np.nanmin(ytop)
    ybot = np.nanmin(Y2,axis=0)
#    ybot[np.isnan(ybot)] = np.nanmax(ybot)
    
    # Find first and last indices to keep
    x0=(xleft.min()==X2[np.argmin(xleft),:]).nonzero()[0][0]
    x1=(xright.max()==X2[np.argmax(xright),:]).nonzero()[0][0]
    y0=(ybot.min()==Y2[:,np.argmin(ybot)]).nonzero()[0][0]
    y1=(ytop.max()==Y2[:,np.argmax(ytop)]).nonzero()[0][0]    
    X2,Y2 = [],[]
    
    x0,x1 = np.sort([x0,x1])
    y0,y1 = np.sort([y0,y1])
    if return_mask:
        mask_out = np.zeros_like(X,dtype=bool)
        mask_out[y0:y1+1,x0:x1+1] = 1
        return X.copy()[y0:y1+1,x0:x1+1],Y.copy()[y0:y1+1,x0:x1+1],Z.copy()[y0:y1+1,x0:x1+1],mask_out
    else:
        return X.copy()[y0:y1+1,x0:x1+1],Y.copy()[y0:y1+1,x0:x1+1],Z.copy()[y0:y1+1,x0:x1+1]
    
    
    

def bindata2d(XY_orig,Z_orig,XY_new,stat_func=np.median):
    X,Y = XY_new    
    dx,dy = X[0,1]-X[0,0],Y[1,0]-Y[0,0]
    xbins = np.hstack([X[0,0]-dx/2.,X[0,:]+dx/2.])
    ybins = np.hstack([Y[0,0]-dy/2.,Y[:,0]+dy/2.])
    nan_inds = np.isnan(Z_orig)
    Z_new = binned_statistic_2d(XY_orig[1][~nan_inds],XY_orig[0][~nan_inds],values=Z_orig[~nan_inds], statistic=stat_func, bins=[ybins,xbins])
    count_mask = Z_new.statistic==0.
    Z_new.statistic[count_mask] = np.nan
    return Z_new.statistic

def xy_from_affine(tform=None,nx=None,ny=None):
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
    return X,Y

def gdal_loadgrid(fname,new_xy,in_extent=None,
                  interp_method='linear',ideal_cell_size=None,
                  crs=None):
    
    if interp_method.lower() in ['linear']:
        resampling = Resampling.bilinear
    elif interp_method.lower() in ['nearest']:
        resampling = Resampling.nearest
    elif interp_method.lower() in ['spline','cubic_spline']:
        resampling = Resampling.cubic_spline
    else:
        resampling = Resampling.bilinear # default to bilinear
    
    height,width = new_xy[0].shape
    xy2,_,rot = grid_rot(new_xy) # calculate rotation of grid
    
    if ideal_cell_size is None:
        xres,yres = calc_dxdy(new_xy,rot_xy=rot,ndec=2)
    else:
        xres,yres = ideal_cell_size,ideal_cell_size
    
    X,Y = new_xy
    geodata = [X[0,0]-np.cos(-rot)*xres/2.+np.sin(-rot)*xres/2.,
               np.cos(-rot)*xres,
               -np.sin(-rot)*xres,
               Y[0,0]-np.cos(-rot)*yres/2.-np.sin(-rot)*yres/2.,
               np.sin(-rot)*yres,
               np.cos(-rot)*yres] 
    
    transform = affine.Affine.from_gdal(*geodata)
    
#    transform = affine.Affine(xres,rot,new_xy[0][0,0],
#                              rot,yres,new_xy[1][0,0])
    
    
    vrt_options = {'resampling': resampling,
                   'tolerance':0.01,'all_touched':True,'num_threads':4,
                   'sample_grid':'YES','sample_steps':100,
                   'source_extra':10,
                   'transform':transform,
                   'height':height,'width':width}
    
    with rasterio.open(fname,'r') as src:
                
        if crs is None:
            # Use input raster crs
            vrt_options.update({'crs':src.crs})
        elif isinstance(crs,(float,int)):
            vrt_options.update({'crs':CRS.from_epsg(crs)})
        else:
            vrt_options.update({'crs':CRS.from_wkt(crs)})
        
#        transform,width,height = calculate_default_transform(src.crs,vrt_options['crs'],src.width,src.height,*src.bounds,
#                                                             dst_width=width,dst_height=height)
#        vrt_options.update({'transform':transform,
#                            'width':width,
#                            'height':height})
        
        with WarpedVRT(src,**vrt_options) as vrt:
            out_rast = vrt.read()[0]
            out_rast[out_rast == vrt.nodata] = np.nan
#            X,Y = xy_from_affine(transform,width,height)
    
    return out_rast
                  
                  
        
def load_and_griddata(fname,new_xy,in_extent=None,mask=None,
                      interp_method = 'linear',ideal_cell_size=None):
    '''
    Load raster dataset (fname) and re-grid to new raster cells specified by new_xy
    
    interp_method: 'linear': uses bilinear grid interpoloation
                   'median': uses median filter, when cell_spacing_orig << cell_spacing_new
                   other: can assign bindata2d function (e.g., np.median, np.std, np.mean) 
                   
    '''                      
    X_temp,Y_temp,Grid_val = read_griddata(fname,in_extent=in_extent)
    
    # Decimate grid to lower resolution from ultrahigh res dataset
    if ideal_cell_size is not None:
        X_temp,Y_temp,Grid_val = decimate_raster(X_temp,Y_temp,Grid_val,
                                                 ideal_cell_size=ideal_cell_size)
    if interp_method.lower() in ('linear','bilinear'):
        if (mask is not None):
            new_xy[0] = np.ma.masked_array(new_xy[0],mask=mask)
            new_xy[1] = np.ma.masked_array(new_xy[1],mask=mask)   
                 
        out_grid = subsection_griddata([X_temp,Y_temp],Grid_val,new_xy)
    elif interp_method in ('median'):
        out_grid = bindata2d([X_temp,Y_temp],Grid_val,new_xy)
    else:
        out_grid = bindata2d([X_temp,Y_temp],Grid_val,new_xy,stat_func=interp_method)
    return out_grid

def decimate_raster(X_temp,Y_temp,Grid_val,ideal_cell_size=None,ndecimate_in=None):
    if ndecimate_in is not None and ndecimate_in>1:
        X_temp = X_temp[::ndecimate_in,::ndecimate_in]
        Y_temp = Y_temp[::ndecimate_in,::ndecimate_in]
        Grid_val = Grid_val[::ndecimate_in,::ndecimate_in]
    
    max_grid_dxy = np.max([np.abs(np.diff(X_temp,axis=1).mean()),np.abs(np.diff(Y_temp,axis=0).mean())])
    if ideal_cell_size is not None:
        cell_size_ratio = ideal_cell_size/max_grid_dxy
        if cell_size_ratio > 5:
            rows,cols = X_temp.shape
            ntimes_gt_ideal = 3
            ndecimate = np.int(np.floor(cell_size_ratio)/ntimes_gt_ideal)
            X2 = X_temp[::ndecimate,::ndecimate]
            Y2 = Y_temp[::ndecimate,::ndecimate]
            Grid_val2 = Grid_val[::ndecimate,::ndecimate]
            # check to see if last entries of arrays are the same (i.e., don't lose boundary values)
            xfix_switch = False
            if X2[0,-1] != X_temp[0,-1]:
                # Add last column into array
                X2 = np.hstack([X2,X_temp[::ndecimate,-1].reshape((-1,1))])
                Y2 = np.hstack([Y2,Y_temp[::ndecimate,-1].reshape((-1,1))])
                Grid_val2 = np.hstack([Grid_val2,Grid_val[::ndecimate,-1].reshape((-1,1))])
                xfix_switch = True
            
            if Y2[-1,0] != Y_temp[-1,0]:
                # Add last row to array
                if xfix_switch:
                    print(X2.shape,np.hstack([X_temp[-1,::ndecimate],X_temp[-1,-1]]).reshape((1,-1)).shape)
                    X2 = np.vstack([X2,np.hstack([X_temp[-1,::ndecimate],X_temp[-1,-1]]).reshape((1,-1))])
                    Y2 = np.vstack([Y2,np.hstack([Y_temp[-1,::ndecimate],Y_temp[-1,-1]]).reshape((1,-1))])
                    Grid_val2 = np.vstack([Grid_val2,np.hstack([Grid_val[-1,::ndecimate],Grid_val[-1,-1]]).reshape((1,-1))])
                else:
                    X2 = np.vstack([X2,X_temp[-1,::ndecimate].reshape((1,-1))])
                    Y2 = np.vstack([Y2,Y_temp[-1,::ndecimate].reshape((1,-1))])
                    Grid_val2 = np.vstack([Grid_val2,Grid_val[-1,::ndecimate].reshape((1,-1))])
            
            X_temp,Y_temp,Grid_val = X2,Y2,Grid_val2
    
        
    return X_temp, Y_temp, Grid_val

def define_mask(shp=None,rast_template=None,
              bands=[1],options=['ALL_TOUCHED=TRUE'],burn_values=[1],
              trans_arg1=None,trans_arg2=None,nan_val=0,out_bool=True):
    '''
    If this doesn't work, try rasterio.mask.mask
    '''
    if isinstance(shp,str):
        shp_ds = ogr.Open(shp)
        shp_layer = shp_ds.GetLayer()
    elif hasattr(shp,'geom_type'):
        # Convert from pyshp to ogr
        shp_layer = ogr.CreateGeometryFromWkt(shp.to_wkt())
    else:
        shp_layer = shp
    
    if isinstance(rast_template,str):
        nrows,ncols,gd = get_rast_info(rast_template)
    else:
        nrows,ncols,gd = rast_template
    
    dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Int32)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(nan_val) #initialise raster with nans
    dst_rb.SetNoDataValue(nan_val)
    dst_ds.SetGeoTransform(gd)

    _ = gdal.RasterizeLayer(dst_ds,bands,shp_layer,
                        trans_arg1,trans_arg2,burn_values=burn_values,
                        options=options)

    dst_ds.FlushCache()
    
    mask_arr=dst_ds.GetRasterBand(1).ReadAsArray()
    
    if out_bool:
        return mask_arr.astype(bool)
    else:
        return mask_arr

def read_hk(k_fnames, griddata_dict=None, load_last_botm=True,load_vk=False):
    '''Read hydraulic conductivity and bottom elevation rasters
    
    Parameters
    ----------
    
    k_fnames: list
        list of nlayer lists of [k_value_layer.tif,k_bottom_elev.tif]
    
    Returns
    --------
    
    hk_array: np.ndarray
        nlay x nrow x ncol array of hk values
            
    hk_botm_array: np.ndarray
        nlay x nrow x ncol array of hk layer bottom elevations
    '''
    hk_list = []
    botm_list = []
    vk_list = []
    for ilay,k_fname in enumerate(k_fnames):
        hk_list.append(gdal_loadgrid(k_fname[0],**griddata_dict))

        if load_vk and len(k_fname)==3:
            vk_list.append(gdal_loadgrid(k_fname[2],**griddata_dict))
        
        if k_fname[1] is not None:
            if ilay != len(k_fnames)-1 or load_last_botm:
                botm_list.append(gdal_loadgrid(k_fname[1],**griddata_dict))
                
    return np.array(hk_list),np.array(vk_list),np.array(botm_list)
        
def make_layers_deeper(in_array,deeper_amount=1.):
    out_array = in_array.copy()
    for ilay in range(out_array.shape[0]-1):
        deep_or_thin_bool = (out_array[ilay]-out_array[ilay+1]) < deeper_amount
        out_array[ilay+1,deep_or_thin_bool] = out_array[ilay,deep_or_thin_bool]-deeper_amount
        
    return out_array
    
    
def smooth_array(in_array,nfilt=3,boundary='extend'):
    from astropy.convolution import Gaussian2DKernel, convolve
    
    g = Gaussian2DKernel(nfilt)
    if isinstance(in_array,np.ndarray):
        if len(in_array.squeeze().shape)>2:
            out_array = np.nan*np.ones_like(in_array)
            for ilay in range(out_array.shape[0]):
                out_array[ilay] = convolve(in_array[ilay],g,boundary=boundary)
        else:
            in_shape = in_array.shape
            out_array = convolve(in_array.squeeze(),g,boundary=boundary)
            out_array = out_array.reshape(in_shape) # preserve dimensionality of array
            
    elif isinstance(in_array,(list,tuple)):
        out_array = []
        for in_temp in in_array:
            out_array.append(convolve(in_temp,g,boundary=boundary))
        
    return out_array
        
def save_txtgrid(fname=None,data=None,delimiter=',',header=None):
    with open(fname,'w') as f_out:
        if (header is not None):
            f_out.write(header)
        for data_line in data:
            f_out.write('{}\n'.format(delimiter.join(data_line.astype('|S'))))
        f_out.close()

def read_txtgrid(fname=None,delimiter=',',comment='#'):
    with open(fname,'r') as f_in:
        load_data = []
        header_info = []
        for iline in f_in:
            if iline[0] in [comment]:
                header_info.append(iline.strip('\n'))
            else:
                pieces = iline.split(delimiter)
                try:
                    idata = [int(piece) for piece in pieces]
                except:
                    try:
                        idata = [float(piece) for piece in pieces]
                    except:
                        idata = pieces
                load_data.append(idata)
                
        f_in.close()
    return load_data,header_info

def save_nc(fname=None,out_data_dict=None,out_desc=None):
    
    
    nc_out = netCDF4.Dataset(fname,'w', format='NETCDF4')
    if 'out_desc' not in out_data_dict.keys():
        nc_out.description = r'No description'
    else:
        nc_out.description = out_data_dict['out_desc']
    
    # Assign dimensions
    for dim in out_data_dict['dims']['dim_order']:
        nc_out.createDimension(dim,out_data_dict['dims'][dim]['data'].size)
        dim_var = nc_out.createVariable(dim,'f8',(dim,),zlib=True)
        dim_var.setncatts(out_data_dict['dims'][dim]['attr'])
        nc_out.variables[dim][:] = out_data_dict['dims'][dim]['data']
    
    # Assign data arrays
    for ikey in out_data_dict['vars']:
        data_var = nc_out.createVariable(ikey,'f8',out_data_dict[ikey]['dims'],zlib=True)
        data_var.setncatts(out_data_dict[ikey]['attr'])        
        nc_out.variables[ikey][:]=  out_data_dict[ikey]['data']
    
    nc_out.close()
        
def load_nc(fname=None):

    out_dict = {}
    f = netCDF4.Dataset(fname)
    for var_name in f.variables.keys():
        out_dict.update({var_name:{'data':f.variables[var_name][:],
                                   'long_name':f.variables[var_name].long_name,
                                   'var_desc':f.variables[var_name].var_desc,
                                   'units':f.variables[var_name].units}})
    
    return out_dict


def landfel_taudem(work_dir=None,xy=None,elev_data=None,
                   proj=None,n_proc=8,sea_level=0.,dem_fmt = '{}_dem.tif',
                   fel_fmt='{}_dem_fel.tif',
                   landfel_fmt = '{}_dem_landfel.tif'):
    '''Use TauDEM to fill dem.'''
    fname = os.path.dirname(work_dir)
    in_dem = dem_fmt.format(fname)
    in_fname = os.path.join(work_dir,in_dem)
    filled_dem = fel_fmt.format(fname)  
    landfel_dem = landfel_fmt.format(fname)
    landfel_fname = os.path.join(work_dir,landfel_dem)
    
    if not os.path.isfile(landfel_fname):
        # Save in_dem (unfilled)
        if not os.path.isfile(in_fname):
            write_gdaltif(in_fname,xy[0],xy[1],elev_data,proj_wkt=proj)
        
        # 1) Run pit removal (i.e., fill sinks)
        premove_dict ={'input_dem':in_dem,'filled_dem':filled_dem,'work_dir':work_dir,
                       'n_proc':n_proc}
        pit_out = pit_remove(**premove_dict)
        
        if elev_data is None:
            X,Y,elev_data = read_griddata(in_dem)
        
        X,Y,elev_fel = read_griddata(os.path.join(work_dir,filled_dem))
        
        # Assign filled areas only to land   
        elev_bool = elev_data<=sea_level
        elev_fel[elev_bool] = elev_data[elev_bool]
        
        write_gdaltif(landfel_fname,X,Y,elev_fel,proj_wkt=proj)  
    else:
        _,_,elev_fel = read_griddata(landfel_fname)
        
    return elev_fel

TauDEM_dir = r"C:\research\programs\TauDEM537exeWin64"
exe_dict = {'mpiexec':r"C:\Program Files\Microsoft MPI\Bin\mpiexec",
            'pit_remove':os.path.join(TauDEM_dir,"PitRemove"),
            'Dinf_FlowDir':os.path.join(TauDEM_dir,"DinfFlowDir"),
            'Dinf_ContributingArea':os.path.join(TauDEM_dir,"AreaDinf"),
            }
        
def pit_remove(input_dem=None,filled_dem=None,work_dir=None,n_proc=None):
    
    from subprocess import Popen, PIPE, STDOUT
    if work_dir is not None:
        # Add work directory path to filename
        input_dem = os.path.join(work_dir,input_dem)
        filled_dem = os.path.join(work_dir,filled_dem)

    run_command = '"{0}" -n "{1}" "{2}" -z \"{3}\\" -fel \"\"{4}\"'.format(exe_dict['mpiexec'],
                                                                           n_proc,
                                                                           exe_dict['pit_remove'],
                                                                           input_dem,
                                                                           filled_dem)
    print(run_command)
    shell_bool=True
    process = Popen(run_command,shell=shell_bool,stdout=PIPE,stderr=STDOUT,universal_newlines=True)
    out,err = process.communicate() 
#    
    return out,err        

# ------ Functions from python gdal cookbook -----------
# Source:https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html#clip-a-geotiff-with-shapefile

def imageToArray(i):
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a=gdalnumeric.fromstring(i.tobytes(),'b')
    a.shape=i.im.size[1], i.im.size[0]
    return a

def arrayToImage(a):
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i=Image.fromstring('L',(a.shape[1],a.shape[0]),
            (a.astype('b')).tostring())
    return i

def world2Pixel(geoMatrix, x, y):
  """
  Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
  the pixel location of a geospatial coordinate
  """
  ulX = geoMatrix[0]
  ulY = geoMatrix[3]
  xDist = geoMatrix[1]
  yDist = geoMatrix[5]
  rtnX = geoMatrix[2]
  rtnY = geoMatrix[4]
  pixel = int((x - ulX) / xDist)
  line = int((ulY - y) / xDist)
  return (pixel, line)

#
#  EDIT: this is basically an overloaded
#  version of the gdal_array.OpenArray passing in xoff, yoff explicitly
#  so we can pass these params off to CopyDatasetInfo
#
def OpenArray( array, prototype_ds = None, xoff=0, yoff=0 ):
    ds = gdal.Open( gdalnumeric.GetArrayFilename(array) )

    if ds is not None and prototype_ds is not None:
        if type(prototype_ds).__name__ == 'str':
            prototype_ds = gdal.Open( prototype_ds )
        if prototype_ds is not None:
            gdalnumeric.CopyDatasetInfo( prototype_ds, ds, xoff=xoff, yoff=yoff )
    return ds

def merge_tifs(rast1=None,rast2=None,intersect_poly=None,
               domain1=None,domain2=None,ibuff=100.,
               wt_fmt='{}_wtdepth.tif',head_fmt='{}_head.tif',
               cell_spacing=10.,marine_mask_val = -500.,
               dstep=10,force_new=[False]*2,small_value=0,extend_m=1e3,
               erf_params = {'mid_loc':0.5,'scale':5}):
    
    blend_func = lambda x: (erf((x-erf_params['mid_loc'])*erf_params['scale'])+1)/2.
    
    nonnan_mat1 = np.array([False])
    # Make filenames
    fname1 = os.path.splitext(rast1)[0][:-5]
    fname2 = os.path.splitext(rast2)[0][:-5]
    
    new_wt_fname1 = wt_fmt.format('_'.join([fname1,'merged']))
    new_wt_fname2 = wt_fmt.format('_'.join([fname2,'merged']))
    new_head_fname1=head_fmt.format('_'.join([fname1,'merged']))
    new_head_fname2=head_fmt.format('_'.join([fname2,'merged']))

    print(" {} to {}".format(os.path.basename(fname1),os.path.basename(fname2)))
    
    # Check to see if merged head tif's already made, use those instead 
    if not force_new[0]:
        if os.path.isfile(new_head_fname1):
            rast1 = new_head_fname1
    if not force_new[1]:    
        if os.path.isfile(new_head_fname2):
            rast2 = new_head_fname2

    # Make line features for borders of models
    d1_xy = np.array(domain1.exterior.xy).T
    d2_xy = np.array(domain2.exterior.xy).T
    int_xy = np.array(intersect_poly.exterior.xy).T

    d1_inds=cgu.unique_rows(np.vstack([d1_xy,int_xy])) # use domain2 to find line closer to domain1
    d2_inds=cgu.unique_rows(np.vstack([d2_xy,int_xy])) # vice versa
    
    d1_line_inds = d1_inds[d1_inds>len(d1_xy)]-len(d1_xy)
    d2_line_inds = d2_inds[d2_inds>len(d2_xy)]-len(d2_xy)
    
    d1_line_inds2 = np.array([i for i in d1_line_inds if i not in d2_line_inds])
    d2_line_inds2 = np.array([i for i in d2_line_inds if i not in d1_line_inds])
    
    d1_line_inds = d1_line_inds2.copy()
    d2_line_inds = d2_line_inds2.copy()
    line_length = len(d1_line_inds)+len(d2_line_inds)
                
    if line_length > 0:
        # And don't let indexes be discontinuous
        diff1 = np.diff(d1_line_inds)
        diff2 = np.diff(d2_line_inds)
        
        step1ind1 = (diff1>1).nonzero()[0]
        step1ind2 = (diff2>1).nonzero()[0]
        # Take maximum length
        if len(step1ind1)>0:
            if step1ind1[0]>len(diff1)-step1ind1[-1]:
                # Use first arm of data
                d1_line_inds2 = d1_line_inds[:step1ind1[0]]
            else:
                # Use second arm of data
                d1_line_inds2 = d1_line_inds[step1ind1[-1]:]
        else:
            d1_line_inds2=d1_line_inds
            
        if len(step1ind2)>0:
            if step1ind2[0]>len(diff2)-step1ind2[-1]:
                # Use first arm of data
                d2_line_inds2 = d2_line_inds[:step1ind2[0]]
            else:
                # Use second arm of data
                d2_line_inds2 = d2_line_inds[step1ind2[-1]:]
        else:
            d2_line_inds2=d2_line_inds
        
        d1_line_xy = int_xy[d1_line_inds2,:]
        d2_line_xy = int_xy[d2_line_inds2,:]
        
        # Use shortest edge (in case one model encompassed by another)
        if len(d1_line_xy)<len(d2_line_xy):
            dist_line_xy = d1_line_xy
            # Need to also include way to set first/second dataset
            # for the blending direction
            blend_switch = False
        else:
            dist_line_xy = d2_line_xy
            blend_switch = True
        
        # Extend boundary line used for caclulating distance from boundary
        if extend_m > 0:
            last_n_pts = 10
            if dist_line_xy.shape[0]<=last_n_pts:
                last_n_pts = dist_line_xy.shape[0]-1
                
            end_vect1 = dist_line_xy[last_n_pts]-dist_line_xy[0]
            end_vect2 = dist_line_xy[-last_n_pts]-dist_line_xy[-1]
            end_vect1_norm = end_vect1/np.linalg.norm(end_vect1)
            end_vect2_norm = end_vect2/np.linalg.norm(end_vect2)
            # add new points to beginning and end of line
            dist_line_xy = np.vstack([dist_line_xy[0]+[-end_vect1_norm*extend_m],dist_line_xy,dist_line_xy[-1]+[-end_vect2_norm*extend_m]])
     
        # Can apply an interior buffer for the intersection polygon area
        ipoly = intersect_poly.buffer(ibuff,cap_style=1,resolution=1,join_style=2)
    
    	# Extract overlapping area of models
        x1,y1,h1,m1 = raster_poly_clip(rast1,ipoly)
        x2,y2,h2,m2 = raster_poly_clip(rast2,ipoly)
    
        # Make new grid extents
        minx1,miny1,maxx1,maxy1=cgu.get_extent([x1,y1])
        minx2,miny2,maxx2,maxy2=cgu.get_extent([x2,y2])
        max_extent = [np.min([minx1,minx2]),np.min([miny1,miny2]),
                      np.max([maxx1,maxx2]),np.max([maxy1,maxy2])]
    
        # Make new spatial grid
        new_x=np.arange(max_extent[0],max_extent[2]+cell_spacing,cell_spacing)
        new_y=np.arange(max_extent[1],max_extent[3]+cell_spacing,cell_spacing)
        newX,newY = np.meshgrid(new_x,new_y)
    
        # Set marine values to nan
        h1[h1==marine_mask_val] = np.nan
        h2[h2==marine_mask_val] = np.nan
    

    
#    h1_interp=h1_orig[m1==1].copy()
#    h1_interp[h1_interp==marine_mask_val] = np.nan
#    
#    h2_interp=h2_orig[m2==1].copy()
#    h2_interp[h2_interp==marine_mask_val] = np.nan
#    
#    # Interpolate model results to new grid
#    new_h1 = cru.griddata(np.c_[x1_orig[m1==1][~np.isnan(h1_interp)],
#                                y1_orig[m1==1][~np.isnan(h1_interp)]],
#                                h1_interp[~np.isnan(h1_interp)],(newX,newY))
#    new_h2 = cru.griddata(np.c_[x2_orig[m2==1][~np.isnan(h2_interp)],
#                                y2_orig[m2==1][~np.isnan(h2_interp)]],
#                                h2_interp[~np.isnan(h2_interp)],(newX,newY))
        # Interpolate model results to new grid
        new_h1 = griddata(np.c_[x1[~np.isnan(h1)],y1[~np.isnan(h1)]],
                                    h1[~np.isnan(h1)],(newX,newY))
        new_h2 = griddata(np.c_[x2[~np.isnan(h2)],y2[~np.isnan(h2)]],
                                    h2[~np.isnan(h2)],(newX,newY))    
        new_h1[new_h1<small_value] = np.nan
        new_h2[new_h2<small_value] = np.nan
                   
        # Find domain centroids
    #    c1x,c1y = np.hstack(domain1.centroid.xy).tolist()
        #c2x,c2y = np.hstack(domain2.centroid.xy).tolist()
    
        # Calculate distance from overlapping grid locations to each domain centroid
        nonnan_mat1 = ~np.isnan(new_h1) & ~np.isnan(new_h2)
    
    # Load original rasters
    x1_orig,y1_orig,h1_orig = read_griddata(rast1)
    x2_orig,y2_orig,h2_orig = read_griddata(rast2)
    if len(nonnan_mat1.nonzero()[0])>0 and line_length>0:
        dmat1a = cgu.calc_dist(dist_line_xy[::10],list(zip(newX[nonnan_mat1].ravel(),newY[nonnan_mat1].ravel())))
        dmat1a = np.min(dmat1a,axis=0) # find closest values
        
#        dmat1a = cgu.calc_dist([[c1x,c1y]],list(zip(newX[nonnan_mat1].ravel(),newY[nonnan_mat1].ravel())))
        #dmat1b = cgu.calc_dist([[c2x,c2y]],list(zip(newX[nonnan_mat1].ravel(),newY[nonnan_mat1].ravel())))
    
        dmat1a_norm = dmat1a/np.mean(dmat1a)
        dmat1a_norm = 1.-(dmat1a_norm-np.min(dmat1a_norm))/(np.max(dmat1a_norm)-np.min(dmat1a_norm))
        #dmat1b_norm= dmat1b/np.mean(dmat1b)
        #dmat1b_norm = 1.-(dmat1b_norm-np.min(dmat1b_norm))/(np.max(dmat1b_norm)-np.min(dmat1b_norm))
    
        dmat1a_mat = np.nan*np.zeros_like(newX)
        #dmat1b_mat = np.nan*np.zeros_like(newX)
        dmat1a_mat[nonnan_mat1] = dmat1a_norm.ravel()
        
        #Extrapolate weights by nearest neighbor
        dmatnans = np.isnan(dmat1a_mat)   
        xy = np.c_[newX[~dmatnans],newY[~dmatnans]]
        xy2 = (newX,newY)
        dmat2 = griddata(xy,dmat1a_mat[~dmatnans],
                            xy2,method='nearest')
        dmat2[~dmatnans] = dmat1a_mat.copy()[~dmatnans]  
    
        # Use error function with center at 0.5
        blend_m1 = np.round(blend_func(dmat2),decimals=5)
        blend_m1[(new_h1<0.) | (new_h2<0.)] = np.nan
        if not blend_switch:
            out_h = blend_m1*new_h1 + (1.-blend_m1)*new_h2
        else:
            out_h = blend_m1*new_h2 + (1.-blend_m1)*new_h1
        out_h[np.isnan(out_h) & ~np.isnan(new_h1)] = new_h1[np.isnan(out_h) & ~np.isnan(new_h1)]
        out_h[np.isnan(out_h) & ~np.isnan(new_h2)] = new_h2[np.isnan(out_h) & ~np.isnan(new_h2)]
    
        # --- Insert merged data into each head array ---
    	# Extract x,y values of original model results using mask
        xtemp,ytemp = x1_orig[m1==1].reshape((-1,1)),y1_orig[m1==1].reshape((-1,1))
        xtemp2,ytemp2 = x2_orig[m2==1].reshape((-1,1)),y2_orig[m2==1].reshape((-1,1))
    
    	# Interpolate from merged heads back to original model grid
        htemp = subsection_griddata([newX,newY],out_h,(xtemp,ytemp),nsections=1)
        htemp2 = subsection_griddata([newX,newY],out_h,(xtemp2,ytemp2),nsections=1)
    
        # Insert new values into original model grid
        new_h3 = h1_orig.copy()
        new_h3[m1==1] = htemp.ravel()
        new_h3[np.isnan(new_h3) & ~np.isnan(h1_orig)] = h1_orig[np.isnan(new_h3) & ~np.isnan(h1_orig)]
        new_h3[h1_orig==marine_mask_val] = marine_mask_val
    
        new_h4 = h2_orig.copy()
        new_h4[m2==1] = htemp2.ravel()
        new_h4[np.isnan(new_h4) & ~np.isnan(h2_orig)] = h2_orig[np.isnan(new_h4) & ~np.isnan(h2_orig)]
        new_h4[h2_orig==marine_mask_val] = marine_mask_val
    else:
        # Use original data with no merging
        new_h3 = h1_orig.copy()
        new_h4 = h2_orig.copy()
    
    # Calculate new water table depths
    # Elevation = wt_depth+head
	# New water table depth = Elevation-new_head
    wt1_fname = wt_fmt.format(fname1)
    xwt1,ywt1,wt1 = read_griddata(wt1_fname)
    out_wt1 = (wt1+h1_orig)-new_h3

    wt2_fname = wt_fmt.format(fname2)
    xwt2,ywt2,wt2 = read_griddata(wt2_fname)
    out_wt2 = (wt2+h2_orig)-new_h4

    # Remove artifacts from out of bounds topography
    out_wt1[(out_wt1<-1.) & (out_wt1>marine_mask_val) & ~np.isnan(out_wt1)] = np.nan
    out_wt2[(out_wt2<-1.) & (out_wt2>marine_mask_val) & ~np.isnan(out_wt2)] = np.nan

    # Save new tifs
    wkt1,gt1 = load_grid_prj(rast1,gt_out=True)
    wkt2,gt2 = load_grid_prj(rast2,gt_out=True)
    write_gdaltif(new_wt_fname1,x1_orig,y1_orig,out_wt1,proj_wkt=wkt1,geodata=gt1)
    write_gdaltif(new_wt_fname2,x2_orig,y2_orig,out_wt2,proj_wkt=wkt2,geodata=gt2)
    write_gdaltif(new_head_fname1,x1_orig,y1_orig,new_h3,proj_wkt=wkt1,geodata=gt1)
    write_gdaltif(new_head_fname2,x2_orig,y2_orig,new_h4,proj_wkt=wkt2,geodata=gt2)



