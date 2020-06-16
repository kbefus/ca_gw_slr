# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:25:24 2016

@author: kbefus
"""

import os,sys
import numpy as np

kbpath = r'C:/Research/Coastalgw/Model_develop/'
sys.path.insert(1,kbpath)

from cgw_model.prep import prep_utils as cprep

#%%


class CRM(object):
    '''
    '''
    
    def __init__(self,work_dir=None,region=None,crm_version=1):
         self.work_dir = work_dir
         self.region = region
         self.crm_version = crm_version
         
         if (self.work_dir is not None) and (self.region is not None) \
             and (self.crm_version is not None):
                 self.fname = os.path.join(self.work_dir,
                                              '{}_crm_v{}.tif'.format(self.region.lower(),
                                                                      self.crm_version))
      
    def load(self):
        if hasattr(self,'fname'):
            temp_extent = cprep.arcpy.Raster(self.fname).extent
            self.extent = [temp_extent.XMin,temp_extent.YMin,
                           temp_extent.XMax,temp_extent.YMax]
                           

class NED(object):
    '''
    '''
    
    def __init__(self,work_dir=None,out_dir=None,crop_extent=None,
                 CRM=None):
        self.work_dir = work_dir
        self.out_dir = out_dir
        self.crop_extent = crop_extent
        self.CRM = CRM
    
    def load(self,ned_spacing = [1.,1.],extent_pad = 2.,
             buffer_size = 0.,mosaic_fname=None,mosaic=True,overwrite_flag=False):
        
        # Create top left corners of NED datasets
        Y,X = np.mgrid[np.floor(self.crop_extent[1])-extent_pad:np.ceil(self.crop_extent[3])+extent_pad:ned_spacing[1],
                       np.floor(self.crop_extent[0])-extent_pad:np.ceil(self.crop_extent[2])+extent_pad:ned_spacing[0]]
        
        self.in_pts = (X<np.ceil(self.crop_extent[2])+buffer_size) & \
                       (X>=np.floor(self.crop_extent[0])-buffer_size) & \
                       (Y<=np.ceil(self.crop_extent[3])+buffer_size) & \
                       (Y>np.ceil(self.crop_extent[1])-buffer_size)
        
        self.filename = os.path.join(self.out_dir,'{}_DEM.tif'.format(self.CRM.region.lower()))
        
        self.fnames = cprep.collect_NED(ned_dir=self.work_dir,XY=[X,Y],internal_pts=self.in_pts)
        
        if mosaic_fname is None:
            self.mosaic_fname = os.path.join(self.out_dir,'{}_NED.tif'.format(self.CRM.region.lower()))
        
        if mosaic:
            if not os.path.isfile(self.mosaic_fname) or overwrite_flag:
                cprep.mosaic_NED(self.fnames,self.mosaic_fname)
            else:
                print 'Mosaic file already exists: {}'.format(self.mosaic_fname)
        
class DEM_merge(object):
    
     def __init__(self,NED=None,fix_shp=None,other_dem=None,shp_not_use=None,
                  shp_use_other=None):
         self.NED=NED        
         self.fix_shp = fix_shp
         self.other_dem = other_dem
         self.shp_not_use = shp_not_use
         self.shp_use_other = shp_use_other
        
     def join_main(self,out_fname=None,CRM_fname=None,
                   NED_fname=None,max_elev_from_bathy=8.):
         if out_fname is None:
             self.join_fname = os.path.join(self.NED.out_dir,'{}_joined.tif'.format(self.NED.CRM.region.lower()))
         else:
             self.join_fname = out_fname
         
         if CRM_fname is None:
             CRM_fname = self.NED.CRM.fname
         
         if NED_fname is None:
             NED_fname = self.NED.mosaic_fname
             
         cprep.join_NED_CRM(NED_fname,CRM_fname,
                            self.join_fname,
                            max_elev_from_bathy=max_elev_from_bathy)
        
        
     def fix(self,dem_fname=None,max_elev_to_fix=None):
         if max_elev_to_fix is None:
             max_elev_to_fix = 0.
         if dem_fname is None:
             self.fix_fname = cprep.fix_NED_CRM(self.join_fname,self.fix_shp,max_elev_to_fix=max_elev_to_fix)
         else:
             self.fix_fname = cprep.fix_NED_CRM(dem_fname,self.fix_shp,max_elev_to_fix=max_elev_to_fix)
        
     def join_fix(self,elev_threshold=None):
         if elev_threshold is None:
             elev_threshold=0.
             
         self.fix2_fname = cprep.join_DEM_bathy(self.fix_fname,self.other_dem,
                                                self.shp_not_use,self.shp_use_other,
                                                elev_threshold=elev_threshold)
     
     def run(self,fix=False,second_join=False,max_elev_from_bathy=8.):
         self.join_main(max_elev_from_bathy=max_elev_from_bathy)
         if fix:
             self.fix()
         if second_join:
             self.join_fix()
        
        
     