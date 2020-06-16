# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:27:41 2016

@author: kbefus
"""

import pandas as pd
from urllib.request import urlopen
import numpy as np
import time,os

from cgw_model.cgw_utils import cgw_feature_utils as cfu
#%%

# -------------- Variables --------------
wq_dict = {'MonitoringLocationIdentifier':'site_no',
           'ActivityStartDate':'sdate',
           'ResultMeasureValue':'value',
           'LatitudeMeasure':'lat',
           'LongitudeMeasure':'long',
           'VerticalMeasure/MeasureValue':'surf_elev',
           'VerticalMeasure/MeasureUnitCode':'surf_elev_units',
           'WellDepthMeasure/MeasureValue':'wdepth',
           'WellDepthMeasure/MeasureUnitCode':'wdepth_units',
           'WellHoleDepthMeasure/MeasureValue':'hdepth',
           'ActivityTopDepthHeightMeasure/MeasureValue':'screen_top',
           'ActivityBottomDepthHeightMeasure/MeasureValue':'screen_bot',
           'USGSPCode':'pcode',
           'ResultMeasure/MeasureUnitCode':'units',
           'DetectionQuantitationLimitMeasure/MeasureValue':'det_limit',
           'DetectionQuantitationLimitMeasure/MeasureUnitCode':'det_unit',
           'AquiferTypeName':'aq_type',
           'AquiferName':'aq_name',
           'ResultValueTypeName':'val_type'}

default_wq_cols = ['MonitoringLocationIdentifier','ActivityStartDate','USGSPCode',
                   'ResultMeasureValue','ResultMeasure/MeasureUnitCode',
                   'DetectionQuantitationLimitMeasure/MeasureValue','ResultValueTypeName',
                   'ActivityTopDepthHeightMeasure/MeasureValue','ActivityBottomDepthHeightMeasure/MeasureValue',
                   ]
                   
default_site_cols = ['MonitoringLocationIdentifier','LongitudeMeasure','LatitudeMeasure','VerticalMeasure/MeasureValue',
                     'VerticalMeasure/MeasureUnitCode',
                    'WellDepthMeasure/MeasureValue',
                    'WellHoleDepthMeasure/MeasureValue','WellDepthMeasure/MeasureUnitCode',
                    'ActivityTopDepthHeightMeasure/MeasureValue',
                    'ActivityBottomDepthHeightMeasure/MeasureValue','WellDepthMeasure/MeasureUnitCode',
                    'AquiferName','AquiferTypeName'] 

param_dict = {'totalC_unfilt':'00690','totalC_filt':'00682','inorgC_unfilt':'00691',
          'inorgC_filt':'00685','orgC_unfilt':'00680','orgC_filt':'00681',
          'totalN_unfilt':'00630','totalN_filt':'00631','NO3asN_filt':'00618',
          'NO3_filt':'71851','NO3_unfilt':'00620',
          'disO':'00300','disOp':'00301','disOlab':'62971',
          'land_altft':'00042',
          'land_alt':'72000',
          'wl_dblsdm':'30210',
          'wl_dlsft':'72019',
          'wl_dtopft':'72020',
          'gwl_NAV29ft':'62600',
          'gwl_NAVD27ft':'62610',
          'gwl_NAVD88ft':'62611',
          'gwl_NAVD27m':'62612',
          'gwl_NAVD88m':'62613',
          'gwl_MSLft':'72150',
          'gwl_dlsm':'99019',
          }

# -------------- Functions --------------
def get_bbox_string(bbox,wq=False,ndec=6):
    '''Returns a string of bbox coordinates for insertion in NWIS call.
    
    Inputs
    --------
    
    bbox: list or np.ndarray
        iterable with the form [min_long,min_lat,max_long,max_lat]
    
    Returns
    -------
    out_fmt: str
        output bbox string formatted    
    
    Source: modified from Wesley Zell, pymodflow.pydata.nwis_tools    
    '''
    start_fmt = '{0:.xf},{1:.xf},{2:.xf},{3:.xf}'
    start_fmt = start_fmt.replace('.x','.{}'.format(ndec))
    out_fmt = start_fmt.format(*bbox)
    if wq:
        out_fmt = out_fmt.replace(',','%2C')
        
    return out_fmt

def replace_txt(txt,orig=None,new=''):
    '''Replace text helper function.
    '''
    return txt.replace(orig,new)

def make_nwis_url(bbox=None,src_type='gwlevels',
                  coordinate_format='decimal_degrees',
                  fmt='rdb',params=None,start='1900-01-01',
                  enddt=None,opts='siteType=GW&siteStatus=all',
                  site_no=None):
    if src_type.lower() == 'gwlevels':
        base_url='https://waterservices.usgs.gov/nwis/gwlevels'
    elif src_type.lower() == 'current':
        base_url='http://waterservices.usgs.gov/nwis/iv'
    
    if (bbox is not None):
        loc_str = get_bbox_string(bbox,wq=False)
        loc_str = '?bBox={}'.format(loc_str)
    elif (site_no is not None):
        loc_str = '?sites={}'.format(site_no)
    
    param_str=''
    if (params is not None):
        try:
            int_test = int(params[0])
            nwis_params = params
    
        except:
            # input parameters are names, not parameter numbers
            nwis_params=[param_dict[iname] for iname in params]

        if isinstance(nwis_params,(list,tuple)):
            param_str = ','.join(nwis_params)
        else:
            param_str = nwis_params
        param_str = 'parameterCd={}'.format(param_str)
     
               
    fmt_str = 'format={}'.format(fmt)
    start_date = 'startDT={}'.format(start)
    
    if enddt is not None:
        enddt = 'endDT={}'.format(enddt)
        
    query_url = '&'.join([i for i in [loc_str,fmt_str,param_str,start_date,enddt,opts] if i not in ['',None]])    
    
    
    out_url = '/'.join([base_url,query_url])   
    return out_url

def make_site_url(sites=None,base_url='https://nwis.waterdata.usgs.gov/nwis/site/?',
                  col_names=['site_no','station_nm','site_tp_cd',
                             'dec_lat_va','dec_long_va','coord_datum_cd',
                             'alt_va','nat_aqfr_cd','aqfr_cd','aqfr_type_cd',
                             'well_depth_va','hole_depth_va','gw_begin_date',
                             'gw_end_date','gw_count_nu']):
    
    site_info='multiple_site_no={}'.format('%2C'.join(sites))
    start_info='group_key=NONE&format=sitefile_output&sitefile_output_format=rdb'
    col_info = '&'.join(['column_name={}'.format(icol) for icol in col_names])
    other_info='date_format=YYYY-MM-DD&rdb_compression=file&list_of_search_criteria=multiple_site_no'
    
    query_url = '&'.join([site_info,start_info,col_info,other_info])
    out_url = ''.join([base_url,query_url])
    return out_url

def make_site_url2(sites=None,base_url='http://waterservices.usgs.gov/nwis/site/?',
                  ):
    if isinstance(sites,str):
        site_info='sites={}'.format(sites)
    else:
        site_info='sites={}'.format(','.join([st.decode("utf-8") for st in sites.astype('|S')]))
    start_info='siteOutput=expanded&format=rdb'
    
    query_url = '&'.join([site_info,start_info])
    out_url = ''.join([base_url,query_url])
    return out_url
    
def make_wq_url(bbox=None,within_dist_dict=None,huc_list=None,loc_code=None,
                params=None,param_type='pCode',timing_dict=None,
                dl_type='Station',base_url='http://www.waterqualitydata.us',
                mimeType='csv',dl_zip='no',dl_sorted='no'):
    '''Make water quality url.
    '''
    
    query_url = 'search?'
    
    # Prescribe location information
    if (bbox is not None):
        loc_str = get_bbox_string(bbox,wq=True)
        loc_str = 'bBox={}'.format(loc_str)
    elif (within_dist_dict is not None):
        loc_str = 'within={}&lat={}&long={}'.format(within_dist_dict['within'],within_dist_dict['lat'],within_dist_dict['long'])
    elif (huc_list is not None):
        loc_str = 'huc={}'.format(';'.join(np.array(huc_list).astype('|S')))
    elif (loc_code is not None):
        loc_str = loc_code
    
    # Prescribe parameter information
    if (params is not None):
        if param_type.lower() in ['pcode']:
            if isinstance(params,(list,tuple)):
                param_str = ';'.join(params)
            else:
                param_str = params
            
            param_str = 'pCode={}'.format(param_str)
        elif param_type.lower() in ['characteristictype']:
            params_str = 'characteristicName={}'
            param_list = ['characteristicType={}'.format(params['type'])]
            for name in params['name']:
                param_list.append(params_str.format(name))
                
            param_str = '&'.join(param_list)
    
    # Prescribe sample timing information
    if (timing_dict is not None):
        if 'start' in timing_dict.keys():
            strt_txt = 'startDateLo={}'.format(timing_dict['start'])
        else:
            strt_txt = ''
        
        if 'end' in timing_dict.keys():
            end_txt = 'startDateHi={}'.format(timing_dict['end'])
        else:
            end_txt = '' 
        
        time_txt_list = [i for i in [strt_txt,end_txt] if i not in ['']]
        if len(time_txt_list)>1:
            time_str = '&'.join(time_txt_list)
        elif len(time_txt_list)==1:
            time_str = time_txt_list[0]
        else:
            time_str = ''
    else:
        time_str = ''
    # Prescribe output file information               
    output_str = 'mimeType={}&zip={}&sorted={}'.format(mimeType,dl_zip,dl_sorted)
    
    query_pieces_list = [loc_str,param_str,time_str,output_str]
    query_pieces_list = [q1 for q1 in query_pieces_list if q1 not in ['']]
    query_url = '{}{}'.format(query_url,'&'.join(query_pieces_list))
   
    out_url = '/'.join([base_url,dl_type,query_url])
    return out_url

def df_from_url(iurl,sep=None,comment='#',header='infer',is_wq=False):
    '''Reads a USGS waterservices rdb response to a dataframe.
    
    Source: Wesley Zell, pymodflow.pydata.nwis_tools.df_from_url     
    '''
    
    irdb = urlopen(iurl)
    try:
        idf = pd.read_csv(irdb,sep=sep,comment=comment,header=header)
        if (is_wq == False):
            idf = idf.drop(idf.index[0])    # Drop the description of the field widths
            
        return idf
    except ValueError as err:
        if err in ['No columns to parse from file']:
            return err # all is well: no sites found


def load_historicgw_df(url_opts=None,max_sites=100, agg_funcs=None,
                       save_dict=None,agg_values=True):
    '''Load historic groundwater site data from NWIS.'''

    data_url = make_nwis_url(**url_opts)
    data_df = df_from_url(data_url,sep='\t')
    if url_opts['site_no'] is None:
        sites_found = data_df['site_no'].dropna().unique().astype('|S')
        if sites_found.shape[0]>max_sites:
            # Split up queries
            nqueries = sites_found.shape[0]/max_sites
            print("{} sites found, breaking into {} calls".format(sites_found.shape[0],nqueries))
            istart=0
            all_dfs=[]
            while istart<=sites_found.shape[0]:
                temp_url =  make_site_url2(sites=sites_found[istart:istart+max_sites])
                temp_df = df_from_url(temp_url,sep='\t')
                all_dfs.append(temp_df)
                istart+=max_sites
            site_df = pd.concat(all_dfs,axis=0,ignore_index=True).reset_index()    
            site_url=temp_url
        elif sites_found.shape[0]>0:
            site_url = make_site_url2(sites=sites_found)
            site_df = df_from_url(site_url,sep='\t')
        else:
            site_df = None
            site_url = None
    else:
        site_url = make_site_url2(sites=url_opts['site_no'])
        site_df = df_from_url(site_url,sep='\t')
    
    if site_df is not None:
        data_df= data_df.apply(pd.to_numeric,errors='ignore')  
        site_df= site_df.apply(pd.to_numeric,errors='ignore')
    
    if save_dict is not None and site_df is not None:
        site_df.to_csv(os.path.join(save_dict['work_dir'],'site_hist_data.csv'),
                       index=False)
        data_df.to_csv(os.path.join(save_dict['work_dir'],'wl_hist_data.csv'),
                       index=False)
    
    if site_df is not None:
        if site_df.shape[0]>0 and agg_values:
            
            
            if agg_funcs is None:
                agg_funcs = ['count',np.median,np.mean,np.max,np.min]
                
            org_df = organize_wq_data(data_df,ind_cols=['site_no'],agg_funcs=agg_funcs,
                                      agg_col='lev_va')    
            merged_df = pd.merge(site_df,org_df,left_on='site_no',right_index=True)
            
            # Fix multi-column names
            new_cols = pd.Index(['_'.join(e) if isinstance(e,tuple) else e for e in merged_df.columns.tolist()])
            merged_df.columns = new_cols
                
            return merged_df,[site_url,data_url]
        elif site_df.shape[0]>0:
            return site_df,data_df
    else:
        print("Historical data not found for bounding box.")
        return None, [site_url,data_url]
            
def get_wq_df(in_url,sep=',',
              cols_keep=None,save_all_cols=False,header=0):
    '''HELPER FUNCTION. Returns a dataframe of water quality observations.

    Source: modified from Wesley Zell, pymodflow.pydata.nwis_tools.get_wq_results    
    '''
    
    data_df = df_from_url(in_url,sep=sep,header=header,is_wq=True)
    
    if isinstance(data_df,pd.DataFrame):
        # Remove missing columns
        df_cols = data_df.columns.values
        if (cols_keep is None):
            # Parse url to find if results or site data
            url_type = in_url.split('.us')[-1].split('/')[1]
            if url_type.lower() in ['result'] and not save_all_cols:
                cols_keep = default_wq_cols
            elif url_type.lower()  in ['station'] and not save_all_cols:
                cols_keep = default_site_cols
            else:
                cols_keep = data_df.columns # keep all if save_all_cols or url is wonky
                
        cols_keep_culled = [icol for icol in cols_keep if icol in df_cols]
        
        try:
            data_df = data_df[cols_keep_culled]
            data_df.columns = [wq_dict[icol] for icol in cols_keep_culled]
            if 'site_no' in data_df.columns:
                data_df['site_no'] = data_df['site_no'].apply(replace_txt,orig='USGS-')
                # Remove any dataframe rows that were read from blank rows in the file
                # (or, in some cases, are comment lines and headers for additional sites)
                data_df['site_no'] = data_df['site_no'].apply(pd.to_numeric,errors='coerce')    
                data_df = data_df.dropna(subset=['site_no'])
        except:
            flag=1
        if 'sdate' in data_df.columns:
            data_df['sdate'] = data_df['sdate'].apply(pd.to_datetime)
    
    return data_df

def organize_wq_data(in_df,ind_cols=['site_no','pcode'],
                     agg_col='value',agg_funcs=None,param_dict=param_dict):
    '''Assimilate and aggregate data by ind_cols
    '''
    func_df = in_df.copy()
    func_df.set_index(ind_cols,inplace=True,drop=True)
    if len(ind_cols)>1:
        group_df = func_df.groupby(level=ind_cols)
        if (agg_funcs is not None):
            agg_df = group_df.agg({agg_col: agg_funcs}).unstack()
            
            col_to_txt = [['{0:05.0f}'.format(idum[-1]),idum[1]] for idum in agg_df.columns.values]
            
            # Find matching colnames
            col_names2 = ['{}_{}'.format(list(param_dict.keys())[list(param_dict.values()).index(txt1[0])],txt1[1]) for txt1 in col_to_txt]
            agg_df.columns = col_names2
    else:
        group_df = func_df.groupby(ind_cols[0])
        if (agg_funcs is not None):
            agg_df = group_df.agg({agg_col: agg_funcs})

    if (agg_funcs is not None):
        return agg_df
    else:
        return group_df

def internal_sites(in_df,in_shp,lonlat_cols=['long','lat']):
    '''Locate sites inside in_shp
    '''

    iloc = cfu.pt_in_shp(in_shp,in_df[lonlat_cols].values)
    return iloc[0],in_df.loc[iloc[0],:]

def merge_dfs(left_df,right_df,**kwargs):
    '''Merge pandas dataframes helper function.
    '''
    
    out_df = pd.merge(left_df,right_df,**kwargs)
    return out_df
    
def plot_wq(df,c=None,s=10,cmap='rainbow',**kwargs):
    '''Plot scatter plot of water quality data.
    '''
    df.plot.scatter('long','lat',s=s,c=c,edgecolor='none',cmap=cmap,**kwargs)


def load_nwis_wq(shp=None, params=None,huc_list=None,agg_funcs=None,
                 bbox=None,loc_code=None,group_only=False,save_dict=None):
    '''Load water quality data from nwis using shapefile.
    '''
    if shp is not None:
        bbox = shp.bbox # minx,miny,maxx,maxy

    try:
        int_test = int(params[0])
        nwis_params = params

    except:
        # input parameters are names, not parameter numbers
        nwis_params=[param_dict[iname] for iname in params]
        
    # Make urls and download data    
    station_url = make_wq_url(bbox=bbox,huc_list=huc_list,loc_code=loc_code,
                              params=nwis_params,dl_type='Station')
    print(station_url)
        
    data_url = make_wq_url(bbox=bbox,huc_list=huc_list,loc_code=loc_code,
                           params=nwis_params,dl_type='Result')
    
    # Convert data to pandas dataframes
    site_df = get_wq_df(station_url)
    data_df = get_wq_df(data_url)
    
    if save_dict is not None:
        if 'id' not in save_dict.keys():
            site_df.to_csv(os.path.join(save_dict['work_dir'],'site_data.csv'),
                           index=False)
            data_df.to_csv(os.path.join(save_dict['work_dir'],'wl_data.csv'),
                           index=False)
        else:
            site_df.to_csv(os.path.join(save_dict['work_dir'],'{}_site_data.csv'.format(save_dict['id'])),
                           index=False)
            data_df.to_csv(os.path.join(save_dict['work_dir'],'{}_wl_data.csv'.format(save_dict['id'])),
                           index=False)
    
    if site_df.shape[0]>0:
        if isinstance(data_df,pd.DataFrame) and \
                        (agg_funcs is not None or group_only):
            data_df = organize_wq_data(data_df,agg_funcs=agg_funcs)
            
        return site_df,data_df
    else:
        print("No sites found on NWIS server for given domain.")
        return None,None

def select_wq_sites(site_df=None,data_df=None,shp=None,agg_funcs=None):
    site_inds,internal_site_df=internal_sites(site_df,shp)
    merged_df = []
    if isinstance(data_df,pd.DataFrame) and internal_site_df.shape[0]>0:
        # Calculate statistics for sample sites
        # if agg_stats=None, returns grouped df by site
        if agg_funcs is not None:
            data_df = organize_wq_data(data_df,agg_funcs=agg_funcs)
        
        merge_dict = {'how':'inner','left_on':'site_no','right_index':True}
        merged_df = merge_dfs(internal_site_df,data_df,**merge_dict)
            
    return merged_df

    
def load_nhdplus_erom(workdir=None,Qtype='MA',Qfields=['Comid','Qincr0001C', 'Qincr0001E','Q0001C', 'Q0001E'],
                      indexcol='COMID',Q_field_out=None):

    erom_file = 'EROM_{}0001.DBF'.format(Qtype)
    Qfield = Qfields[1:]
    
    
    nhdplus_erom_fpath = os.path.join(workdir,erom_file)
    Q_df = load_NHD_dbf(nhdplus_erom_fpath,Qfields,indexcol)
    if Q_field_out is not None:
        # Need two output fields
        Q_field_out_new = []
        for Qfieldtemp in Qfield:
            Q_field_out_new.append(''.join([Q_field_out,'_',Qfieldtemp[-1]]))
        Q_field_out = Q_field_out_new
        Q_df[Q_field_out] = Q_df[Qfield].copy()
        Q_df.drop(Qfield,inplace=True,axis=1)
        
    Q_units = 'cfs'
    Q_df[Qfield] = to_m3yr(Q_df[Qfield],Q_units)
    return Q_df


def load_NHD_dbf(fname,fields2get,comid_name_out):
    """
    Load NHDplus dbf file to pandas dataframe.
    
    Useage:\n
    ws_data = load_NHD_dbf(fname,fields2get,comid_name_out)
    
    Inputs:\n
    fname = full path of NHD dbf file with extension.\n
    fields2get = list of columns from dbf to load, first column set as index of df.\n
    comid_name_out =  name of comid (i.e., index) to use, sometimes different than dbf name.\n
    
    Output:\n
    ws_data = pandas dataframe of loaded columns.
    
    """
    from pysal import open as pysalopen
    dbfdata = pysalopen(fname,'r')
    orig_dbf_fields = np.array(dbfdata.field_info)[:,0]
    matching_fields = match_fieldnames(fields2get,orig_dbf_fields)
    ws_data = pd.DataFrame(dbfdata.by_col_array(matching_fields),columns=fields2get).drop_duplicates()
    ws_data.set_index(fields2get[0],inplace=True,drop=True)
    ws_data.index.name = comid_name_out
    return ws_data


def match_fieldnames(inlist,inlist2):
    inlist_lower = [i.lower() for i in inlist]
    inlist2_lower = [i.lower() for i in inlist2]
    
    out_list = []
    for alist in inlist_lower:
        if alist in inlist2_lower:
            # Assign corresponding name from inlist2
            out_list.append(inlist2[inlist2_lower.index(alist)])
            continue
    return out_list
    
def to_m3yr(datain,unitsin,area=None):
    cfs_2_m3yr = (86400.*365.25)/(3.2808399**3.)
    if area is None:
        # covert from cfs to m3/yr  
        dataout = datain.multiply(cfs_2_m3yr) #cfs to m^3/yr
    else:
        if np.char.equal('mm/yr',unitsin):
            dataout = datain.divide(1e3).multiply(area.multiply(1e6),axis=0)

        elif np.char.equal('100 mm/yr',unitsin):
            dataout = datain.divide(1e5).multiply(area.multiply(1e6),axis=0)
            
        elif np.char.equal('cfs',unitsin):
            dataout = datain.multiply(cfs_2_m3yr) #cfs to m^3/yr
        else:
            print("Unit not found for {}".format(unitsin))
    return dataout









