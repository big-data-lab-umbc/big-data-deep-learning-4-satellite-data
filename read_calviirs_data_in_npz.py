# -*- coding: utf-8 -*-
"""
Created on Thu Jul 09 14:19:19 2020

@author: Sahara Ali
"""
import numpy as np
import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, calendar, sys, fnmatch, datetime
from pyhdf.SD import SD, SDC
import h5py
import netCDF4
from matplotlib import colors as CS
from tempfile import TemporaryFile

verbose  = False

#SatelliteData class reads a specified list of variables from the hdf filepath provided
class SatelliteData(object):
    def __init__(self,datatype,version,fname,variables=None,verbose=False):
        self.datatype = datatype
        self.version = version
        self.fname   = fname
        self.verbose = verbose
        self.dataset = {}
        print(version.capitalize())
        if version  == 'HDF4':
            hdf = SD(self.fname, SDC.READ)
            if variables is None:
                for d in hdf.datasets():
                    self.dataset[d]=np.array(hdf.select(d).get())
                    if verbose:print(d)
            else:
                for d in variables:
                    self.dataset[d]=np.array(hdf.select(d).get())
                    if verbose:print('reading',d)

        elif version  == 'HDF5':
            hdf = h5py.File(fname, 'r')
            if variables is None:
                allvariables = list(hdf.keys())
                print('allvariables',allvariables)
                for d in allvariables:
                    self.dataset[d]=np.array(hdf[d])
                    if verbose:print(d)
            else:
                for d in variables:
                    self.dataset[d]=np.array(hdf[d])
                    if verbose:print(d)

        else:
            print(datatype,'  not supported')
        if verbose:
            for key in  self.dataset:
                print('variable: ',key, ' dimension', self.dataset[key].shape,\
                       'min, max', np.nanmin(self.dataset[key]),np.nanmax(self.dataset[key]))
            print('finished')

#List of features to read from calipso, viirs and IFF datasets
            
calipso_data_list = ['CALIOP_N_Clay_1km','CALIOP_N_Clay_5km','CALIOP_Liq_Fraction_1km','CALIOP_Liq_Fraction_5km','CALIOP_Ice_Fraction_1km','CALIOP_Ice_Fraction_5km','CALIOP_Clay_Top_Altitude','CALIOP_Clay_Base_Altitude','CALIOP_Clay_Top_Temperature','CALIOP_Clay_Base_Temperature','CALIOP_Clay_Optical_Depth_532','CALIOP_Clay_Opacity_Flag','CALIOP_Clay_Integrated_Attenuated_Backscatter_532','CALIOP_Clay_Integrated_Attenuated_Backscatter_1064','CALIOP_Clay_Final_Lidar_Ratio_532','CALIOP_Clay_Color_Ratio','CALIOP_Alay_Aerosol_Type_Mode','CALIOP_Alay_Top_Altitude','CALIOP_Alay_Base_Altitude','CALIOP_Alay_Top_Temperature','CALIOP_Alay_Base_Temperature','CALIOP_Alay_Integrated_Attenuated_Backscatter_532','CALIOP_Alay_Integrated_Attenuated_Backscatter_1064','CALIOP_Alay_Color_Ratio','CALIOP_Alay_Optical_Depth_532']
position_list = ['Latitude','Longitude']
aux_data_list =['Surface_Temperature','Surface_Emissivity','IGBP_SurfaceType','SnowIceIndex']
VIIRS_data_list=['VIIRS_SZA','VIIRS_SAA','VIIRS_VZA','VIIRS_VAA','VIIRS_M01','VIIRS_M02','VIIRS_M03','VIIRS_M04','VIIRS_M05','VIIRS_M06','VIIRS_M07','VIIRS_M08','VIIRS_M09','VIIRS_M10','VIIRS_M11','VIIRS_M12','VIIRS_M13','VIIRS_M14','VIIRS_M15','VIIRS_M16']
label_list = ['Pixel_Label']

calipso_all = np.array([])
calipso_all = np.empty((0,25), float)

viirs_all = np.array([])
viirs_all = np.empty((0,20), float)

iff_all = np.array([])
iff_all = np.empty((0,4), float)

latlon_all = np.array([])
latlon_all = np.empty((0,2), float)

label_all = np.array([])
label_all = np.empty((0,1), int)

days = 365 #Specify number of days to read. Same string gets appended to npz file name saved e.g 365 days 
year = '/2017/'
#Path to load collocated dataset in h5
root = '/umbc/xfs1/cybertrn/common/Data/calipso-virrs-collocated/calipso-viirs-merged' + year

#Path to save collocated dataset in npz
sav_path = '/umbc/xfs1/cybertrn/common/Data/calipso-virrs-collocated/calipso-viirs-merged'

for doy in range(1,days+1): 
    doy_str = str(doy).zfill(3)
    if ( os.path.exists(root + doy_str) != True ):
        continue
    ind_files = glob.glob( root + doy_str + '/*.h5' ) 
    for ind_file in ind_files:
        flag = 0
        pathname= ind_file
        print(pathname)
        calipso_data =SatelliteData('Collocated_CALIPSO_features','HDF5',
                                    pathname,\
                                    variables=calipso_data_list,\
                                    verbose=verbose)
        iff_data = SatelliteData('Collocated_aux_features','HDF5',
                                    pathname,\
                                    variables=aux_data_list,\
                                    verbose=verbose)
        viirs_data = SatelliteData('Collocated_VIIRS_features','HDF5',
                                    pathname,\
                                    variables=VIIRS_data_list,\
                                    verbose=verbose)
        label_data = SatelliteData('Labels','HDF5',
                                    pathname,\
                                    variables=label_list,\
                                    verbose=verbose)
        position_data = SatelliteData('Lat_Long','HDF5',
                                    pathname,\
                                    variables=position_list,\
                                   verbose=verbose)
        
        
        max_dim = max(calipso_data.dataset['CALIOP_Alay_Aerosol_Type_Mode'].shape[0], calipso_data.dataset['CALIOP_Alay_Aerosol_Type_Mode'].shape[1])
        print('Max_dim Calipso: ',max_dim)
        calipso = np.empty((0,max_dim), float)
        viirs = np.empty((0,max_dim), float)
        iff = np.empty((0,max_dim), float)
        latlon = np.empty((0,max_dim), float)
        label = np.empty((0,max_dim), int)
        
        for i in calipso_data.dataset.keys():
            try:
                if calipso_data.dataset[i].shape[0] <calipso_data.dataset[i].shape[1]:
                    print(calipso_data.dataset[i].shape)
                    temp = np.nanmean(calipso_data.dataset[i],axis=0)
                    calipso = np.vstack((calipso, temp))

                else:
                    print('working:',i)
                    temp = np.nanmean(calipso_data.dataset[i],axis=1)
                    temp_idx = np.isnan(temp)
                    temp[temp_idx] = 0
                    calipso = np.vstack((calipso, temp))
            except:
                print("Something went wrong when reading Calipso.")
                flag = 1
            finally:
                if flag == 1:
                    break
        
        if flag == 1:
            print("Skipping file due to CALIPSO")
            continue
        
        for i in viirs_data.dataset.keys():
            temp = viirs_data.dataset[i]
            temp_idx = np.isnan(viirs_data.dataset[i])
            temp[temp_idx] = 0
            viirs = np.vstack((viirs, temp))
        
        for i in position_data.dataset.keys():
            temp = position_data.dataset[i]
            latlon = np.vstack((latlon, np.transpose(temp)))
        
        for i in label_data.dataset.keys():
            temp = label_data.dataset[i]
            label = np.vstack((label, np.transpose(temp)))
        
        for i in iff_data.dataset.keys():
                try:
                    if iff_data.dataset[i].shape[0] < 100:
                        print('iff_data.dataset[i].shape[0] < 100: ',i)
                        temp = np.nanmean(iff_data.dataset[i],axis=0)
                        temp_idx = np.isnan(temp)
                        temp[temp_idx] = 0
                        iff = np.vstack((iff, temp))
                    else:
                        print('working iff', i)
                        temp = iff_data.dataset[i]
                        iff = np.vstack((iff, temp))
                except:
                    print("Something went wrong when reading IFF.")
                    flag = 1
                finally:
                    if flag == 1:
                        break
        
        if flag == 1:
            print("Skipping file due to IFF")
            continue
        try:        
            calipso = np.transpose(calipso)
            calipso_all = np.vstack((calipso_all,calipso))
        except:
            print("Something went wrong when reading CALIPSO.")
            flag = 1

        #Append data from individual files into one numpy array for each category/source              
        viirs = np.transpose(viirs)
        print('Shape VIIRS:',viirs.shape)
        print('Shape VIIRS_all:',viirs_all.shape)
        viirs_all = np.vstack((viirs_all,viirs))
         
        iff = np.transpose(iff)
        iff_all = np.vstack((iff_all,iff))
        
        latlon = np.transpose(latlon)
        latlon_all = np.vstack((latlon_all,latlon))
        
        label = np.transpose(label)
        label_all = np.vstack((label_all, label))
 
    #Save all numpy datasets in one npz file
    test_data = TemporaryFile()
    mod = 'train' #Specify 'train' for training data and 'test' for test data generation
    np.savez(sav_path + '/' + mod + str(days) + '_days' + '.npz', calipso=calipso_all, viirs=viirs_all, iff = iff_all, latlon = latlon_all, label = label_all)
    _ = test_data.seek(0)    




