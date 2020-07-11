import sys
import csv
import numpy as np
from pyhdf.SD import SD, SDC
import h5py
import glob
import pandas as pd
import os
import datetime
from datetime import timedelta
from netCDF4 import Dataset
#import mysatellite as ms
#from datetimerange import DateTimeRange
#import satellite_collocation
#import matplotlib
#import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *
import warnings


year_index = int(sys.argv[1]) #Specify year through command line argument
year_string = str(year_index).zfill(4)

index_path = '/umbc/xfs1/cybertrn/common/Data/calipso-virrs-collocated/calipso_index_new/' + year_string + '/'
viirs_path = '/umbc/xfs1/cybertrn/common/Data/calipso-virrs-collocated/viirs_data_new/' + year_string + '/'
ancil_path = '/umbc/xfs1/cybertrn/common/Data/calipso-virrs-collocated/calipso_simplified_labels/' + year_string + '/'
save_path  = '/umbc/xfs1/cybertrn/common/Data/calipso-virrs-collocated/calipso-viirs-merged/' + year_string + '/'

for doy in range(1,366):
    doy_str = str(doy).zfill(3)
    sav_path = save_path + str(doy).zfill(3) + '/'
    if ( os.path.exists(index_path + doy_str) != True ):
        continue
    ind_files = glob.glob( index_path + doy_str + '/*.hdf' )
    for ind_file in ind_files:
        print(ind_file)

        ind_id = SD(ind_file,SDC.READ)

        #geolocation:
        lon01km = ind_id.select('caliop_clay_longitude_1km').get()[:]
        lat01km = ind_id.select('caliop_clay_latitude_1km').get()[:]
        
        nclay05 = ind_id.select('caliop_clay_maximum_number_layers').get()[:]
        nclay01 = ind_id.select('caliop_clay_maximum_number_layers_1km').get()[:]
        ice_f05 = ind_id.select('caliop_clay_ice_fraction').get()[:]
        liq_f05 = ind_id.select('caliop_clay_liquid_fraction').get()[:]
        ice_f01 = ind_id.select('caliop_clay_ice_fraction_1km').get()[:]
        liq_f01 = ind_id.select('caliop_clay_liquid_fraction_1km').get()[:]
        ctoph05  = ind_id.select('caliop_clay_layer_top_altitude').get()[:]
        ctopt05  = ind_id.select('caliop_clay_layer_top_temperature').get()[:]
        cboth05  = ind_id.select('caliop_clay_layer_base_altitude').get()[:]
        cbott05  = ind_id.select('caliop_clay_layer_base_temperature').get()[:]
        copacity = ind_id.select('caliop_clay_opacity_flag').get()[:]
        cfod532  = ind_id.select('caliop_clay_feature_optical_depth_532').get()[:]
        ccolrat  = ind_id.select('caliop_clay_color_ratio').get()[:]
        ciab532  = ind_id.select('caliop_clay_integrated_attenuated_backscatter_532').get()[:]
        ciab1064 = ind_id.select('caliop_clay_integrated_attenuated_backscatter_1064').get()[:]
        clidrat  = ind_id.select('caliop_clay_final_lidar_ratio_532').get()[:]

        #no nalayer in this data
        atoph05 = ind_id.select('caliop_alay_layer_top_altitude').get()[:]
        atopt05 = ind_id.select('caliop_alay_layer_top_temperature').get()[:]
        aboth05 = ind_id.select('caliop_alay_layer_base_altitude').get()[:]
        abott05 = ind_id.select('caliop_alay_layer_base_temperature').get()[:]
        afod532 = ind_id.select('caliop_alay_feature_optical_depth_532').get()[:] 
        acolrat = ind_id.select('caliop_alay_color_ratio').get()[:]
        aiab532 = ind_id.select('caliop_alay_integrated_attenuated_backscatter_532').get()[:]
        aiab1064= ind_id.select('caliop_alay_integrated_attenuated_backscatter_1064').get()[:]
        atype   = ind_id.select('caliop_alay_aerosoltype_mode').get()[:]
        ind_id.end()
    
        ind_name = os.path.basename(ind_file)
        vnp_pos = ind_name.find('vnp')
        timeflag = ind_name[vnp_pos:-4]
        viirs_files = glob.glob( viirs_path + doy_str + '/*' + timeflag + '*.h5')
        if ( len(viirs_files) != 1 ):
            continue
        viirs_file = viirs_files[0]

        ancillary_files = glob.glob( ancil_path + doy_str + '/*' + timeflag + '*.h5')
        if ( len(ancillary_files) != 1 ):
            continue
        ancillary_file = ancillary_files[0]

        viirs_id = h5py.File(viirs_file,'r')

        # geometry:
        viirs_sza = viirs_id['VIIRS_SZA'][:]
        viirs_vza = viirs_id['VIIRS_VZA'][:]
        viirs_saa = viirs_id['VIIRS_SAA'][:]
        viirs_vaa = viirs_id['VIIRS_VAA'][:]

        # observations:
        viirs_m01 = viirs_id['VIIRS_M01'][:]
        viirs_m02 = viirs_id['VIIRS_M02'][:]
        viirs_m03 = viirs_id['VIIRS_M03'][:]
        viirs_m04 = viirs_id['VIIRS_M04'][:]
        viirs_m05 = viirs_id['VIIRS_M05'][:]
        viirs_m06 = viirs_id['VIIRS_M06'][:]
        viirs_m07 = viirs_id['VIIRS_M07'][:]
        viirs_m08 = viirs_id['VIIRS_M08'][:]
        viirs_m09 = viirs_id['VIIRS_M09'][:]
        viirs_m10 = viirs_id['VIIRS_M10'][:]
        viirs_m11 = viirs_id['VIIRS_M11'][:]
        viirs_m12 = viirs_id['VIIRS_M12'][:]
        viirs_m13 = viirs_id['VIIRS_M13'][:]
        viirs_m14 = viirs_id['VIIRS_M14'][:]
        viirs_m15 = viirs_id['VIIRS_M15'][:]
        viirs_m16 = viirs_id['VIIRS_M16'][:]
        viirs_id.close()

        #ancillary:
        ancil_id = h5py.File(ancillary_file,'r')
        igbp = ancil_id['IGBP_surf'][:]
        snic = ancil_id['SNIC_surf'][:]
        skint = ancil_id['skin_temperature'][:]
        skine = ancil_id['skin_emis'][:]
        ancil_id.close()

        n_pixel = len(viirs_m01)
        category = np.zeros( n_pixel, dtype=np.int8 )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_l5 = np.squeeze(np.nanmean(liq_f05,axis=0))
            mean_l1 = np.squeeze(np.nanmean(liq_f01,axis=0))
            mean_i5 = np.squeeze(np.nanmean(ice_f05,axis=0))
            mean_i1 = np.squeeze(np.nanmean(ice_f01,axis=0))
            mean_a5 = np.squeeze(np.nanmean(atype,axis=0))

    #set filter
    #category 1: clean (no cloud no aerosol)
    #category 2: liquid (liquid only no aerosol)
    #category 3: ice (ice only no aerosol)
    #category 4: ice and liquid ( cloud only no aerosol)
    #category 5: aerosol (aerosol only no cloud)
    #category 6: aerosol and cloud

        for i in range(n_pixel):
            if ( (nclay05[i]==0) & (nclay01[i]==0) & (mean_a5[i]<0.01) ):
                category[i] = 1
                continue
            if ( (nclay05[i]==0) & (nclay01[i]==0) & (mean_a5[i]>=0.01) ):
                category[i] = 5
                continue
            if ( (nclay05[i]>0) & (nclay01[i]>0) & (mean_a5[i]>=0.01) ):
                category[i] = 6
                continue

            if ( (nclay05[i]>0) & (nclay01[i]>0) & (mean_l5[i]>0.99) & (mean_l1[i]>0.99) & (mean_a5[i]<0.01) ):
                category[i] = 2
                continue
            if ( (nclay05[i]>0) & (nclay01[i]>0) & (mean_i5[i]>0.99) & (mean_i1[i]>0.99) & (mean_a5[i]<0.01) ):
                category[i] = 3
                continue
            if ( (nclay05[i]>0) & (nclay01[i]>0) & (mean_l5[i]>0.01) & (mean_l1[i]>0.01) & (mean_i5[i]>0.01) & (mean_i1[i]>0.01) & (mean_a5[i]<0.01) ):
                category[i] = 4
                continue

        #print np.sum(category)
        n_cat = np.zeros(7)
        for i in range(7):
            n_cat[i] = len(np.where(category==i)[0])

        print(n_cat)
        
        #save file
        print(timeflag)
        if not os.path.exists(sav_path):
            os.makedirs(sav_path)
        save_name = 'match_data.' +  timeflag + '.h5'
        save_id = h5py.File(sav_path + save_name, 'w')

        save_id.create_dataset('Longitude', data = lon01km)
        save_id.create_dataset('Latitude',  data = lat01km)
        save_id.create_dataset('Pixel_Label', data = category)

        save_id.create_dataset('VIIRS_SZA', data = viirs_sza)
        save_id.create_dataset('VIIRS_VZA', data = viirs_vza)
        save_id.create_dataset('VIIRS_SAA', data = viirs_saa)
        save_id.create_dataset('VIIRS_VAA', data = viirs_vaa)

        save_id.create_dataset('VIIRS_M01', data = viirs_m01)
        save_id.create_dataset('VIIRS_M02', data = viirs_m02)
        save_id.create_dataset('VIIRS_M03', data = viirs_m03)
        save_id.create_dataset('VIIRS_M04', data = viirs_m04)
        save_id.create_dataset('VIIRS_M05', data = viirs_m05)
        save_id.create_dataset('VIIRS_M06', data = viirs_m06)
        save_id.create_dataset('VIIRS_M07', data = viirs_m07)
        save_id.create_dataset('VIIRS_M08', data = viirs_m08)
        save_id.create_dataset('VIIRS_M09', data = viirs_m09)
        save_id.create_dataset('VIIRS_M10', data = viirs_m10)
        save_id.create_dataset('VIIRS_M11', data = viirs_m11)
        save_id.create_dataset('VIIRS_M12', data = viirs_m12)
        save_id.create_dataset('VIIRS_M13', data = viirs_m13)
        save_id.create_dataset('VIIRS_M14', data = viirs_m14)
        save_id.create_dataset('VIIRS_M15', data = viirs_m15)
        save_id.create_dataset('VIIRS_M16', data = viirs_m16)

        save_id.create_dataset('IGBP_SurfaceType', data = igbp)
        save_id.create_dataset('SnowIceIndex', data = snic)
        save_id.create_dataset('Surface_Temperature', data = skint)
        save_id.create_dataset('Surface_Emissivity', data = skine)

        save_id.create_dataset('CALIOP_N_Clay_5km', data = nclay05)
        save_id.create_dataset('CALIOP_N_Clay_1km', data = nclay01)
        save_id.create_dataset('CALIOP_Ice_Fraction_1km', data = ice_f01)
        save_id.create_dataset('CALIOP_Ice_Fraction_5km', data = ice_f05)
        save_id.create_dataset('CALIOP_Liq_Fraction_1km', data = liq_f01)
        save_id.create_dataset('CALIOP_Liq_Fraction_5km', data = liq_f05)
        save_id.create_dataset('CALIOP_Clay_Top_Altitude', data = ctoph05)
        save_id.create_dataset('CALIOP_Clay_Base_Altitude', data = cboth05)
        save_id.create_dataset('CALIOP_Clay_Top_Temperature', data = ctopt05)
        save_id.create_dataset('CALIOP_Clay_Base_Temperature', data = cbott05)
        save_id.create_dataset('CALIOP_Clay_Opacity_Flag', data = copacity)
        save_id.create_dataset('CALIOP_Clay_Optical_Depth_532', data = cfod532)
        save_id.create_dataset('CALIOP_Clay_Color_Ratio', data = ccolrat)
        save_id.create_dataset('CALIOP_Clay_Integrated_Attenuated_Backscatter_532', data = ciab532)
        save_id.create_dataset('CALIOP_Clay_Integrated_Attenuated_Backscatter_1064', data = ciab1064)
        save_id.create_dataset('CALIOP_Clay_Final_Lidar_Ratio_532', data = clidrat)
        save_id.create_dataset('CALIOP_Alay_Top_Altitude', data = atoph05)
        save_id.create_dataset('CALIOP_Alay_Base_Altitude', data = aboth05)
        save_id.create_dataset('CALIOP_Alay_Top_Temperature', data = atopt05)
        save_id.create_dataset('CALIOP_Alay_Base_Temperature', data = abott05)
        save_id.create_dataset('CALIOP_Alay_Optical_Depth_532', data = afod532)
        save_id.create_dataset('CALIOP_Alay_Color_Ratio', data = acolrat)
        save_id.create_dataset('CALIOP_Alay_Integrated_Attenuated_Backscatter_532', data = aiab532)
        save_id.create_dataset('CALIOP_Alay_Integrated_Attenuated_Backscatter_1064', data = aiab1064)
        save_id.create_dataset('CALIOP_Alay_Aerosol_Type_Mode', data = atype)

        save_id.close()
