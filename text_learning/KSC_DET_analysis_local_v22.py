# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:48:01 2016

@author: Chou_h
"""
import sys
sys.path.append(r"R:\Hungtao\program\scripts\HAMRENG")
import os
from KSD_parser import KSD_Parser
import matplotlib.pyplot as plt 

from Tkinter import Tk
import tkFileDialog
import pandas as pd
from HTfunctions import *




    

Tk().withdraw()

filename = r'\\smb.oberon.ncf.wdc.com\EAMR\EAMR_WDB_Data\Spin-stand\TestData\zKSC_Customer_DET'

#filename = r"\\msj.wdc.com\msjdata\SJ2 Shares\Public Department Shares\Magnetic Recording\Write\For MHO\for Harold\050516"
#filename = r'\\wdtbeamr01\eamr\NCF\Spin-stand\TestData\zKSC_Customer_DET'
directory2 = tkFileDialog.askopenfilename(initialdir=filename)
#directory2 = tkFileDialog.askopenfilename(initialdir=r'\\wdtbeamr01\eamr\Test_data')

filefolder = os.path.dirname(directory2)
wafername = os.path.basename(directory2)[:4]
#filename = os.path.basename(directory2)[:-3] +'csv'
#filename2 = os.path.basename(directory2)[:-4]+'_split.csv'

''' use ksc name to decide if it is performance det or customer det '''
df2, ksc_name = KSD_Parser(directory2)  

#save_folder = u'R:/EAMR Database/datatable/DET_WDTHO/ToBeUploaded'
#save_folder = os.path.dirname(directory2)

if "Performance" in ksc_name: 
    #save_folder = r'\\smb.oberon.ncf.wdc.com\EAMR\EAMR_WDB_Data\Spin-stand\TestData\zKSC_Performance_DET\summary'
    save_folder = filefolder
    filename = os.path.basename(directory2)[:-4] +'_P.csv'
    filename2 = os.path.basename(directory2)[:-4]+'_split_P.csv'
else:
#    save_folder = r'\\smb.oberon.ncf.wdc.com\EAMR\EAMR_WDB_Data\Spin-stand\TestData\zKSC_Customer_DET\summary'
    save_folder = filefolder
    filename = os.path.basename(directory2)[:-4] +'_C.csv'
    filename2 = os.path.basename(directory2)[:-4]+'_split_C.csv'
save_filename= save_folder+'\\'+filename
save_filename2= save_folder+'\\'+filename2



''' make the head id and start time into the first two columns '''
df3 = df2.copy()
del(df3['Head ID'])
del(df3['Start Time'])
print 1
print df2['Start Time']
df2 = df2[['Head ID', 'Start Time']+df3.columns.tolist()]


''' SBR info '''
if len(filename.split('-')) >= 2 and 'F' in filename.split('-')[1]:
    df2['SBR'] = filename.split('-')[0]+'_'+filename.split('-')[1]



df2.to_csv(save_filename, index=False)



""" 
To split the data and calculate dWSSNR, etc, also bin the parts, use PES Analysis AvgNRRO to idenitfy the rows that are done in the same run.
"""
df2['For_Split'] = df2['PES Analysis AvgNRRO'].fillna(method = 'ffill')
df2['Head ID2'] = df2['Head ID']


''' 
make the  head id, zone into index, use Seq # to differentiate in case 
the head are ret-ested and have multiple data, but it will not work
if seq is the same (very unlikely to happen)
'''

df3 = df2.set_index(['Head ID2','Seq #', 'For_Split', 'Zone'])  
#df3 = df2.set_index(['Head ID2','Seq #', 'Zone'])  

# assign unstack dataframe always crashes, have to save if and reopen it.
df3.unstack('Zone').to_csv(save_filename2, index = False)

df_s = pd.read_csv(save_filename2)
df_s.drop(df_s.index[[0,1]], inplace = True)   # delete the 1st two non-data rows due to the un-stack.
df_s=df_s.dropna(axis=1,how='all')

df_s['dWSSNR_LC'] = df_s['wsSNR_init dB.1']-df_s['wsSNR_init dB']
df_s['dMWW_LC'] = df_s['WrWidth uin.1']-df_s['WrWidth uin']
df_s['dMWW_LC (nm)'] = df_s['dMWW_LC']*25.4

if "Performance" in ksc_name and "TDS" not in ksc_name:
    df_s['dWSSNR_MWW'] = df_s['wsSNR_init dB.2']-df_s['wsSNR_init dB']
    df_s['dLC_MWW'] = df_s['Laser wr mA.2'] -  df_s['Laser wr mA']
    df_s[r'dLP%'] = df_s['dLC_MWW']/(df_s['Laser wr mA']-12)
else:
    df_s['dAE_TD'] = df_s['zAETD zAETD_R_TDownPower.2']-df_s['zAETD zAETD_R_TDownPower']


df_s['wsSNR init (dB)'] = df_s['wsSNR_init dB.1']

df_s = df_s.rename(columns={'wsSNR_final dB.1': 'wsSNR final (dB)'})

df_s['bin'] = 'D'

bin_C = (df_s['wsSNR_init dB.1']>=8) & (df_s['dWSSNR_LC'] >= -1) & (df_s['Laser wr mA']  <= 80)


df_s['bin'][bin_C] = 'C'



bin_B = (df_s['wsSNR_init dB.1']>=9) & (df_s['dWSSNR_LC'] >= -1) & (df_s['dWSSNR_LC'] <= 2) & (df_s['wsSNR final (dB)'] > 8.5) & \
(df_s['zAETD zAETD_R_TDownPower'] <= 160) & (df_s['zAETD zAETD_R_TDownPower'] >= 100) \
&  (df_s['Laser wr mA']  <= 80)


df_s['bin'][bin_B] = 'B'

bin_A = (df_s['wsSNR_init dB.1']>=10) &  (df_s['dWSSNR_LC'] >= -0.5)  & (df_s['dWSSNR_LC'] <= 2) & (df_s['wsSNR final (dB)'] > 9.5) & \
(df_s['zAETD zAETD_R_TDownPower'] <= 160) & (df_s['zAETD zAETD_R_TDownPower'] >= 100) \
&  (df_s['Laser wr mA']  <= 80)

df_s['bin'][bin_A] = 'A'



''' read the doe file and merge with data file's Column label'''
doefile = r"\\smb.oberon.ncf.wdc.com\EAMR\EAMR TEST UPDATE\Spin Stand\HGA Inventory\HGA data\B4_NFT_DOE_table.csv"
doe = pd.read_csv(doefile)
df_s['Column'] = df_s['Head ID'].map(lambda x: str(x)[7])
df_s2 = pd.merge(df_s, doe, on = 'Column', how = 'left')


#dups = duplicate_columns(df_s2)
#df_s2 = df_s2.drop(dups, axis=1)
df_s2.rename(columns={'zAETD zAETD_R_TDownPower': 'TdP (mW)'}, inplace=True)
# drop duplicate columns
df_s2_temp = df_s2.dropna().T.drop_duplicates().T  # find the dataframe that does not have duplicate columns.
df_s2 = df_s2[df_s2_temp.columns]

if ('ksc version.1' in df_s2.columns) & ('ksc version' not in df_s2.columns):
    df_s2 = df_s2.rename(columns={"ksc version.1": "ksc version"})    

df_s2['wsSNR init (dB)'] = df_s2['wsSNR_init dB.1']

df_s2.to_csv(save_filename2, index = False)





#Saveimage = True
Saveimage = False

if Saveimage:
    from savePTM_v2 import search_and_savePTM
    wafername = "WJKY"
    dirpath = r'R:\EAMR_WDB_Data\Wafer DOE\E0060_G4A-B4 PSPP'+'\\'+wafername+r'\WDB metrology\PTM+'
    fig_folder = r'R:\EAMR TEST UPDATE\Spin Stand\HGA Inventory\HGA data\PTM+'
    key='150kx'

    search_and_savePTM(save_filename2, dirpath, key, fig_folder)
