# -*- coding: utf-8 -*-
import PhotUtils
import argparse
import pickle
import pyfits
import glob
import sys
import os
import numpy as np
import re
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Use for UTC -> BJD conversion:
import dateutil
import matplotlib.dates as mdates
from astropy.coordinates import SkyCoord
from astropy import units as u

# Use for WCS obtention:
from astropy import wcs
from astropy.io import fits

###################################################################

# Get user input:
parser = argparse.ArgumentParser()
parser.add_argument('-project',default=None)
parser.add_argument('-datafolder',default=None)
parser.add_argument('-minap',default = 5)
parser.add_argument('-maxap',default = 50)
parser.add_argument('-apstep',default = 1)
# If you are doing photometric monitoring of an object, sometimes you want to join 
# all the images from one object taken on the same band in the same output. Set this 
# to True if you want to do this:

# Run astrometry on the images?
parser.add_argument('--get_astrometry', dest='get_astrometry', action='store_true')
parser.set_defaults(get_astrometry=False)

# Generate gaussian-filtered version of the images? (sometimes good for astrometry of 
# highly defocused targets)
parser.add_argument('--gf_opt_astrometry', dest='gf_opt_astrometry', action='store_true')
parser.set_defaults(gf_opt_astrometry=False)

# Refine the centroids of each target?
parser.add_argument('--ref_centers', dest='ref_centers', action='store_true')
parser.set_defaults(ref_centers=True)

args = parser.parse_args()

# Get the project name (see the userdata.dat file):
project = args.project
# Get datafolder/date of the observations that the user wants to reduce:
datafolder = args.datafolder

# Check for which project the user whishes to download data from:
fprojects = open('../userdata.dat','r')
while True:
    line = fprojects.readline()
    if line != '':
        if line[0] != '#':
            cp,cf = line.split()
            cp = cp.split()[0]
            cf = cf.split()[0]
            if project.lower() == cp.lower():
                break
    else:
        print '\t > Project '+project+' is not on the list of saved projects. '
        print '\t   Please associate it on the userdata.dat file.'

out_raw_folder = cf + 'LCOGT/raw/'
out_red_folder = cf + 'LCOGT/red/'

# Define apertures for aperture photometry:
min_aperture = int(args.minap)
max_aperture = int(args.maxap)
aperture_step = int(args.apstep)

get_astrometry = args.get_astrometry
print get_astrometry
ref_centers = args.ref_centers
print ref_centers
gf_opt_astrometry = args.gf_opt_astrometry

###################################################################
print ('\n')
print ('\t ###################################')
print ('\t Pre-processing....')
print ('\t ###################################')

if not os.path.exists(out_red_folder+'/'+datafolder+'/'):
        os.mkdir(out_red_folder+'/'+datafolder+'/')

# Now, organize all observed objects in the given observing night:
files = glob.glob(out_raw_folder+'/'+datafolder+'/*.fits')
files_fz = glob.glob(out_raw_folder+'/'+datafolder+'/*.fits.fz')
for i in range(len(files_fz)):
    fits_name = files_fz[i].split('.fz')[0]
    if fits_name not in files:
        files.append(files_fz[i])
all_objects = [] # This saves all the objects
all_ras = []     # This saves the RA of each object
all_decs = []    # This saves the DEC of each object
object_in_files = len(files)*[''] # This saves what object is on each file

good_objects = []
for i in range(len(files)):
    f = files[i]
    print f
    try:
      d,h = pyfits.getdata(f,header=True)
      if 'Observatory' in h['SITE'].split()[-1]:
          obj_name = h['OBJECT']+'-'+h['FILTER']+'-'+h['SITE'].split()[-2]+h['SITE'].split()[-1]+h['ENCID']
      else:
          obj_name = h['OBJECT']+'-'+h['FILTER']+'-'+h['SITE'].split()[-1]+h['ENCID']
      print obj_name
      object_in_files[i] = obj_name
      if obj_name not in all_objects:
          all_objects.append(obj_name)
          all_ras.append(h['RA'])
          all_decs.append(h['DEC'])
          out_folder = out_red_folder+'/'+datafolder+'/'+obj_name
          if not os.path.exists(out_folder):
               os.mkdir(out_folder)
      good_objects.append(i)
    except:
        print 'File ',f,' is corrupted. Skipping it'
files = [ files[i] for i in good_objects ]
object_in_files = [ object_in_files[i] for i in good_objects ]

print '\t Found ',len(all_objects),' object(s) for the observations under '+datafolder
print '\t They are:',all_objects

print ('\t ###################################')
print ('\t Going to photometric extraction....')
print ('\t ###################################')

# Create apertures:
R = np.arange(min_aperture,max_aperture+1,aperture_step)

# Get photometry for the objects:
for i in range(len(all_objects)):
	obj_name = all_objects[i]
	print '\t Working on '+obj_name
	out_data_folder = out_red_folder+'/'+datafolder+'/'+obj_name+'/'
        all_files = []
        for j in range(len(files)):
            if obj_name == object_in_files[j]:
                all_files.append(files[j])
        # Convert RA and DECs of object to decimal degrees:
        ra_obj,dec_obj = PhotUtils.CoordsToDecimal([[all_ras[i],all_decs[i]]])
        print ra_obj
        print dec_obj            
        if not os.path.exists(out_data_folder+'photometry.pkl'):
           master_dict = None
        else:
           master_dict = pickle.load(open(out_data_folder+'photometry.pkl','r'))

        # Get master dictionary for photometry:
        if 'Inter-American' in obj_name:
           master_dict = PhotUtils.getPhotometry(all_files,'LCOGT',R,ra_obj,dec_obj,out_data_folder,obj_name.split('-')[-3],\
                                              get_astrometry = get_astrometry, refine_cen = ref_centers, master_dict = master_dict,\
                                              gf_opt = gf_opt_astrometry)
        else:
           master_dict = PhotUtils.getPhotometry(all_files,'LCOGT',R,ra_obj,dec_obj,out_data_folder,obj_name.split('-')[-2],\
                                              get_astrometry = get_astrometry, refine_cen = ref_centers, master_dict = master_dict,\
                                              gf_opt = gf_opt_astrometry)

        # Save dictionary:
        print ('\t Saving photometry at '+out_data_folder+'...')
        OUT_FILE = open(out_data_folder+'photometry.pkl','w')
        pickle.dump(master_dict,OUT_FILE)
        OUT_FILE.close() 
