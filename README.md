# lcogtAP

This repository stores codes to perform aperture photometry on a set of images from 
the LCOGT network. It supports data from all LCOGT sites.

Author: Néstor Espinoza (nsespino@uc.cl)

DEPENDENCIES
------------

This code makes use of four important python libraries:

    + Numpy.
    + Scipy.
    + Pyfits.
    + Astropy

All of them are open source and can be easily installed in any machine. 

Furthermore, it makes heavy use of the astrometry package, which can be 
downloaded from here: 

                http://astrometry.net/use.html. 

Instructions useful for Mac users on how to install this package might be 
found here: 

    http://www2.lowell.edu/users/massey/Macsoftware.html#Astrom. 

USAGE
------------

Once all of the above is installed, using the code is easy. However, it relies 
on having internet connection:

 1. First, edit the userdata.dat file and fill in a project name and a folder 
    where the data is for that project. It is assumed inside the project 
    folder there is another folder named `LCOGT/raw`, which contains the data 
    for different dates. For example, if the project folder for the 
    project `kpBRIGHTSTARS` is `/data/keyproject/bright_stars/`, it is assumed 
    the folder `/data/keyproject/bright_stars/LCOGT/raw` exists, and contains 
    the data for different dates in separate folders (e.g., 
    `/data/keyproject/bright_stars/LCOGT/raw/20170213` ).

2. From terminal, go to the pipeline folder (`cd pipeline`) and run:

       python automatic_lcogt.py -project NAMEOFTHEPROJECT

   Where `NAMEOFTHEPROJECT` corresponds to the project in the userdata.dat file 
   that you filled in in step 1. This will run the pipeline and save the products 
   under a `red` folder, inside the project's folder (e.g., if the project was 
   `kpBRIGHTSTARS`, products will be saved in `/data/keyproject/bright_stars/LCOGT/red`). 
   The pipeline will generate photometry for all the datasets in the folder for which no 
   photometry is yet available that have dates (which is measured from the folder name, 
   i.e., if the dataset is in `/data/keyproject/bright_stars/LCOGT/red/20170320` it assumes 
   the dataset is from 2017/03/20) whithin one week from today (measured from your computer's 
   date).

If you are interested in what the pipeline does, read the next section. If you don't care, 
move to the "Products" section.

WHAT DOES THE PIPELINE DO?
---------------------------

In the background, what the pipeline does is to use the `pipeline/get_photometry_lcogt.py` 
code to get the photometry for all the stars in the field for all the images. It first 
identifies the objects in the image by doing a query to 2MASS (previous to running an 
astrometric solution with astrometry, if the images don't have it). This photometry is 
stored in a `photometry.pkl` file in the `red` folder.

Once this is done, the code calls the `post_processing/transit_photometry.py` code to generate 
differential photometry for the target star. In order to identify the target star for which 
differential photometry should be performed, the code uses the object name in the headers of 
the images and queries the Kepler webpage, Simbad, or the `manual_object_coords.dat` file in 
this folder (in that order). If the target name is not identified in any of these cases, then 
no differential photometry is performed and you have to perform it on your own. Aperture photometry 
is performed for 5 apertures: opt, 5, 15, 20 and 30 pixels. The `opt` aperture is an optimal aperture 
calculated on the rms of the lightcurve; the `get_photometry_lcogt.py` code does apertures from 5 
pixels to 50 pixels, and the `post_processing/transit_photometry.py` code searches for the aperture 
with the smallest RMS.

It is important to note that the differential photometry performed by the code is based on first 
combining the 10 comparison stars closer in brightness and color to the target, and then generating 
a "super comparison" star with those, which is finally divided to the target star. This is done 
in order to not "touch" the target star, so the target could have systematic effects/trends arising 
from airmass, variability, etc. In general these are useful, because one can later detrend the 
resulting lightcurve with other methods. 

PRODUCTS
--------

The products of the `automatic_lcogt.py` code are stored in the `red` folder for each date and for 
each target. Inside, you will find:

 1. The `photometry.pkl` file, which contains all the aperture photometry for all the stars in the field 
    for all the apertures (5 to 50 pixels).

 2. Folders named `sinistro_ap`, where `ap` is one of the apertures in pixels. The folder `sinistro` (without 
    aperture number) is the folder that contains the optimal aperture (see previous section).

Each of the `sinistro_ap` folders contain two `.dat` files: the one with the `norm_flux.dat` sufix contains the 
times (in BJD), relative flux and error on the relative flux of the (corrected by the comparison stars) target 
star, while the other `.dat` file contains the same in differential magnitudes. 

The folder `LC` is very important, as it contains the differential magnitude of the target and the comparison 
stars in the `.epdlc` format, which is useful if you want to detrend your data or play with other ways of combining 
the extracted fluxes of the target and its comparison stars. This format contains lots of information, but the most 
important are the following columns: (0) file name, (1) times in BJD, (2) relative magnitude of the target, (3) error 
on the magnitude, (17) Centroid (X-axis) on the image, (18) Centroid (Y-axis) on the image, (19) Background flux, (20) 
Error on background flux, (21) (2.35/FWHM)^2 (not useful for defocused images), (22) Hour angle and (23) Zenith angle.

The `post_processing_outputs` folder contains the photometry for all the apertures in `.dat` files, similar to the 
`.dat` files inside the `sinistro_ap` folders. 

Finally, all the other folders contain images in png format of the target and the comparison stars.


POST-PROCESSING USAGE
----------------------

In the eventuality in which the post-processing fails, you can do it yourself inside the `post_processing` folder. 
A common usage is:

           python transit_photometry.py -project NAMEOFTHEPROJECT -datafolder 20160520 -target_name TARGETNAME -ra "14:00:00" -dec " -60:00:00" --plt_images

This will save the photometry in the `red/20160520` folder inside the project `NAMEOFTHEPROJECT` for the target 
`TARGETNAME`.
