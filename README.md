# lcogtAP

This private repository stores codes to perform aperture photometry on a set of images from 
the LCOGT network. It supports data from all LCOGT sites.

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

Once all of the above is installed, using the code is easy. Simply 
modify the parameters needed for each observation and run the code 
(e.g., get_photometry_lcogt.py). Note that the code requires internet 
connection (because it does a query to 2MASS to get all the sources from 
your image).

POST-PROCESSING USAGE
----------------------

The post-processing usage is simply. A common usage is:

           python transit_photometry.py -project HATSOUTH -datafolder 20160520 -target_name TARGETNAME -ra "14:00:00" -dec " -60:00:00" --plt_images
