""" Search VLASS for a given RA and Dec """

import numpy as np
import subprocess
import os
import glob
import pyfits
import matplotlib.pyplot as plt
from urllib.request import urlopen
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time


def get_tiles():
    """ Get tiles 
    I ran wget https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php
    """

    inputf = open("VLASS_dyn_summary.php", "r")
    lines = inputf.readlines()
    inputf.close()

    header = list(filter(None, lines[0].split("  ")))
    # get rid of white spaces
    header = np.array([val.strip() for val in header])

    names = []
    dec_min = []
    dec_max = []
    ra_min = []
    ra_max = []
    obsdate = []
    epoch = []

    # Starting at lines[3], read in values
    for line in lines[3:]:
        dat = list(filter(None, line.split("  "))) 
        dat = np.array([val.strip() for val in dat]) 
        names.append(dat[0])
        dec_min.append(float(dat[1]))
        dec_max.append(float(dat[2]))
        ra_min.append(float(dat[3]))
        ra_max.append(float(dat[4]))
        obsdate.append(dat[6])
        epoch.append(dat[5])

    names = np.array(names)
    dec_min = np.array(dec_min)
    dec_max = np.array(dec_max)
    ra_min = np.array(ra_min)
    ra_max = np.array(ra_max)
    obsdate = np.array(obsdate)
    epoch = np.array(epoch)

    return (names, dec_min, dec_max, ra_min, ra_max, epoch, obsdate)


def search_tiles(tiles, ra_h, dec_d):
    """ Now that you've processed the file, search for the given RA and Dec """
    names, dec_min, dec_max, ra_min, ra_max, epochs, obsdate = tiles
    has_dec = np.logical_and(dec_d > dec_min, dec_d < dec_max)
    has_ra = np.logical_and(ra_h > ra_min, ra_h < ra_max)
    in_tile = np.logical_and(has_ra, has_dec)
    name = names[in_tile]
    epoch = epochs[in_tile]
    date = obsdate[in_tile]
    if len(name) > 1:
        print("Error: this source is in more than one tile")
    elif len(name) == 0:
        print("Sorry, no tile found.")
        return None, None, None
    else:
        return name[0], epoch[0], date[0]


def get_subtiles(tilename, epoch):
    """ For a given tile name, get the filenames in the VLASS directory.
    Parse those filenames and return a list of subtile RA and Dec.
    RA and Dec returned as a SkyCoord object
    """
    urlpath = urlopen(
        'https://archive-new.nrao.edu/vlass/quicklook/%s/%s/' %(epoch,tilename))
    string = (urlpath.read().decode('utf-8')).split("\n")
    vals = np.array([val.strip() for val in string])
    keep_link = np.array(["href" in val.strip() for val in string])
    keep_name = np.array([tilename in val.strip() for val in string])
    string_keep = vals[np.logical_and(keep_link, keep_name)]
    fname = np.array([val.split("\"")[1] for val in string_keep])
    pos_raw = np.array([val.split(".")[4] for val in fname])
    if '-' in pos_raw[0]:
        # dec < 0
        ra_raw = np.array([val.split("-")[0] for val in pos_raw])
        dec_raw = np.array([val.split("-")[1] for val in pos_raw])
    else:
        # dec > 0
        ra_raw = np.array([val.split("+")[0] for val in pos_raw])
        dec_raw = np.array([val.split("+")[1] for val in pos_raw])
    ra = []
    dec = []
    for ii,val in enumerate(ra_raw):
        hms = "%sh%sm%ss" %(val[1:3], val[3:5], val[5:])
        ra.append(hms)
        dms = "%sd%sm%ss" %(
                dec_raw[ii][0:2], dec_raw[ii][2:4], dec_raw[ii][4:])
        dec.append(dms)
    ra = np.array(ra)
    dec = np.array(dec)
    c = SkyCoord(ra, dec, frame='icrs')
    return fname, c


def get_cutout(imname, name, c):
    # Position of source
    ra_deg = c.ra.deg
    dec_deg = c.dec.deg

    # Open image and establish coordinate system
    im = pyfits.open(imname)[0].data[0,0]
    w = WCS(imname)

    # Find the source position in pixels.
    # This will be the center of our image.
    src_pix = w.wcs_world2pix([[ra_deg, dec_deg, 0, 0]], 0)
    x = src_pix[0,0]
    y = src_pix[0,1]

    # Check if the source is actually in the image
    pix1 = pyfits.open(imname)[0].header['CRPIX1']
    pix2 = pyfits.open(imname)[0].header['CRPIX2']
    badx = np.logical_or(x < 0, x > 2 * pix1)
    bady = np.logical_or(y < 0, y > 2 * pix1)
    if np.logical_or(badx, bady):
        return -1

    # Set the dimensions of the image
    # Say we want it to be 13.5 arcseconds on a side,
    # to match the DES images
    delt1 = pyfits.open(imname)[0].header['CDELT1']
    delt2 = pyfits.open(imname)[0].header['CDELT2']
    cutout_size = 12 / 3600 # in degrees
    dside1 = -cutout_size/2./delt1
    dside2 = cutout_size/2./delt2

    vmin = -1e-4
    vmax = 1e-3

    im_plot = im[int(y-dside1):int(y+dside1),int(x-dside2):int(x+dside2)]
    median_rms = np.median(np.std(im_plot))

    plt.imshow(
            np.flipud(im_plot),
            extent=[-0.5*cutout_size*3600.,0.5*cutout_size*3600.,
                    -0.5*cutout_size*3600.,0.5*cutout_size*3600],
            vmin=vmin,vmax=vmax,cmap='YlOrRd')

    plt.title(name)
    plt.xlabel("Offset in RA (arcsec)")
    plt.ylabel("Offset in Dec (arcsec)")

    plt.savefig(name + ".png") 

    return median_rms


def search_vlass(names, ra, dec, dates):
    """ 
    Searches the VLASS catalog for a list of sources

    Parameters
    ----------
    names: array of names of the sources
    ra: array of RAs in decimal hours
    dec: array of Decs in decimal degrees
    dates: array of dates in astropy Time format
    """

    nobj = len(names)
    limits = np.zeros(nobj)
    obsdates = np.zeros(nobj)

    for ii,name in enumerate(names):
        print("Running for %s" %name)

        # Find the VLASS tile
        tiles = get_tiles()
        tilename, epoch, obsdate = search_tiles(tiles, ra[ii], dec[ii])

        if tilename is None:
            limits[ii] = -1
            obsdates[ii] = -1

        else:
            # The VLASS quicklook site only has 1.1, unfortunately
            if epoch != "VLASS1.1":
                print("Sorry, not available yet")
                limits[ii] = -1
                obsdates[ii] = -1
            else:
                print("Searching for subtile")
                print(tilename, epoch)
                subtiles, c_tiles = get_subtiles(tilename, epoch)
                dist = c.separation(c_tiles)
                subtile = subtiles[np.argmin(dist)]

                url_get = "https://archive-new.nrao.edu/vlass/quicklook/%s/%s/%s" %(
                        epoch, tilename, subtile)
                imname = "%s.I.iter1.image.pbcor.tt0.subim.fits" %subtile[0:-1]
                print(imname)
                if glob.glob(imname):
                    median_flux = get_cutout(imname, name, c)
                else:
                    fname = url_get + imname
                    subprocess.run(["wget", fname])
                    median_flux = get_cutout(imname, name, c)
                print("Upper limit is %s uJy" %(median_flux*1e6))
                limits[ii] = median_flux*1e6
                obsdates[ii] = Time(obsdate, format='iso').mjd
    return limits, obsdates


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=\
        '''
        Searches VLASS for a source.
        User needs to supply name, RA (in decimal hours),
        Dec (in decimal degrees), and (optionally) date (in astropy Time format).
        If there is a date, then will only return VLASS images taken after that date
        (useful for transients with known explosion dates).
        
        Usage: finder.py <Name> <RA [hrs]> <Dec [deg]> <(optional) Date [astropy Time]>
        ''', formatter_class=argparse.RawTextHelpFormatter)
        
    #Check if correct number of arguments are given
    if len(sys.argv) < 3:
        print ("Usage: finder.py <RA> <Dec> <Name>  <rad [deg]> <telescope [P200|Keck]>")
        sys.exit()
     
    name = str(sys.argv[1])
    ra = sys.argv[2]
    dec = sys.argv[3]

    if (len(sys.argv) >= 4):
        date = Time(sys.argv[4])
        print ('Searching for observations after %s' %date)
        search_vlass(name, ra, dec, date) 
    else:
        print ('Searching all obs dates')
        search_vlass(name, ra, dec) 
        
        