import numpy as np
import subprocess
import os
import sys
import argparse
import glob
import time
import psutil
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from urllib.request import urlopen
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time
from contextlib import closing


fname = "VLASS_dyn_summary.php"

import urllib.request

url = 'https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php'
output_file = 'VLASS_dyn_summary.php'

urllib.request.urlretrieve(url, output_file)

print(f'File downloaded to: {output_file}')

def get_tiles():
    """ Get tiles 
    I ran wget https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php
    """

    inputf = open(fname, "r")
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

def search_tiles(tiles, c):
    """ Now that you've processed the file, search for the given RA and Dec
    
    Parameters
    ----------
    c: SkyCoord object
    """
    ra_h = c.ra.hour
    dec_d = c.dec.deg
    names, dec_min, dec_max, ra_min, ra_max, epochs, obsdate = tiles
    has_dec = np.logical_and(dec_d > dec_min, dec_d < dec_max)
    has_ra = np.logical_and(ra_h > ra_min, ra_h < ra_max)
    in_tile = np.logical_and(has_ra, has_dec)
    name = names[in_tile]
    epoch = epochs[in_tile]
    date = obsdate[in_tile]
    if len(name) == 0:
        print("Sorry, no tile found.")
        return None, None, None
    else:
        return name, epoch, date

def get_subtiles(tilename, epoch):
    """ For a given tile name, get the filenames in the VLASS directory.
    Parse those filenames and return a list of subtile RA and Dec.
    RA and Dec returned as a SkyCoord object
    """
    if epoch =='VLASS1.2':
        epoch = 'VLASS1.2v2'
    elif epoch =='VLASS1.1':
        epoch = 'VLASS1.1v2'
    url_full = 'https://archive-new.nrao.edu/vlass/quicklook/%s/%s/' %(epoch,tilename)
    print(url_full)
    urlpath = urlopen(url_full)
    # Get site HTML coding
    string = (urlpath.read().decode('utf-8')).split("\n")
    # clean the HTML elements of trailing and leading whitespace
    vals = np.array([val.strip() for val in string])
    # Make list of HTML link elements
    keep_link = np.array(["href" in val.strip() for val in string])
    # Make list of HTML elements with the tile name
    keep_name = np.array([tilename in val.strip() for val in string])
    # Cross reference the two lists above to keep only the HTML elements with the tile name and a link
    string_keep = vals[np.logical_and(keep_link, keep_name)]
    # Keep only the links from the HTML elements (they are the 7th element since 6 quote marks precede it)
    fname = np.array([val.split("\"")[7] for val in string_keep])
    # Take out the element of the link that encodes the RA and declination
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
        # 24 hours is the same as hour 0
        if val[1:3] == '24':
            rah = '00'
        else:
            rah = val[1:3]
        # calculate RA in hours mins and seconds
        hms = "%sh%sm%ss" %(rah, val[3:5], val[5:])
        ra.append(hms)
        # calculate Dec in degrees arcminutes and arcseconds
        dms = "%sd%sm%ss" %(
                dec_raw[ii][0:2], dec_raw[ii][2:4], dec_raw[ii][4:])
        dec.append(dms)
    ra = np.array(ra)
    dec = np.array(dec)
    c_tiles = SkyCoord(ra, dec, frame='icrs')
    return fname, c_tiles

def get_cutout(imname, name, c, epoch, save_dir="images"):
    print("Generating cutout")
    # Position of source
    ra_deg = c.ra.deg
    dec_deg = c.dec.deg

    print("Cutout centered at position %s, %s" % (ra_deg, dec_deg))

    # Open image and establish coordinate system
    hdul = pyfits.open(imname)
    im = hdul[0].data[0, 0]
    w = WCS(hdul[0].header)
    hdul.close()

    # Find the source position in pixels.
    # This will be the center of our image.
    src_pix = w.wcs_world2pix([[ra_deg, dec_deg, 0, 0]], 0)
    x = src_pix[0, 0]
    y = src_pix[0, 1]

    # Check if the source is actually in the image
    pix1 = hdul[0].header['CRPIX1']
    pix2 = hdul[0].header['CRPIX2']
    badx = np.logical_or(x < 0, x > 2 * pix1)
    bady = np.logical_or(y < 0, y > 2 * pix2)
    if np.logical_and(badx, bady):
        print("Tile has not been imaged at the position of the source")
        return None
    else:
        # Set the dimensions of the image
        # Say we want it to be 12 arcseconds on a side,
        # to match the DES images
        image_dim_arcsec = 12
        delt1 = hdul[0].header['CDELT1']
        delt2 = hdul[0].header['CDELT2']
        cutout_size = image_dim_arcsec / 3600  # in degrees
        dside1 = -cutout_size / 2. / delt1
        dside2 = cutout_size / 2. / delt2

        vmin = -1e-4
        vmax = 1e-3

        im_plot_raw = im[int(y - dside1):int(y + dside1), int(x - dside2):int(x + dside2)]
        im_plot = np.ma.masked_invalid(im_plot_raw)

        # 3-sigma clipping (find root mean square of values that are not above 3 standard deviations)
        rms_temp = np.ma.std(im_plot)
        keep = np.ma.abs(im_plot) <= 3 * rms_temp
        rms = np.ma.std(im_plot[keep])

        # Find peak flux in entire image
        # Check if im_plot.flatten() is empty
        if im_plot.flatten().size == 0:
            print("Tile has not been imaged at the position of the source")
            return None
        else:
            peak_flux = np.ma.max(im_plot.flatten())

        fig, ax = plt.subplots(figsize=(6, 6))  # Create a square figure
        ax.imshow(
            np.flipud(im_plot),
            extent=[-0.5 * cutout_size * 3600., 0.5 * cutout_size * 3600.,
                    -0.5 * cutout_size * 3600., 0.5 * cutout_size * 3600],
            vmin=vmin, vmax=vmax, cmap='YlOrRd')

        peakstr = "Peak Flux %s mJy" % (np.round(peak_flux * 1e3, 3))
        rmsstr = "RMS Flux %s mJy" % (np.round(rms * 1e3, 3))

        title_str = r'$\bf{%s}$' % epoch + '\n' + '%s: %s;\n%s' % (name, peakstr, rmsstr)
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel("Offset in RA (arcsec)")
        ax.set_ylabel("Offset in Dec (arcsec)")

        ax.set_aspect('equal')  # Ensure the plot is square
        ax.figure.tight_layout()  # Adjust layout to fit everything nicely

        filename = f"{name}_{epoch}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)

        print(f"PNG saved successfully: {filepath}")

    return peak_flux, rms, filepath

def run_search(name, c, date=None):
    """ 
    Searches the VLASS catalog for a source

    Parameters
    ----------
    names: name of the sources
    c: coordinates as SkyCoord object
    date: date in astropy Time format
    """
    print("Running for %s" %name)
    print("Coordinates %s" %c)
    print("Date: %s" %date)

    # Find the VLASS tile(s)
    tiles = get_tiles()
    tilenames, epochs, obsdates = search_tiles(tiles, c)

    # Sort the tiles by the epochs so most recent goes last
    combined = list(zip(tilenames, epochs, obsdates))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    tilenames, epochs, obsdates = zip(*combined_sorted)

    past_epochs = ["VLASS1.1v2", "VLASS1.2v2", "VLASS2.1", "VLASS2.2", "VLASS3.1"]
    current_epoch = "VLASS3.2"

    results = []
    list_epochs = []
    list_dates = []

    if tilenames[0] is None:
        print("There is no VLASS tile at this location")
        results = ['images\\unimaged.png', 'images\\unimaged.png', 'images\\unimaged.png']
        list_epochs = ['NA', 'NA', 'NA']
        list_dates = ['NA', 'NA', 'NA']
        return results, list_epochs, list_dates

    else:
        for ii,tilename in enumerate(tilenames):
            start_time = time.time()
            print()
            print("Looking for tile observation for %s" %tilename)
            epoch = epochs[ii]
            obsdate = obsdates[ii]
            if obsdate != 'Scheduled':
                obsdate = Time(obsdate, format = 'iso')
            # Adjust name so it works with the version 2 ones for 1.1 and 1.2
            if epoch=='VLASS1.2':
                epoch = 'VLASS1.2v2'
            elif epoch =='VLASS1.1':
                epoch = 'VLASS1.1v2'

            observed = True

            if epoch not in past_epochs:
                if epoch == current_epoch:
                    # Make list of observed tiles 
                    url_full = 'https://archive-new.nrao.edu/vlass/quicklook/%s/' %(epoch)
                    with closing(urlopen(url_full)) as urlpath:
                        # Get site HTML coding
                        string = (urlpath.read().decode('utf-8')).split("\n")
                        # clean the HTML elements of trailing and leading whitespace
                        vals = np.array([val.strip() for val in string])
                        # Make list of useful html elements
                        files = np.array(['alt="[DIR]"' in val.strip() for val in string])
                        useful = vals[files]
                        # Splice out the name from the link
                        obsname = np.array([val.split("\"")[7] for val in useful])
                        observed_current_epoch = np.char.replace(obsname, '/', '')

                    # Check if tile has been observed yet for the current epoch
                    if epoch not in observed_current_epoch:
                        observed = False
                        results.append('images\\unimaged.png')
                        list_epochs.append(epoch)
                        list_dates.append(obsdate)
                        print("Sorry, tile will be observed later in this epoch")
    
                else:
                    observed = False
                    results.append('images\\unimaged.png')
                    list_epochs.append(epoch)
                    list_dates.append(obsdate)
                    print("Sorry, tile will be observed in a later epoch")
                    
            if observed:
                print("Tile Found:")
                print(tilename, epoch)
                subtiles, c_tiles = get_subtiles(tilename, epoch)
                # Find angular separation from the tiles to the location
                dist = c.separation(c_tiles)
                # Find tile with the smallest separation 
                subtile = subtiles[np.argmin(dist)]
                url_get = "https://archive-new.nrao.edu/vlass/quicklook/%s/%s/%s" %(
                        epoch, tilename, subtile)
                imname="%s.I.iter1.image.pbcor.tt0.subim.fits" %subtile[0:-1]
                fname = url_get + imname
                print(fname)
                png_name = "images\\" + name + "_" + epoch + ".png"
                if os.path.exists(png_name):
                    print(f"PNG file {png_name} already exists. Skipping download.")
                else:
                    cmd = "curl -O %s" %fname
                    print(cmd)
                    os.system(cmd)
                    out = get_cutout(imname, name, c, epoch)
                # Get image cutout and save FITS data as png
                    if out is not None:
                        peak, rms, png_name = out
                        print("Peak flux is %s uJy" %(peak*1e6))
                        print("RMS is %s uJy" %(rms*1e6))
                        limit = rms*1e6
                        print("Tile observed on %s" %obsdate)
                        print(limit, obsdate)
                    else:
                        png_name = "images\\unimaged.png"
                        print("Sorry, tile has not been imaged at the position of the source")

                # Wait briefly to ensure handles are closed
                # time.sleep(1)

                # Attempt to delete the file
                # if os.path.exists(imname):
                #     if delete_file(imname):
                #        # File was successfully deleted
                #        print("yay")
                #    else:
                #    # Handle deletion failure as needed
                #        pass
                # append list elements
                results.append(png_name)
                list_epochs.append(epoch)
                list_dates.append(obsdate)  

                end_time = time.time()
                duration = end_time - start_time
                print(f"Run search completed in {duration:.2f} seconds.")

        return results, list_epochs, list_dates

def delete_file(file_path):
    """
    Attempt to delete the file, retrying if it fails initially.
    """
    max_retries = 3
    retry_delay = 1  # seconds

    for retry in range(max_retries):
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
            return True  # File deleted successfully
        except Exception as e:
            print(f"Error deleting file: {e}")
            print(f"Retrying ({retry + 1}/{max_retries})...")
            time.sleep(retry_delay)

    print(f"Failed to delete {file_path} after {max_retries} retries.")
    return False  # File deletion failed               


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=\
        '''
        Searches VLASS for a source.
        User needs to supply name, RA (in decimal degrees),
        Dec (in decimal degrees), and (optionally) date (in mjd).
        If there is a date, then will only return VLASS images taken after that date
        (useful for transients with known explosion dates).
        
        Usage: vlass_search.py <Name> <RA [deg]> <Dec [deg]> <(optional) Date [mjd]>
        ''', formatter_class=argparse.RawTextHelpFormatter)
        
    #Check if correct number of arguments are given
    if len(sys.argv) < 3:
        print("Usage: vlass_search.py <Name> <RA [deg]> <Dec [deg]> <(optional) Date [astropy Time]>")
        sys.exit()
     
    name = str(sys.argv[1])
    ra = float(sys.argv[2])
    dec = float(sys.argv[3])
    c = SkyCoord(ra, dec, unit='deg')

    if glob.glob("/Users/annaho/Dropbox/astro/tools/Query_VLASS/VLASS_dyn_summary.php"):
        pass
    else:
        print("Tile summary file is not here. Download it using wget:\
               wget https://archive-new.nrao.edu/vlass/VLASS_dyn_summary.php")

    if (len(sys.argv) > 4):
        date = Time(float(sys.argv[4]), format='mjd')
        print ('Searching for observations after %s' %date)
        run_search(name, c, date) 
    else:
        print ('Searching all obs dates')
        run_search(name, c) 