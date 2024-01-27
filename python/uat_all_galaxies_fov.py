#!/usr/bin/env python

'''
GOAL:
- get all galaxies within FOV and redshift range of Halpha filters
- use NSA as parent sample
- write out NSA table for surviving galaxies

'''


from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np
import os

homedir = os.getenv("HOME")
##############################################################
# redshifts for targets not in RASSCALS or WBL file
##############################################################
alt_redshift = {b'MKW8s':0.027, b'MKW8':0.027, b'NGC5846':0.00571, b'HCG79':0.01450, b'Coma':0.02310, b'Abell1367':0.02200}

alt_coords = {b'MKW8':[220.159167, 3.476389], b'NGC5846':[226.622017, 1.605625],  b'HCG79':[239.799454, 20.758555]} #RA and Dec

##############################################################
# dictionary of Halpha filters 
##############################################################
lmin={'ha4':6573., 'ha8':6606., 'ha12':6650., 'ha16':6682., 'INT197':6540.5}
lmax={'ha4':6669., 'ha8':6703., 'ha12':6747., 'ha16':6779., 'INT197':6615.5}


##############################################################
#  DEFINE PATHS
##############################################################
catalog_dir = homedir+'/research/HalphaGroups/catalogs/'
nsapath = homedir+'/research/NSA/'

##############################################################
# read in NSA file
##############################################################
nsa = fits.getdata(nsapath+'nsa_v0_1_2.fits')
mstar = fits.getdata(nsapath+'nsa_v1_2_fsps_v2.4_miles_chab_charlot_sfhgrid01.fits')
##############################################################
# read in galaxy tables
##############################################################
rasscals_file = 'RASSCALS_groups_positions.fits'
wbl_file = 'WBL_groups_positions.fits'
groups_file = 'UAT_group_centers.csv'

rasc = fits.getdata(catalog_dir+rasscals_file,1)
wbl = fits.getdata(catalog_dir+wbl_file,1)
gc = ascii.read(catalog_dir+groups_file, delimiter=',')

##############################################################
# get redshift of each group by match to rasccals or wbl file
##############################################################

rascdict = dict((a,b) for a,b in zip(rasc.RASSCALS,np.arange(len(rasc.RASSCALS))))

# wbl names have a space between

for i in range(len(wbl.WBL)):
    wbl.WBL[i] = (wbl.WBL[i].replace(b' ',b''))
wbldict = dict((a,b) for a,b in zip(wbl.WBL,np.arange(len(wbl.WBL))))

# find redshift
redshift = np.zeros(len(gc['Target']),'f')
mosaic_flag = np.ones(len(gc['Target']),'bool')
for i in range(len(gc['Target'])):
    name,numb = str(gc['Target'][i]).split('-')
    name = name.encode('utf-8')
    if numb.find('h') > -1:
        mosaic_flag[i] = False
    try:
        irasc = rascdict[name]
        redshift[i] = float(rasc.cz[irasc])/3.e5
    except KeyError:
        try:
            iwbl = wbldict[name]
            redshift[i] = float(wbl.z[iwbl])
        except KeyError:
            try:
                redshift[i] = alt_redshift[name]
            except:
                print(i, ' no redshift for ',name)
            

##############################################################
# set 'distance' and set z range according to filter throughput
##############################################################
g_distance = np.zeros(len(gc['RA']),'f')
zmin = np.zeros(len(gc['RA']),'f')
zmax = np.zeros(len(gc['RA']),'f')
for i in range(len(gc['RA'])):
    zmax[i]=((lmax[gc['Filter'][i]])/6563.)-1
    zmin[i]=((lmin[gc['Filter'][i]])/6563.)-1
    g_distance[i] = 0.5*(zmin[i]+zmax[i])*3.e5/70.

##############################################################
# declare coord of groups and NSA as SkyCoord so we can use
# the amazing SkyCoord functions
##############################################################
group_coord = SkyCoord(gc['RA'],gc['Dec'], distance=g_distance*u.Mpc, unit=(u.hourangle,u.deg),frame='icrs')
#nsa_coord = SkyCoord(nsa.RA,nsa.DEC, distance=nsa.Z*3.e5/70.*u.Mpc, unit=(u.deg,u.deg),frame='icrs')
nsa_coord = SkyCoord(nsa.RA,nsa.DEC,  unit=(u.deg,u.deg),frame='icrs')

##############################################################
# set FOV size to +/- 0.25 deg for HDI and +/-0.5 deg for Mosaic
##############################################################
dtheta = 0.25*np.ones(len(gc['RA']),'f')
dtheta[mosaic_flag] = 2.*dtheta[mosaic_flag]


##############################################################
# This function is AMAZING!!!
#
# Find all galaxies within 1.45 deg (sqrt(2)*1deg for mosaic, plus a little)
#  of group center to
# limit size of NSA catalog
#
# idxc is the group number
# idxcatalog is the row from NSA catalog
#
# did I mention that this function is AMAZING???
##############################################################
idxc, idxcatalog, d2d, d3d = nsa_coord.search_around_sky(group_coord, 1.45*u.deg)

newnsa = nsa[idxcatalog]

compra = group_coord.ra[idxc]
compdec = group_coord.dec[idxc]
compz = redshift[idxc]
compzmin = zmin[idxc]
compzmax = zmax[idxc]
compdtheta = dtheta[idxc]

##############################################################
# get distance from group centers, not center of pointings
##############################################################


##############################################################
# Now cut again based on square FOV and redshift range
##############################################################

# redshift cut
zflag = (newnsa.Z > compzmin) & (newnsa.Z < compzmax)
# RA cut
rflag = (newnsa.RA > (compra.value - compdtheta)) & (newnsa.RA < (compra.value + compdtheta))
# Dec cut
dflag = (newnsa.DEC > (compdec.value - compdtheta)) & (newnsa.DEC < (compdec.value + compdtheta))


finalnsa = newnsa[(zflag & rflag & dflag)]
finalmstar = mstar[idxcatalog][(zflag & rflag & dflag)]
gname = gc['Target'][idxc][(zflag & rflag & dflag)]
gra = group_coord.ra[idxc][(zflag & rflag & dflag)]
gdec = group_coord.dec[idxc][(zflag & rflag & dflag)]

dist3d = d3d[(zflag & rflag & dflag)]
dist2d = d2d.deg[(zflag & rflag & dflag)]
##############################################################
# Write out nsa table for galaxies w/in Halpha FOV
##############################################################

fits.writeto(catalog_dir+'uat_halpha_nsa.fits',finalnsa, overwrite=True)

fits.writeto(catalog_dir+'uat_halpha_moustakas_mstar.fits',finalmstar, overwrite=True)

##############################################################
# Write out table with extra columns appended
##############################################################

# group_name, group_ra, group_dec, group_redshift, dist_2d, dist_3d, local density

g = Table([gname,gra,gdec,dist2d,dist3d],names=('group_name', 'group_RA', 'group_DEC', 'dist_2d','dist_3d'))
g.write(catalog_dir+'uat_halpha_group_info.fits',overwrite=True)
