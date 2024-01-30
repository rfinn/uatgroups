#!/usr/bin/env python

"""
GOAL: 
measure the biweight location and scale for each group

NOTES:
* did some code development in this notebook

https://github.com/rfinn/uatgroups/blob/main/notebooks/hagroups-center-scale.ipynb

* also cannabalizing from uat_all_galaxies_fov.py, which is trying to get a sample that is within the halpha FOV

"""

from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
from astropy.table import Column
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.stats import biweight_scale,biweight_location
from astropy.stats import sigma_clip

from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt




import numpy as np
import os
import sys
import argparse

##############################################################
# redshifts for targets not in RASSCALS or WBL file
##############################################################
alt_redshift = {'MKW8s':0.027, 'MKW8':0.027, 'NGC5846':0.00571, 'HCG79':0.01450, 'Coma':0.02310, 'Abell1367':0.02200}

alt_coords = {'MKW8':[220.159167, 3.476389],'MKW8s':[220.159167, 3.476389], 'NGC5846':[226.622017, 1.605625],  'HCG79':[239.799454, 20.758555],'Coma':[194.953054, 27.980694],'Abell1367':[176.152083, 19.758889]} #RA and Dec



##############################################################
# functions
##############################################################

def get_biweight(z,nsigma=2):
    """
    PARAMS:
    z : recession vel of galaxies in the vicinity of a group
    nsigma : sigma to use in sigma clipping
    
    RETURN:
    c : biweight center
    s : biweight scale
    masked_data : input array, as masked array, where mask=True indicated clipped data
    """
    from astropy.stats import sigma_clip, bootstrap
    from scipy.stats import scoreatpercentile
    #sigclip = SigmaClip(sigma=scale_cut,maxiters=10,cenfunc=biweight_location,stdfunc=biweight_scale)
    good_data = sigma_clip(z, sigma=nsigma, stdfunc=biweight_scale,cenfunc=biweight_location)
    #print("number rejected: ",np.sum(good_data.mask))
    c = biweight_location(good_data[~good_data.mask])
    s = biweight_scale(good_data[~good_data.mask])
    centers = bootstrap(good_data[~good_data.mask],bootnum=100,bootfunc=biweight_location)
    scales = bootstrap(good_data[~good_data.mask],bootnum=100,bootfunc=biweight_scale)    

    percentiles = [50-34,50,84]# 60% conf interval
    centers_err = scoreatpercentile(centers,percentiles)
    scales_err = scoreatpercentile(scales,percentiles)    
    return centers_err, scales_err, good_data

def plothist(vinput,vr,vcenter,vscale,nbins=10):

    mybins = np.linspace(np.min(vr),np.max(vr),nbins)
    t = plt.hist(vinput,histtype='step',bins=mybins)
    t = plt.hist(vr,bins=mybins,histtype='stepfilled',alpha=.5,hatch='//')    
    # plot a gaussian
    ymin,ymax = plt.ylim()
    xline = np.linspace(vcenter-5*vscale,vcenter+5*vscale,100)

    yline = 1/(vscale*np.sqrt(2*np.pi))*np.exp(-0.5*((xline-vcenter)/vscale)**2)
    scale = .8*ymax/np.max(yline)
    #print(yline)
    plt.plot(xline,scale*yline,'r',alpha=.6)
    plt.axvline(x=vcenter,c='k',alpha=.5)    
    plt.axvline(x=vcenter-3*vscale,ls='--',c='k',alpha=.5)    
    plt.axvline(x=vcenter+3*vscale,ls='--',c='k',alpha=.5)
    # plot number of galaxies used in calculation
    plt.text(0.05,0.85,f"N={np.sum(~vr.mask)}",transform=plt.gca().transAxes,horizontalalignment='left',fontsize=10)
if __name__ == "__main__":


    ###################################################################
    #### SET UP ARGPARSE
    ###################################################################

    parser = argparse.ArgumentParser(description ='Calculate the biweight center and scale for UAT Groups!')
    parser.add_argument('--agc',dest = 'agc', default=False,action='store_true',help='Use the AGC to calculate the velocity dispersion and central velocities.  Otherwise use NSA catalog.')
    

    args = parser.parse_args()
    agcflag = args.agc
    ##############################################################
    #  DEFINE PATHS
    ##############################################################
    homedir = os.getenv("HOME")
    catalog_dir = homedir+'/research/HalphaGroups/catalogs/'
    nsapath = homedir+'/research/NSA/'
    agcpath = homedir+'/research/AGC/'    

    ##############################################################
    # read in NSA file
    ##############################################################
    nsa = fits.getdata(nsapath+'nsa_v0_1_2.fits')
    mstar = fits.getdata(nsapath+'nsa_v1_2_fsps_v2.4_miles_chab_charlot_sfhgrid01.fits')

    ##############################################################
    # read in AGC file
    ##############################################################
    agc = fits.getdata(agcpath+'agcnorthminus1.2019Sep24.fits')


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
        #wbl.WBL[i] = (wbl.WBL[i].replace(b' ',b''))
        wbl.WBL[i] = (wbl.WBL[i].replace(' ',''))
    wbldict = dict((a,b) for a,b in zip(wbl.WBL,np.arange(len(wbl.WBL))))

    # get a list of unique groups
    pointing_list = []
    for i in range(len(gc['Target'])):
        name,numb = str(gc['Target'][i]).split('-')
        pointing_list.append(name)
    group_list = list(set(pointing_list))
    group_list.sort()
    gra = np.zeros(len(group_list),'d')
    gdec = np.zeros(len(group_list),'d')
    redshift = np.zeros(len(group_list),'d')
    # find redshift
    
    for i in range(len(group_list)):
        name=group_list[i]
        try:
            irasc = rascdict[name]
            redshift[i] = float(rasc.cz[irasc])/3.e5
            gra[i] = rasc['_RAJ2000'][irasc]
            gdec[i] = rasc['_DEJ2000'][irasc]            
        except KeyError:
            try:
                iwbl = wbldict[name]
                redshift[i] = float(wbl.z[iwbl])
                gra[i] = wbl['_RAJ2000'][iwbl]
                gdec[i] = wbl['_DEJ2000'][iwbl]            
                
            except KeyError:
                try:
                    redshift[i] = alt_redshift[name]
                    gra[i],gdec[i] = alt_coords[name]
                    
                except KeyError:
                    print(i, ' no redshift for ',name)

    ##############################################################
    # get angular size corresponding to 1.5 Mpc at each cluster
    ##############################################################
    # returns value in Mpc per radian
    DA_Mpcrad = cosmo.angular_diameter_distance(redshift)
    # convert to Mpc per deg
    DA_Mpcdeg = DA_Mpcrad.value/180*np.pi

    # define physical separation to use in center/scale calculations
    dR_Mpc = 1.7 # Mpc
    dR_deg = dR_Mpc/DA_Mpcdeg
    #print("dR_deg = ",dR_deg)
    dv=3000.
    dz = dv/3e5 # +/- 4000 km/s
    ##############################################################
    # loop through groups

    ##############################################################
    ngal = np.zeros(len(redshift),'i')    
    biweight_center = np.zeros(len(redshift),'d')
    
    biweight_center_err_up = np.zeros(len(redshift),'d')
    biweight_center_err_down = np.zeros(len(redshift),'d')
    
    biweight_sigma = np.zeros(len(redshift),'d')
    biweight_sigma_err_up = np.zeros(len(redshift),'d')
    biweight_sigma_err_down = np.zeros(len(redshift),'d')        

    plt.figure(figsize=(14,14))
    plt.subplots_adjust(hspace=.6,wspace=.4)
    for i in range(len(redshift)):
    #for i in [1]:
        #print()
        # find galaxies within 1.5 Mpc and +/- 4000 km/s
        #print(dR_deg[i],dz)

        if agcflag:
            # compare with NSA
            dr_flag = np.sqrt((gra[i]-agc['radeg'])**2 + (gdec[i]-agc['decdeg'])**2) < dR_deg[i]
            dz_flag = ((redshift[i]*3.e5-agc['vopt']) < dv) & ((agc['vopt']-redshift[i]*3.e5) < dv) & (agc['vopt'] > 0)

            # find location and scale
            flag = dr_flag & dz_flag
            vinput = agc['vopt'][flag]
        else:
            # compare with NSA
            dr_flag = np.sqrt((gra[i]-nsa['RA'])**2 + (gdec[i]-nsa['DEC'])**2) < dR_deg[i]
            dz_flag = ((redshift[i]-nsa['Z'])*3.e5 < dv) & ((nsa['Z']-redshift[i])*3.e5 < dv)

            # find location and scale
            flag = dr_flag & dz_flag
            vinput = nsa['Z'][flag]*3.e5
        if np.sum(flag) == 0:
            print(f"WARNING: no galaxies found for {group_list[i]}")
            plt.subplot(8,8,i+1)
            plt.title(group_list[i])
            continue
        else:
            #print(f"number of remaining galaxies = {np.sum(flag)}")
            if i == 5: # check NGC5846
                print(vinput)
            c,s, t = \
                get_biweight(vinput,nsigma=3)
            
            clower,biweight_center[i],cupper=c
            biweight_center_err_down[i] = c[1]-c[0]
            biweight_center_err_up[i] = c[2]-c[1]
            
            slower,biweight_sigma[i],supper=s
            biweight_sigma_err_down[i] = s[1]-s[0]
            biweight_sigma_err_up[i] = s[2]-s[1]            

            ngal[i] = np.sum(flag)
            #print(f"ninput={len(vinput)},ngood={np.sum(~t.mask)},nreturn={len(t)}")
            plt.subplot(8,8,i+1)
            plothist(vinput,t,biweight_center[i],biweight_sigma[i])
            #plt.axvline(x=redshift[i]*3.e5,c='c')
            #plt.axvline(x=redshift[i]*3.e5-dv,c='c',ls=':')
            #plt.axvline(x=redshift[i]*3.e5+dv,c='c',ls=':')            
            plt.title(group_list[i])
            #plt.xlim((redshift[i]-1.*dz)*3.e5,(redshift[i]+1.*dz)*3.e5)
            print(f"{group_list[i]:9s}:N={np.sum(flag):3d},vr={biweight_center[i]:4.0f}+{cupper-c[1]:3.0f}-{c[1]-c[0]:3.0f}, sigma={biweight_sigma[i]:4.0f}+{supper-s[1]:3.0f}-{s[1]-s[0]:3.0f}")
            
    plt.savefig('velhists.png')
    # save the table
    colnames = ['Group','vr_ref','RA','DEC','ngal',\
                'biweight_center','biweight_center_err_down','biweight_center_err_up',\
                'biweight_sigma','biweight_sigma_err_down','biweight_sigma_err_up']
    tabcols = [group_list,redshift*3.e5,gra,gdec,ngal,\
                    biweight_center,biweight_center_err_down,biweight_center_err_up,\
                    biweight_sigma,biweight_sigma_err_down,biweight_sigma_err_up]

    newtab = Table(tabcols,names=colnames)
    if agcflag:
        outfile = 'uat_groups_center_scale_agc.fits'
    else:
        outfile = 'uat_groups_center_scale_nsa.fits'
    newtab.write(outfile,format='fits',overwrite=True)
