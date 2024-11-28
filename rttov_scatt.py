import numpy as np
from numpy import ma
import netCDF4
import sys
import os
import glob
from datetime import datetime, timedelta
import argparse
from metpy.calc import mixing_ratio_from_relative_humidity, density, relative_humidity_from_specific_humidity
from metpy.units import units, masked_array

rttov_installdir = 'rttov/'

rttov_wrapper_path = os.path.join(rttov_installdir, 'lib')
rttov_python_wrapper_path = os.path.join(rttov_installdir, 'wrapper')
sys.path.append(rttov_wrapper_path)
sys.path.append(rttov_python_wrapper_path)
import pyrttov


parser = argparse.ArgumentParser(description='Dataset generator')
parser.add_argument('--begin', type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=False, help='Begin date in YYYY/MM/DD format', default=datetime(2014, 4, 2))
parser.add_argument('--end',   type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=False, help='End date in YYYY/MM/DD format',   default=datetime(2014, 4, 3))
args = parser.parse_args()

DAY_BEGIN = args.begin
DAY_END   = args.end

OUTPUT_DIR = '/mnt/nas04/mykhailo/atlas_data'


########## Functions ############## 
def extrapolate_bottom(a2in, a1idxbot, a1idxy, coef):
    a1valbot = a2in[a1idxy, a1idxbot] # value at the surface layer
    a2valfill = coef*(org2z - a1zbot.reshape(-1,1)) + a1valbot.reshape(-1,1) # extrapolated = coef*height + base
    a2out = a2in.copy()
    a2out[a2in.mask] = a2valfill[a2in.mask]
    return a2out

def fill_bottom(a2in, a1idxbot, a1idxy):
    nx, ny = a2in.shape
    a1valbot = a2in[a1idxy, a1idxbot] # value at the surface layer
    # a2valbot = np.array([a1valbot.tolist() for i in range(ny)]).T
    a2valbot = np.tile(a1valbot, (ny, 1)).T
    a2out = a2in.copy()
    a2out[a2in.mask] = a2valbot[a2in.mask]
    return a2out

def tb2rad(T, nu):
    '''
    Planck function
    T  : K
    nu : wavenumber [cm-1]
    rad: mW m-2 sr-1 (cm-1)^-1 (original radiance unit of RTTOV)
    '''
    c1 = 1.1909e-8  # W m-2 sr-1 cm-1 cm3
    c2 = 1.438786   # cm K
    return c1*nu**3 / (np.exp(c2*nu/T) -1.0) * 1e+3

def rad2tbPlanck(rad, nu):
    '''
    rad: mW m-2 sr-1 (cm-1)^-1 (original radiance unit of RTTOV)
    nu : wavenumber [cm-1]
    '''
    rad = rad*1e-3
    c1 = 1.1909e-8  # W m-2 sr-1 cm-1 cm3
    c2 = 1.438786   # cm K
    denom = np.log(c1*(nu**3)/rad +1.0)
    return c2*nu / denom

def abs_to_rel_density(orig_dens, orig_units, pressure, pressure_units, temperature, temperature_units, relative_humidity):
    mixing_ratio = mixing_ratio_from_relative_humidity(pressure_units*pressure, temperature_units*temperature, relative_humidity)
    air_density = density(pressure_units*pressure, temperature_units*temperature, mixing_ratio)
    return (orig_units*orig_dens)/((air_density + orig_units*orig_dens).to(orig_units))

def make_half_levels(p, sp):
    p_left = np.hstack([p, np.zeros((nprofiles, 1), dtype=np.float64)])
    p_right = np.hstack([np.zeros((nprofiles, 1), dtype=np.float64), p])
    p_half = (p_left + p_right) / 2
    p_half[:,nlevels] = sp
    p_half[:,0] = np.zeros(nprofiles)
    
    # ??? RTTOV-SCATT requires level=(last) to be exactly surface pressure
    # Sometimes level=(last-1) is larger, than surface, in that case make (last-1) average surface and level=(last-2)
    # p_half_bad = p_half[:,nlevels] <= p_half[:,nlevels-1]
    # p_half[p_half_bad, nlevels-1] = ((p_half[:,nlevels]+p_half[:,nlevels-2])/2)[p_half_bad]
    # Or maybe make it ~same as (last) ?
    for i in range(nlevels-1, -1, -1):
        p_half[:, i] = np.minimum(p_half[:, i], p_half[:, i+1]-0.0001)

    assert(p_half.shape[0] == p.shape[0] and p_half.shape[1] == p.shape[1]+1)

    return p_half


if __name__ == '__main__':
    ldtime = np.arange(idtime, edtime, timedelta(days=1)).tolist()

    # Initialize model
    gmiRttov = pyrttov.RttovScatt()
    gmiRttov.FileCoef = os.path.join(rttov_installdir, 'rtcoef_rttov13', 'rttov13pred54L', 'rtcoef_gpm_1_gmi.dat')
    gmiRttov.FileHydrotable = os.path.join(rttov_installdir, 'rtcoef_rttov13', 'hydrotable', 'hydrotable_gpm_gmi.dat')

    gmiRttov.Options.AddInterp = True
    gmiRttov.Options.StoreTrans = True
    gmiRttov.Options.VerboseWrapper = False
    gmiRttov.Options.LuserCfrac = False
    gmiRttov.Options.StoreRad2 = True
    gmiRttov.Options.StoreRad = True

    try:
        gmiRttov.loadInst()
    except pyrttov.RttovError as e:
        sys.stderr.write(f'error loading instrument(s): {e!s}\n')
        sys.exit(1)

    # Start daily loop
    for dtime in ldtime:
        print('processing', dtime)
        year, mon, day = dtime.year, dtime.month, dtime.day

        ssearch = f'/media/nisioji/nas03/data/PMM/NASA/CSATGPM.COIN/2B/V05/{year:04}/{mon:02}/2B.CSATGPM.COIN.COIN2022.{year:04}{mon:02}{day:02}*.nc4'
        lpath = sorted(glob.glob(ssearch))

        for ncpath in lpath[:]:
            print(ncpath)
            label = ncpath.split('.')[-4]
            oid = ncpath.split('.')[-3]

            try:
                with netCDF4.Dataset(ncpath, 'r') as nc:
                    org1ts  = nc['ECMWF-AUX'].variables['Skin_temperature'][:]
                    org1ps  = nc['ECMWF-AUX'].variables['Surface_pressure'][:] *0.01 # Pa->hPa
                    org1t2m = nc['ECMWF-AUX'].variables['Temperature_2m'][:]
                    org1u10 = nc['ECMWF-AUX'].variables['U10_velocity'][:]
                    org1v10 = nc['ECMWF-AUX'].variables['V10_velocity'][:]
                    org2ta  = nc['ECMWF-AUX'].variables['Temperature'][:]
                    org2sh  = nc['ECMWF-AUX'].variables['Specific_humidity'][:]
                    org2pa  = nc['ECMWF-AUX'].variables['Pressure'][:] * 0.01 # Pa->hPa
                    org1lat = nc['2B-GEOPROF'].variables['Latitude'][:]
                    org1lon = nc['2B-GEOPROF'].variables['Longitude'][:]
                    org1elev= nc['2B-GEOPROF'].variables['DEM_elevation'][:] * 0.001 # m->km
                    org2z   = nc['2B-GEOPROF'].variables['Height'][:] * 0.001 # m->km

                    # hydrometeors
                    org2clw = nc['2C-RAIN-PROFILE'].variables['cloud_liquid_water'][:].astype('float64')
                    org2ciw = nc['2C-ICE'].variables['IWC'][:].astype('float64')
                    org2rain = nc['2C-RAIN-PROFILE'].variables['precip_liquid_water'][:].astype('float64')
                    org2snow = nc['2C-RAIN-PROFILE'].variables['precip_ice_water'][:].astype('float64')
                    org2cloud = nc['2B-GEOPROF-LIDAR'].variables['CloudFraction'][:] * 0.01 # %->0-1
                    try:
                        cdate = nc.center_date_CSat
                    except:
                        cdate = nc.predicted_date_CSat
                    cdate = datetime.strptime(cdate, '%Y/%m/%d %H:%M:%S')
            except Exception as e:
                sys.stderr.write(f'incomplete input data from CSATGPM file: {ncpath}, {e!s}\n')
                continue

            datyear, datmon, datday = cdate.year, cdate.month, cdate.day
            
            # Mak invalid Z and ECMWF-AUX
            a1mask_z = np.any(ma.masked_invalid(org2z).mask, axis=1)
            a1mask_ps = ma.masked_invalid(org1ps).mask
            a1mask = a1mask_z | a1mask_ps

            # ?
            if np.any(a1mask):
                print('misisng')
                a1zmean  = org2z.mean(axis=0)
                a1tamean = org2ta.mean(axis=0)
                a1shmean = org2sh.mean(axis=0)
                a1pamean = org2pa.mean(axis=0)
                org1ts[a1mask]  = 273
                org1ps[a1mask]  = 1000
                org1t2m[a1mask] = 273
                org1u10[a1mask] = 2
                org1v10[a1mask] = 2
                org2z[a1mask,:]  = a1zmean
                org2ta[a1mask,:] = a1tamean
                org2sh[a1mask,:] = a1shmean
                org2pa[a1mask,:] = a1pamean
            
            # Ocean is -9999.0
            org1elev[org1elev==-9.999] = 0 

            # Extrapolation
            a1idxbot = org2pa.argmax(axis=1) # indexes of the surface layer (vertical)
            a1idxy = np.arange(a1idxbot.shape[0]) # indexes of the surface layer (horizontal)
            a1zbot = org2z[a1idxy, a1idxbot] # height of the surface level

            # Extrapolate temperature [K]
            coef = -6.5*0.001  # K/m
            a2ta = extrapolate_bottom(org2ta, a1idxbot, a1idxy, coef)

            # Extrapolate pressure [Pa]
            coef = -1.293 * 9.8  # kg/m2 * m/s2
            a2pa = extrapolate_bottom(org2pa, a1idxbot, a1idxy, coef)
            np.save('org2pa.npy', org2pa.data)

            # Extrapolate specific humidity [kg/kg]
            a2sh = fill_bottom(org2sh, a1idxbot, a1idxy)

            # Bottom level specific humidity [kg/kg]
            a1shs = a2sh[a1idxy, a1idxbot]
            try:
                # Convert to relative density (kg/kg) <-> (g/g)
                relative_humidity = relative_humidity_from_specific_humidity(units.hPa*org2pa, units.K*org2ta, org2sh)
                org2clw = abs_to_rel_density(org2clw, units('g/m^3'), org2pa, units.hPa, org2ta, units.K, relative_humidity)
                org2ciw = abs_to_rel_density(org2ciw, units('g/m^3'), org2pa, units.hPa, org2ta, units.K, relative_humidity)
                org2rain = abs_to_rel_density(org2rain, units('g/m^3'), org2pa, units.hPa, org2ta, units.K, relative_humidity)
                org2snow = abs_to_rel_density(org2snow, units('g/m^3'), org2pa, units.hPa, org2ta, units.K, relative_humidity)
            except Exception as e:
                sys.stderr.write(f'error convert to relative density: {e!s}\n')
                continue

            # Replace invalid radar product with 0
            org2clw[org2clw.mask] = 0
            org2ciw[org2ciw.mask] = 0
            org2rain[org2rain.mask] = 0
            org2snow[org2snow.mask] = 0
            org2cloud[org2cloud.mask] = 0
            org2cloud[org2cloud<0] = 0

            # Clip bottoms
            idxlowest = a1idxbot.max()
            a2ta = a2ta[:,:idxlowest+1]
            a2pa = a2pa[:,:idxlowest+1]
            a2sh = a2sh[:,:idxlowest+1]
            a2z  = org2z[:,:idxlowest+1]
            a2clw = org2clw[:,:idxlowest+1]
            a2iciw = org2ciw[:,:idxlowest+1]
            a2rain = org2rain[:,:idxlowest+1]
            a2snow = org2snow[:,:idxlowest+1]
            a2cloud = org2cloud[:,:idxlowest+1]


            # Set up the profile data
            # Declare an instance of Profiles
            nlevels = a2pa.shape[1]
            nprofiles = a2pa.shape[0]
            myProfiles = pyrttov.ProfilesScatt(nprofiles, nlevels)

            myProfiles.GasUnits = 1 # 1: kg/kg over moist air (default)
            myProfiles.P = a2pa
            myProfiles.T = a2ta
            myProfiles.Q = a2sh
            
            # ?
            myProfiles.Angles = [[52.821, 0] for i in range(nprofiles)]  # zenangle, azangle  (<=89GH)

            # Make S2m
            as2m = np.array([org1ps, org1t2m, a1shs, org1u10, org1v10]).T
            myProfiles.S2m = as2m  # (s2m%p, s2m%t, s2m%q, s2m%u, s2m%v)

            # Make Skin
            a2skin = np.array([org1ts, [3.5e+1]*nprofiles, [0]*nprofiles, [3.0e+0]*nprofiles, [5.0e+0]*nprofiles, [1.5e+1]*nprofiles, [1.0e-1]*nprofiles, [3.0e-1]*nprofiles]).T
            myProfiles.Skin =  a2skin # (skin%t, skin%salinity, skin%foam_fraction, skin%fastem(1:5))   # (nprofiles,8)

            # Make SurfType from autosnow
            try:
                a1surftype = np.fromfile(f'/mnt/nas04/utsumi/PMM/MATCH.CSAT.R05/jdb.GPM.GMI.V05.new/{year:04}/{mon:02}/{day:02}/autosnow.{label}.{oid}.int8.bin', dtype='int8')
            except Exception as e:
                sys.stderr.write(f'error loading surftype atlas: {e!s}\n')
                continue
            # Autosnow: {0:'water', 1:'snow-free land', 2:'snow-covered land', 3:'ice'}
            # RTTOV: {0:'land', 1:'sea', 2:'seaice'}
            a1surftype[a1surftype==0] = -1
            a1surftype[a1surftype==1] = 0
            a1surftype[a1surftype==2] = 0
            a1surftype[a1surftype==3] = -2
            a1surftype *= -1
            a1surftype[a1surftype > 2] = 0 # Unknown surftype
            myProfiles.SurfType = a1surftype

            # Make SurfGeom
            a2surfgeom = np.array([org1lat, org1lon, org1elev]).T
            myProfiles.SurfGeom = a2surfgeom  # (latitude, longitude, elevation)

            # Make datetimes
            a2datetimes = np.tile([datyear, datmon, datday, 0, 0, 0], (nprofiles, 1))
            myProfiles.DateTimes = a2datetimes  # (year, month, day, hour, minute, second). The time is not currently used by RTTOV, so can be zero

            # Make hydrometeors
            a2ph = make_half_levels(a2pa, org1ps)
            myProfiles.Ph = a2ph

            try:
                myProfiles.HydroFrac = a2cloud
                myProfiles.Clw = a2clw
                myProfiles.Ciw = a2iciw
                myProfiles.Snow = a2snow
                myProfiles.Rain = a2rain
            except:
                sys.stderr.write(f'error setting profiles: {e!s}\n')
                continue

            # dump('org2cloud.npy', org2cloud)
            # dump('org2clw.npy', org2clw)
            # dump('org2ciw.npy', org2ciw)
            # dump('org2rain.npy', org2rain)
            # dump('org2snow.npy', org2snow)
            # Associate the profiles with each Rttov instance
            gmiRttov.Profiles = myProfiles

            # Load the emissivity and BRDF atlases
            mwAtlas = pyrttov.Atlas()
            mwAtlas.AtlasPath = os.path.join(rttov_installdir, 'emis_data', 'telsem2')
            mwAtlas.loadMwEmisAtlas(a2datetimes[0][1])

            # Channel type loop (first 9 channels and the last 4 channels)
            a2tb_clear_sim = np.empty([nprofiles, 13], dtype='float64')
            a2tb_sim = np.empty([nprofiles, 13], dtype='float64')
            a2emis_fg= np.empty([nprofiles, 13], dtype='float64')

            for chtype in [0, 1]:
                if chtype == 0:
                    channels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    nchan = len(channels)
                    gmiRttov.Profiles.Angles[:,0] = 52.821  # ~89GHz
                elif chtype == 1:
                    channels = [10, 11, 12, 13]
                    nchan = len(channels)
                    gmiRttov.Profiles.Angles[:,0] = 49.195  # 166GH~

                # Set up the surface emissivity/reflectance arrays and associate with the Rttov objects
                surfemisrefl_gmi = np.zeros((2, nprofiles, len(channels)), dtype=np.float64)
                gmiRttov.SurfEmis = surfemisrefl_gmi

                # Call RTTOV
                # Surface emissivity/reflectance arrays must be initialised *before every call to RTTOV*
                # Negative values will cause RTTOV to supply emissivity/BRDF values (i.e. equivalent to
                # calcemis/calcrefl TRUE - see RTTOV user guide)
                surfemisrefl_gmi[:,:,:] = -1.

                # Call emissivity and BRDF atlases
                try:
                    # Do not supply a channel list for SEVIRI: this returns emissivity/BRDF values for all
                    # *loaded* channels which is what is required
                    surfemisrefl_gmi[0,:,:] = mwAtlas.getEmisBrdf(gmiRttov, channels)

                except pyrttov.RttovError as e:
                    # If there was an error the emissivities/BRDFs will not have been modified so it
                    # is OK to continue and call RTTOV with calcemis/calcrefl set to TRUE everywhere
                    sys.stderr.write(f'error calling atlas: {e!s}\n')
                # Call the RTTOV direct model for each instrument:
                # no arguments are supplied to runDirect so all loaded channels are
                # simulated
                try:
                    gmiRttov.runDirect(channels)

                except pyrttov.RttovError as e:
                    sys.stderr.write(f'error running RTTOV direct model: {e!s}\n')
                    sys.exit(1)

                # Output array
                if chtype == 0:
                    a2tb_clear_sim[:,:9]  = gmiRttov.BtClear
                    a2tb_sim[:,:9]   = gmiRttov.Bt
                    a2emis_fg[:,:9]  = gmiRttov.SurfEmis[0,:,:]

                elif chtype == 1:
                    a2tb_clear_sim[:,9:]  = gmiRttov.BtClear
                    a2tb_sim[:,9:]   = gmiRttov.Bt
                    a2emis_fg[:,9:]  = gmiRttov.SurfEmis[0,:,:]

                else:
                    sys.stderr.write(f'check chtype {chtype}\n')
                    sys.exit(1)


                # Land-sea mask of TELSEM2
                if chtype == 0:
                    a1landsea = mwAtlas.getEmisBrdf(gmiRttov, channels)[:,0]
                    a1landsea = (a1landsea >= 0).astype('int8')        

            # Emissivity calculation
            a1nu  = gmiRttov.WaveNumbers  # cm^-1
            ts    = np.array(myProfiles.Skin[:,0]).reshape(-1,1)
            a2emis_fg= a2emis_fg
            rad_sfc_ts  = tb2rad(ts, a1nu)
            
            #-- Screen bad TBobs and corresponding simulations
            miss = -9999.

            # Set missing to the invalid profiles
            miss = -9999.
            a2emis_fg[a1mask,:] = miss
            a2tb_sim[a1mask,:] = miss

            # Save output data
            matchdir = os.path.join(output_dir, f'{year:04}', f'{mon:02}', f'{day:02}')
            os.makedirs(matchdir, exist_ok=True)
            
            emis_fg_path = os.path.join(matchdir, f'es_telsem_13ch.{label}.{oid}.float32.npy')
            emis_es_path = os.path.join(matchdir, f'es_rttov_13ch.{label}.{oid}.float32.npy')
            tc_fg_path   = os.path.join(matchdir, f'Tc_fg_13ch.{label}.{oid}.float32.npy')
            tc_obs_path   = os.path.join(matchdir, f'Tc_obs_13ch.{label}.{oid}.float32.npy')
            tau_path     = os.path.join(matchdir, f'tau_tot_13ch.{label}.{oid}.float32.npy')
            rad_atm_up_path = os.path.join(matchdir, f'rad_atm_up_13ch.{label}.{oid}.float32.npy')
            rad_atm_dn_path = os.path.join(matchdir, f'rad_atm_dn_13ch.{label}.{oid}.float32.npy')
            landsea_telsem_path = os.path.join(matchdir, f'landsea_telsem.{label}.{oid}.int8.npy')
            
            np.save(emis_fg_path, a2emis_fg.astype('float32'))
            np.save(tc_fg_path, a2tb_clear_sim.astype('float32')) 
            np.save(tc_obs_path, a2tb_sim.astype('float32'))
            np.save(landsea_telsem_path, a1landsea.astype('int8')) 

            print(emis_fg_path)