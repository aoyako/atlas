import numpy as np
from numpy import ma
import netCDF4
import sys
import os
import glob
from datetime import datetime, timedelta
import argparse
from metpy.calc import mixing_ratio_from_relative_humidity, density, relative_humidity_from_specific_humidity, specific_humidity_from_dewpoint
from metpy.units import units, masked_array
from yako_util.era5 import get_pressure_levels

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

ADD_DELTA = True
# OUTPUT_DIR = '/mnt/nas04/mykhailo/rttov_atlas'
OUTPUT_DIR = '/mnt/nas04/mykhailo/rttov_atlas_delta' # es + delta
DATA_DIR = '/mnt/nas04/mykhailo/atlas_data'

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
    ldtime = np.arange(DAY_BEGIN, DAY_END, timedelta(days=1)).tolist()

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

    for dtime in ldtime:
        print('processing', dtime)
        year, month, day = dtime.year, dtime.month, dtime.day

        ssearch = f'/media/nisioji/nas03/data/PMM/NASA/CSATGPM.COIN/2B/V05/{year:04}/{month:02}/2B.CSATGPM.COIN.COIN2022.{year:04}{month:02}{day:02}*.nc4'
        lpath = sorted(glob.glob(ssearch))

        for ncpath in lpath[:]:
            print(ncpath)
            label = ncpath.split('.')[-4]
            oid = ncpath.split('.')[-3]

            print(label, oid)

            # For era5, profile data is from surface to TOA!
            try:
                org1lat = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'gmi_lat-{label}.{oid}.npy'))
                org1lon = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'gmi_lon-{label}.{oid}.npy'))
                org1ts = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_skin_temperature-{label}.{oid}.npy'))
                org1dt2m = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_2m_dewpoint_temperature-{label}.{oid}.npy'))
                org1ps = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_surface_pressure-{label}.{oid}.npy')) * 0.01
                org1t2m =  np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_2m_temperature-{label}.{oid}.npy'))
                org1u10 = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_u10-{label}.{oid}.npy'))
                org1v10 = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_v10-{label}.{oid}.npy'))
                org2ta = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_temperature-{label}.{oid}.npy'))
                org2sh = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_specific_humidity-{label}.{oid}.npy'))
                org2clw = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_cloud_liqud_water-{label}.{oid}.npy'))
                org2ciw = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_cloud_ice_water-{label}.{oid}.npy'))
                org2rain = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_precip_liquid_water-{label}.{oid}.npy'))
                org2snow = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_precip_ice_water-{label}.{oid}.npy'))
                org2cloud = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'era5_fraction_of_cloud_cover-{label}.{oid}.npy'))
                org1surf = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'autosnow-{label}.{oid}.npy'))
                org1elev = np.load(os.path.join(DATA_DIR, f'{year:04}', f'{month:02}', f'{day:02}', f'geoprof_dem_elevation-{label}.{oid}.npy')) * 0.001
                org2pa = np.tile(get_pressure_levels(), (len(org2sh), 1))
            except Exception as e:
                print(e)
                continue
            

            # Compute specific humidity
            org1sh = specific_humidity_from_dewpoint(org1ps*units.hPa, org1dt2m*units.K)

            nprofiles = org2pa.shape[0]
            nlevels = org2pa.shape[1]
            myProfiles = pyrttov.ProfilesScatt(nprofiles, nlevels)

            myProfiles.GasUnits = 1 # 1: kg/kg over moist air (default)
            myProfiles.P = org2pa.reshape(nprofiles, nlevels)
            myProfiles.T = org2ta.reshape(nprofiles, nlevels)[:,::-1]
            myProfiles.Q = np.clip(org2sh.reshape(nprofiles, nlevels)[:,::-1], a_min=1e-11, a_max=None)

            as2m = np.array([org1ps.reshape(-1), org1t2m.reshape(-1), org1sh.reshape(-1), org1u10.reshape(-1), org1v10.reshape(-1)]).T
            myProfiles.S2m = as2m

            a2skin = np.array([org1ts.reshape(-1), [3.5e+1]*nprofiles, [0]*nprofiles, [3.0e+0]*nprofiles, [5.0e+0]*nprofiles, [1.5e+1]*nprofiles, [1.0e-1]*nprofiles, [3.0e-1]*nprofiles]).T
            myProfiles.Skin =  a2skin

            myProfiles.Angles = [[52.821, 0] for i in range(nprofiles)]  # zenangle, azangle  (<=89GH)

            # Autosnow: {0:'water', 1:'snow-free land', 2:'snow-covered land', 3:'ice'}
            # RTTOV: {0:'land', 1:'sea', 2:'seaice'}
            org1surf = org1surf.reshape(-1)
            org1surf[org1surf==0] = -1
            org1surf[org1surf==1] = 0
            org1surf[org1surf==2] = 0
            org1surf[org1surf==3] = -2
            org1surf *= -1
            org1surf[org1surf > 2] = 0 # Unknown surftype
            myProfiles.SurfType = org1surf

            a2surfgeom = np.array([org1lat.reshape(-1), org1lon.reshape(-1), org1elev.reshape(-1)]).T
            myProfiles.SurfGeom = a2surfgeom

            a2datetimes = np.tile([year, month, day, 0, 0, 0], (nprofiles, 1))
            myProfiles.DateTimes = a2datetimes

            a2ph = make_half_levels(org2pa.reshape(nprofiles, nlevels), org1ps.reshape(-1))
            myProfiles.Ph = a2ph

            myProfiles.HydroFrac = org2cloud.reshape(nprofiles, nlevels)[:,::-1]
            myProfiles.Clw = org2clw.reshape(nprofiles, nlevels)[:,::-1]
            myProfiles.Ciw = org2ciw.reshape(nprofiles, nlevels)[:,::-1]
            myProfiles.Snow = org2snow.reshape(nprofiles, nlevels)[:,::-1]
            myProfiles.Rain = org2rain.reshape(nprofiles, nlevels)[:,::-1]

            gmiRttov.Profiles = myProfiles
            mwAtlas = pyrttov.Atlas()
            mwAtlas.AtlasPath = os.path.join(rttov_installdir, 'emis_data', 'telsem2')
            mwAtlas.loadMwEmisAtlas(a2datetimes[0][1])

            a2tb_clear_sim = np.empty([nprofiles, 13], dtype='float64')
            a2tb_sim = np.empty([nprofiles, 13], dtype='float64')
            a2emis_fg = np.empty([nprofiles, 13], dtype='float64')

            if ADD_DELTA:
                shared_delta_0_6 = np.random.uniform(-0.3, 0.3, size=(1))
                shared_delta_7_12 = np.random.uniform(-0.3, 0.3, size=(1))
                delta = np.hstack([
                    np.tile(shared_delta_0_6, (nprofiles, 7)),
                    np.tile(shared_delta_7_12, (nprofiles, 6)),
                ])

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

                    # Add delta to emissvity
                    if ADD_DELTA:
                        valid_mask = (surfemisrefl_gmi[0,:,:] >= 0)
                        if chtype == 0:
                            dd = delta[:,:9]
                        elif chtype == 1:
                            dd = delta[:,9:]
                        surfemisrefl_gmi[0,valid_mask] += dd[valid_mask]
                        surfemisrefl_gmi[0,valid_mask] = np.clip(surfemisrefl_gmi[0,valid_mask], 0, 1)

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
                    # sys.exit(1)
                    continue

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

                if chtype == 0:
                    a1landsea = mwAtlas.getEmisBrdf(gmiRttov, channels)[:,0]
                    a1landsea[a1landsea >= 0] = 1
                    a1landsea[a1landsea < 0] = 0


            matchdir = os.path.join(OUTPUT_DIR, f'{year:04}', f'{month:02}', f'{day:02}')
            os.makedirs(matchdir, exist_ok=True)
            
            emis_fg_path = os.path.join(matchdir, f'es_telsem_13ch-{label}.{oid}.npy')
            tc_fg_path   = os.path.join(matchdir, f'Tc_clear_13ch-{label}.{oid}.npy')
            tc_obs_path   = os.path.join(matchdir, f'Tc_hydro_13ch-{label}.{oid}.npy')
            landsea_telsem_path = os.path.join(matchdir, f'landsea_telsem-{label}.{oid}.npy')

            np.save(emis_fg_path, a2emis_fg)
            np.save(tc_fg_path, a2tb_clear_sim) 
            np.save(tc_obs_path, a2tb_sim)
            np.save(landsea_telsem_path, a1landsea)

            print('finished', label, oid)