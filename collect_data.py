import yako_util.util as q
import yako_util.autosnow as y_autosnow
import yako_util.era5 as y_era5
from functools import cache, wraps
import re
from datetime import datetime
import numpy as np
import netCDF4
import os
import joblib
from tqdm import tqdm
import sys
import argparse

# What files are the backbone for the dataset (GPM/Cloudsat)
MASTER_FILES_DIR = '/media/nisioji/nas03/data/PMM/NASA/CSATGPM.COIN/2B/V05'

# Location of all processed files
# Output directory structure is not fixed, so matching by signature is preferred
OUTPUT_DIR = '/mnt/nas04/mykhailo/atlas_data'

# Cache function outputs in _cache
USE_CACHE = False

parser = argparse.ArgumentParser(description='Dataset generator')
parser.add_argument('--begin', type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=False, help='Begin date in YYYY/MM/DD format', default=datetime(2014, 4, 2))
parser.add_argument('--end',   type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=False, help='End date in YYYY/MM/DD format',   default=datetime(2014, 4, 3))
args = parser.parse_args()

# Limit number of items collected
DAY_BEGIN = args.begin
DAY_END   = args.end

# All files should be structured as var_name.{year}{month}{day}-{orbit}.{number}
# Get signature obtains file id as (year, month, day, orbit, number)
@cache
def get_signature(filepath: str) -> tuple:
    pattern = r'(\d{4})(\d{2})(\d{2})-([a-zA-Z0-9]{7}-[a-zA-Z0-9]{7})\.(\d{6})'

    res = re.findall(pattern, filepath)
    if len(res) != 0:
        return res[0]

# Returns string from signature data
def make_signature(signature: tuple) -> str:
    return f'{signature[0]}{signature[1]}{signature[2]}-{signature[3]}.{signature[4]}'

# Util function (TODO: move to util)
# If CACHE is set, cache function results in _cache
def file_cache(func):
    cache_dir = "_cache"
    os.makedirs(cache_dir, exist_ok=True)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not USE_CACHE:
            return func(*args, **kwargs)
        
        cache_file = os.path.join(cache_dir, f"{func.__name__}.z")
        if os.path.exists(cache_file):
            return joblib.load(cache_file)
        result = func(*args, **kwargs)
        joblib.dump(result, cache_file)
        return result

    return wrapper

# Util function (TODO: move to util)
# Returns a handler to extract data from nc4
def extract_from_nc4(group, variable):
    def extr(x):
        with netCDF4.Dataset(x, 'r') as nc:
            data  = nc[group].variables[variable][:]
        return data.data
    return extr

# When having a 2d list [[a1, a2, a3], [b1, b2, b3]]:
# Apply fn_before for axis=0, then apply fn_each for axis=1 and after fn_after for axis=0
def deep_map(lst, fn_before, fn_each, fn_after):
    fn_before = fn_before or (lambda x: x)
    fn_after  = fn_after or (lambda x: x)

    before = list(map(fn_before, lst))
    after = list(
        map(
            lambda arr: fn_after(list(map(fn_each, arr))) if fn_each else fn_after(arr),
            before
        )
    )
    return after

# Returns a list containing (signature, locations(lat,lon)[], datetime[]) for each file
# Time and location is defined for CloudSat pixels, if BASE_SAT='CloudSat'
@file_cache
def dataset_time_locs() -> list[tuple[tuple, np.ndarray, np.ndarray]]:
    def time_filter(file):
        sig = get_signature(file)
        file_date = datetime(int(sig[0]), int(sig[1]), int(sig[2]))
        if DAY_BEGIN <= file_date and file_date < DAY_END:
            return True
        return False
    
    def full_dataset_filter(file):
        groups = ['1C-R.GPM.GMI', '2A.GPM.DPR', '2A.GPM.GMI.GPROF',
                   '2B-CLDCLASS', '2B-CLDCLASS-LIDAR', '2B-CWC-RO',
                   '2B-GEOPROF', '2B-GEOPROF-LIDAR', '2B.GPM.DPRGMI',
                   '2C-ICE', '2C-PRECIP-COLUMN', '2C-RAIN-PROFILE',
                   '2C-SNOW-PROFILE', 'ECMWF-AUX']
        with netCDF4.Dataset(file, 'r') as nc:
            return set(groups).issubset(set(nc.groups.keys()))
        return False

    # Obtain all files
    gmp_files = list(filter(lambda x: '.nc4' in x, q.list_files(MASTER_FILES_DIR)))
    # Filter by dates & complteness
    gmp_files = sorted(list(filter(time_filter, gmp_files)))
    gmp_files = sorted(list(filter(full_dataset_filter, gmp_files)))
    
    signatures = list(map(get_signature, gmp_files))

    gpm_locations_lat = deep_map(
        list(map(extract_from_nc4('ECMWF-AUX', 'Latitude'), gmp_files)),
        lambda x: x.data,
        None,
        np.array
    )

    gpm_locations_lon = deep_map(
        list(map(extract_from_nc4('ECMWF-AUX', 'Longitude'), gmp_files)),
        lambda x: x.data,
        None,
        np.array
    )

    gpm_times = deep_map(
        list(map(extract_from_nc4('ECMWF-AUX', 'Beam_time'), gmp_files)),
        lambda x: x.data,
        datetime.fromtimestamp,
        np.array,
    )

    gpm_lat_lon = list(map(lambda x: np.array(x).T, zip(gpm_locations_lat, gpm_locations_lon)))

    # Size is ok
    assert(len(gpm_lat_lon) == len(gpm_times))
    assert(len(gpm_lat_lon) == len(signatures))
    assert(list(map(lambda x: x.shape[0], gpm_lat_lon)) == list(map(lambda x: x.shape[0], gpm_times)))
    
    return list(zip(signatures, gpm_times, gpm_lat_lon))

# Save single variable array to file
def save_single(variable, data, signature):
    filename = f'{variable}-{make_signature(signature)}.npy'
    dir = os.path.join(OUTPUT_DIR, signature[0], signature[1], signature[2])
    os.makedirs(dir, exist_ok=True)

    filepath = os.path.join(dir, filename)
    np.save(filepath, data)

# Save variable over track
def save(variable, data, track_data):
    list(map(lambda i: save_single(variable, data[i], track_data[i][0]), range(len(track_data))))

def from_gmi(track_data, group, variable):
    def gmi_by_signature(sig):
        return os.path.join(MASTER_FILES_DIR, sig[0], sig[1], f'2B.CSATGPM.COIN.COIN2022.{sig[0]}{sig[1]}{sig[2]}-{sig[3]}.{sig[4]}.V05.nc4')
    
    extractor = extract_from_nc4(group, variable)
    files = list(map(lambda x: gmi_by_signature(x[0]), track_data))
    data = list(map(extractor, files))
    data = deep_map(
        data,
        lambda x: x.data,
        None,
        np.array,
    )

    return data

if __name__ == '__main__':
    print(f'Processing between {DAY_BEGIN.date()} and {DAY_END.date()}')
    if USE_CACHE:
        print('Cache is used, results might not change from the previous run!')

    track_data = dataset_time_locs()
    if len(track_data) == 0:
        sys.exit(0)

    times = list(zip(*track_data))[1]
    locs = list(zip(*track_data))[2]

    plevels = y_era5.get_pressure_levels()

    # Variable: (extractor, (args)). First arg is track data by default
    process_data = {
        # 'gmi_tc': (from_gmi, ('1C-R.GPM.GMI/S1', 'Tc')),
        # 'gmi_lat': (from_gmi, ('ECMWF-AUX', 'Latitude')),
        # 'gmi_lon': (from_gmi, ('ECMWF-AUX', 'Longitude')),
        # 'gmi_scan_id': (from_gmi, ('1C-R.GPM.GMI', 'scan_indices')),
        # 'cloudsat_snowfall_rate_sfc': (from_gmi, ('2C-SNOW-PROFILE', 'snowfall_rate_sfc')),
        # 'autosnow': (lambda tdata: list(map(lambda i: y_autosnow.at(times[i].tolist(), locs[i].tolist()),range(len(track_data)))), ())
        # 'era5_2m_temperature': (lambda tdata: list(map(lambda i: y_era5.at_single(times[i].tolist(), locs[i].tolist(), ['2m_temperature']),range(len(track_data)))), ()),

        # 'era5_2m_dewpoint_temperature': (lambda tdata: list(map(lambda i: y_era5.at_single(times[i].tolist(), locs[i].tolist(), ['2m_dewpoint_temperature']),range(len(track_data)))), ()),
        # 'era5_surface_pressure': (lambda tdata: list(map(lambda i: y_era5.at_single(times[i].tolist(), locs[i].tolist(), ['surface_pressure']),range(len(track_data)))), ()),
        # 'era5_skin_temperature': (lambda tdata: list(map(lambda i: y_era5.at_single(times[i].tolist(), locs[i].tolist(), ['skin_temperature']),range(len(track_data)))), ()),
        
        # 'era5_u10': (lambda tdata: list(map(lambda i: y_era5.at_single(times[i].tolist(), locs[i].tolist(), ['10m_u_component_of_wind']),range(len(track_data)))), ()),
        # 'era5_v10': (lambda tdata: list(map(lambda i: y_era5.at_single(times[i].tolist(), locs[i].tolist(), ['10m_v_component_of_wind']),range(len(track_data)))), ()),
        # 'era5_temperature': (lambda tdata: list(map(lambda i: y_era5.at_p(times[i].tolist(), locs[i].tolist(), ['specific_cloud_liquid_water_content'], plevels),range(len(track_data)))), ()),
        'era5_cloud_liqud_water': (lambda tdata: list(map(lambda i: y_era5.at_p(times[i].tolist(), locs[i].tolist(), ['specific_cloud_liquid_water_content'], plevels),range(len(track_data)))), ()),
        'era5_cloud_ice_water': (lambda tdata: list(map(lambda i: y_era5.at_p(times[i].tolist(), locs[i].tolist(), ['specific_cloud_ice_water_content'], plevels),range(len(track_data)))), ()),
        'era5_precip_liquid_water': (lambda tdata: list(map(lambda i: y_era5.at_p(times[i].tolist(), locs[i].tolist(), ['specific_rain_water_content'], plevels),range(len(track_data)))), ()),
        'era5_precip_ice_water': (lambda tdata: list(map(lambda i: y_era5.at_p(times[i].tolist(), locs[i].tolist(), ['specific_snow_water_content'], plevels),range(len(track_data)))), ()),
        # 'era5_specific_humidity': (lambda tdata: list(map(lambda i: y_era5.at_p(times[i].tolist(), locs[i].tolist(), ['specific_humidity'], plevels),range(len(track_data)))), ()),
    }

    for variable, (fn, args) in tqdm(process_data.items()):
        while True:
            try:
                data = fn(track_data, *args)
                save(variable, data, track_data)
                break
            except Exception:
                continue
