import yako_util.util as q
from datetime import datetime
import numpy as np
import os
from tqdm import tqdm
import argparse
from collect_data import get_signature, make_signature, save_single
from collections import defaultdict
import sys

# MAPPED_FILES_DIR_RTTOV = '/mnt/nas04/mykhailo/rttov_atlas_delta'
MAPPED_FILES_DIR_RTTOV = '/mnt/nas04/mykhailo/rttov_atlas'
MAPPED_FILES_DIR_ATLAS = '/mnt/nas04/mykhailo/atlas_data'

MASTER_FILES_DIR = '/mnt/nas04/mykhailo/atlas_data'
# OUTPUT_DIR = '/mnt/nas04/mykhailo/atlas_data_pmbase_delta'
OUTPUT_DIR = '/mnt/nas04/mykhailo/atlas_data_pmbase'


parser = argparse.ArgumentParser(description='Dataset generator')
parser.add_argument('--begin', type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=False, help='Begin date in YYYY/MM/DD format', default=datetime(2014, 4, 2))
parser.add_argument('--end',   type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=False, help='End date in YYYY/MM/DD format',   default=datetime(2014, 4, 3))
args = parser.parse_args()

DAY_BEGIN = args.begin
DAY_END   = args.end

def get_base_signatures():
    def time_filter(file):
        sig = get_signature(file)
        file_date = datetime(int(sig[0]), int(sig[1]), int(sig[2]))
        if DAY_BEGIN <= file_date and file_date < DAY_END:
            return True
        return False

    scan_id_files = list(filter(lambda x: 'gmi_scan_id' in x, q.list_files(MASTER_FILES_DIR)))
    scan_id_files = sorted(list(filter(time_filter, scan_id_files)))
    
    signatures = list(map(get_signature, scan_id_files))

    return signatures

def get_var_files_by_signatures(varname, source_dir, signatures):
    def sig_filter(file):
        sig = get_signature(file)
        return sig in signatures
    
    var_files = list(filter(lambda x: varname in x, q.list_files(source_dir)))
    var_files = sorted(list(filter(sig_filter, var_files)))
    var_files_signs = list(map(get_signature, var_files))

    return dict(zip(var_files_signs, var_files))

def extract_var_data(file):
    return np.load(file)

def make_coin_ids(signature, id_file) -> tuple[dict, list]:
    matches = defaultdict(list)
    id = extract_var_data(id_file)

    obs_order = []
    for pos, (x, y) in enumerate(id):
        if (x,y) not in matches:
            obs_order.append((x,y))
        matches[(x,y)].append(pos)
    # keys = sorted(list(matches.keys()))
    keys = obs_order

    return matches, keys

def process_var(varname, dir, matchup, fn_group):
    signatures = set(matchup.keys())
    var_files = get_var_files_by_signatures(varname, dir, signatures)
    var_data = {k:extract_var_data(v) for k, v in var_files.items()}

    result = {}
    for sig, (matches, keys) in matchup.items():
        if sig not in var_data:
            print(f'Files for {varname} does not contain signature={sig}')
            continue
        
        data = var_data[sig]
        # Make sure that the order in iteration is preserved
        # res = []
        # for key in keys:
        #     trt = list(filter(lambda x: x < len(data), matches[key]))
        #     if len(trt) == 0:
        #         print(len(data), trt, matches[key])
        #     res.append(
        #         fn_group(
        #             data[
        #                 list(filter(lambda x: x < len(data), matches[key]))
        #             ]
        #         )
        #     )
        # print(len(data), matches[-1])
        res = list(map(lambda x: fn_group(data[matches[x]]), keys))

        result[sig] = np.array(res)
    
    return result

# TODO: Duplicate function
def save_single(variable, data, signature):
    filename = f'{variable}-{make_signature(signature)}.npy'
    dir = os.path.join(OUTPUT_DIR, signature[0], signature[1], signature[2])
    os.makedirs(dir, exist_ok=True)

    filepath = os.path.join(dir, filename)
    np.save(filepath, data)

def process_save(varname, dir, coin, fn):
    sig_data = process_var(varname, dir, coin, fn)
    [save_single(varname, data, sig) for sig, data in sig_data.items()]

# 0 - water only
# 1 - land only
# 2 - snow land only
# 3 - ice only
# 4 - frozen sea (ice and water)
# 5 - beach (land and water)
# 6 - frozen beach (snow/land and water/ice)
# 7 - glacier (snow/land and ice)
def surface_reclassifier(data):
    # Autosnow: {0:'water', 1:'snow-free land', 2:'snow-covered land', 3:'ice'}
    has_water = 0 in data
    has_land = 1 in data
    has_snow = 2 in data
    has_ice = 3 in data

    if has_ice and (has_snow or has_land) and not has_water:
        return 7
    if (has_ice and (has_snow or has_land) and has_water) or (has_snow and has_water):
        return 6
    if has_water and has_land:
        return 5
    if has_water and has_ice and not (has_land or has_snow):
        return 4
    if has_ice:
        return 3
    if has_snow:
        return 2
    if has_land:
        return 1

    return 0


if __name__ == '__main__':
    print(f'Processing between {DAY_BEGIN.date()} and {DAY_END.date()}')

    signatures = get_base_signatures()
    id_sig_fiels = get_var_files_by_signatures('gmi_scan_id', MASTER_FILES_DIR, signatures)
    coin = {sig: make_coin_ids(sig, file) for sig, file in id_sig_fiels.items()}

    # TODO: Shouldn't it be with/without save (consistency)
    # Variable: (extractor, (args)). First arg is variable name, second - coin by default
    process_data = {
        'autosnow': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: surface_reclassifier(x), )),
        'cloudsat_snowfall_rate_sfc': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'ecmwf_2m_temperature': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_2m_dewpoint_temperature': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_2m_temperature': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_cloud_ice_water': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_cloud_liqud_water': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_fraction_of_cloud_cover': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_precip_ice_water': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_precip_liquid_water': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_skin_temperature': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_specific_humidity': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_surface_pressure': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_temperature': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_u10': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'era5_v10': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'geoprof_dem_elevation': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'gmi_lat': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'gmi_lon': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'gmi_scan_id': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'gmi_lon': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),
        'gmi_tc': (MAPPED_FILES_DIR_ATLAS, process_save, (lambda x: np.mean(x, axis=0), )),

        'Tc_clear_13ch': (MAPPED_FILES_DIR_RTTOV, process_save, (lambda x: np.mean(x, axis=0), )),
        'Tc_hydro_13ch': (MAPPED_FILES_DIR_RTTOV, process_save, (lambda x: np.mean(x, axis=0), )),
        'es_telsem_13ch': (MAPPED_FILES_DIR_RTTOV, process_save, (lambda x: np.mean(x, axis=0), )),
        'landsea_telsem': (MAPPED_FILES_DIR_RTTOV, process_save, (lambda x: np.mean(x, axis=0), )),
    }

    for variable, (dir, fn, args) in tqdm(process_data.items()):
        fn(variable, dir, coin, *args)