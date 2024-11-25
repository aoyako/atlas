import yako_util.util as q
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
from collect_data import get_signature, make_signature, save_single
from collections import defaultdict

MASTER_FILES_DIR = '/mnt/nas04/mykhailo/atlas_data'
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

def get_var_files_by_signatures(varname, signatures):
    def sig_filter(file):
        sig = get_signature(file)
        return sig in signatures
    
    var_files = list(filter(lambda x: varname in x, q.list_files(MASTER_FILES_DIR)))
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

def process_var(varname, matchup, fn_group):
    signatures = set(matchup.keys())
    var_files = get_var_files_by_signatures(varname, signatures)
    var_data = {k:extract_var_data(v) for k, v in var_files.items()}

    result = {}
    for sig, (matches, keys) in matchup.items():
        if sig not in var_data:
            print(f'Files for {varname} does not contain signature={sig}')
            continue
        
        data = var_data[sig]
        # Make sure that the order in iteration is preserved
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

def process_save(varname, coin, fn):
    sig_data = process_var(varname, coin, fn)
    [save_single(varname, data, sig) for sig, data in sig_data.items()]

if __name__ == '__main__':
    print(f'Processing between {DAY_BEGIN.date()} and {DAY_END.date()}')

    signatures = get_base_signatures()
    id_sig_fiels = get_var_files_by_signatures('gmi_scan_id', signatures)
    coin = {sig: make_coin_ids(sig, file) for sig, file in id_sig_fiels.items()}

    # TODO:  Shouldn't it be with/without save (consistency)
    # Variable: (extractor, (args)). First arg is variable name, second - coin by default
    process_data = {
        'gmi_tc': (process_save, (lambda x: np.mean(x, axis=0), )),
        'gmi_lat': (process_save, (lambda x: np.mean(x, axis=0), )),
        'gmi_lon': (process_save, (lambda x: np.mean(x, axis=0), )),
    }

    for variable, (fn, args) in tqdm(process_data.items()):
        fn(variable, coin, *args)