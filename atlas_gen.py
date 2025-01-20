import yako_util.util as q
from yako_util import era5 as era5
from datetime import datetime, timedelta
import numpy as np
import os
from tqdm import tqdm
import argparse
from collect_data import get_signature, make_signature, save_single
from collections import defaultdict
import sys
import joblib
from metpy.units import units
from metpy.constants import g

parser = argparse.ArgumentParser(description='Dataset generator')
parser.add_argument('--begin', type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=False, help='Begin date in YYYY/MM/DD format', default=datetime(2014, 4, 2))
parser.add_argument('--end',   type=lambda s: datetime.strptime(s, '%Y/%m/%d'), required=False, help='End date in YYYY/MM/DD format',   default=datetime(2014, 4, 3))
args = parser.parse_args()

DAY_BEGIN = args.begin
DAY_END   = args.end

OUTPUT_DIR = '/mnt/nas04/mykhailo/atlas_v1'
SOURCE_DIR = '/mnt/nas04/mykhailo/atlas_data_pmbase'
MODEL = ''
RESOLUTION = int(1/0.05) # pixels per grid
TTL = timedelta(days=15)

print(f"""
    RUNNING ATLAS GENERATOR
    
    SOURCE_DIR: {SOURCE_DIR}
    OUTPUT_DIR: {OUTPUT_DIR}
    MODEL: {MODEL}
    RESOLUTION: {RESOLUTION}
    TTL: {TTL}
""")


atlas_es = np.full((180*RESOLUTION, 360*RESOLUTION, 13), -1.0, dtype=float)
atlas_updated = np.full((2, 3), datetime(1991, 1, 1), dtype=object)
atlas_owner = np.full((2, 3), None, dtype=object)

OWNED_GMI = 1
OWNED_TELSEM = 2

model = joblib.load(MODEL)

day_iter = np.arange(DAY_BEGIN, DAY_END, timedelta(days=1)).tolist()

for current_date in day_iter:
    year, month, day = current_date.year, current_date.month, current_date.day

    # List files
    files = q.list_files(f'{SOURCE_DIR}/{year:04}/{month:02}/{day:02}/')

    lat_files = sorted(list(filter(lambda x: 'gmi_lat' in x, files)))
    lon_files = sorted(list(filter(lambda x: 'gmi_lon' in x, files)))
    es_files = sorted(list(filter(lambda x: 'es_telsem_13ch' in x, files)))
    landsea_telsem_files = sorted(list(filter(lambda x: 'landsea_telsem' in x, files)))
    cloud_liquid_water_files = sorted(list(filter(lambda x: 'era5_cloud_liqud_water' in x, files)))
    cloud_ice_water_files = sorted(list(filter(lambda x: 'era5_cloud_ice_water' in x, files)))
    precip_liquid_water_files = sorted(list(filter(lambda x: 'era5_precip_liquid_water' in x, files)))
    precip_ice_water_files = sorted(list(filter(lambda x: 'era5_precip_ice_water' in x, files)))
    tc_gmi_files = sorted(list(filter(lambda x: 'gmi_tc' in x, files)))
    autosnow_files = sorted(list(filter(lambda x: 'autosnow' in x, files)))
    surface_pressure_files = sorted(list(filter(lambda x: 'surface_pressure' in x, files)))
    tm_temperature_files = sorted(list(filter(lambda x: 'era5_2m_temperature' in x, files)))
    tm_dewpoint_temperature_files = sorted(list(filter(lambda x: 'era5_2m_dewpoint_temperature' in x, files)))

    # Load data
    tc_gmi = np.concatenate(list(map(np.load, tc_gmi_files)), axis=0)
    pressure = np.tile(era5.get_pressure_levels()[::-1], (tc_gmi.shape[0], 1))
    nprofiles = pressure.shape[0]
    nlevels = pressure.shape[1]
    lat = np.concatenate(list(map(np.load, lat_files)), axis=0)
    lon = np.concatenate(list(map(np.load, lon_files)), axis=0)
    cloud_liquid_water = np.concatenate(list(map(np.load, cloud_liquid_water_files)), axis=0).reshape(nprofiles, nlevels)
    cloud_ice_water = np.concatenate(list(map(np.load, cloud_ice_water_files)), axis=0).reshape(nprofiles, nlevels)
    precip_liquid_water = np.concatenate(list(map(np.load, precip_liquid_water_files)), axis=0).reshape(nprofiles, nlevels)
    precip_ice_water = np.concatenate(list(map(np.load, precip_ice_water_files)), axis=0).reshape(nprofiles, nlevels)
    es = np.concatenate(list(map(np.load, es_files)), axis=0).reshape(nprofiles, 13)
    landsea_telsem = np.concatenate(list(map(np.load, landsea_telsem_files)), axis=0).reshape(nprofiles)
    autosnow = np.concatenate(list(map(np.load, autosnow_files)), axis=0).reshape(nprofiles)
    surface_pressure = np.concatenate(list(map(np.load, surface_pressure_files)), axis=0).reshape(nprofiles)
    tm_temperature = np.concatenate(list(map(np.load, tm_temperature_files)), axis=0).reshape(nprofiles)
    tm_dewpoint_temperature = np.concatenate(list(map(np.load, tm_dewpoint_temperature_files)), axis=0).reshape(nprofiles)

    # Masks
    qr_avg = (precip_liquid_water[:,:-1] + precip_liquid_water[:,1:]) / 2
    qi_avg = (precip_ice_water[:,:-1] + precip_ice_water[:,1:]) / 2
    qc_avg = (cloud_liquid_water[:,:-1] + cloud_liquid_water[:,1:]) / 2
    qci_avg = (cloud_ice_water[:,:-1] + cloud_ice_water[:,1:]) / 2

    dp = -np.diff((pressure*units.hPa).to(units.Pa), axis=1)
    tcwv = np.sum(((qr_avg + qi_avg + qc_avg + qci_avg) * dp) / g, axis=1).to('kg/m2').m
    clear_sky_mask = tcwv <= 0.15

    gmi_valid_measurement = (
        np.all(tc_gmi >= 50, axis=1) &
        np.all(tc_gmi <= 500, axis=1)
    )

    telsem_surf = landsea_telsem == 1

    sea_mask = autosnow == 0
    land_mask = autosnow == 1
    snow_mask = autosnow == 2
    ice_mask = autosnow == 3
    frozen_sea_mask = autosnow == 4
    beach_mask = (autosnow == 5) | (autosnow == 6)
    glacier_mask = autosnow == 7

    lat_mask = (lat >= 40) | (lat <= -40)

    surface_mask = (land_mask|snow_mask|ice_mask) & telsem_surf & lat_mask
    use_mask = surface_mask & clear_sky_mask & gmi_valid_measurement

    # Process
    lat_lon = list(zip(
        list(map(q.lat_stoz, lat)),
        list(map(q.lon_stoz, lon)),
    ))

    features = np.array([
        *(tc_gmi.T),
        surface_pressure,
        tm_dewpoint_temperature,
        tm_temperature,
    ]).T
    features = features[use_mask]

    # Fill gmi
    atlas_updated[lat_lon[use_mask]] = current_date
    es_estimated = model.predict(features)
    atlas_es[lat_lon[use_mask]] = es_estimated
    atlas_owner[lat_lon[use_mask]] = OWNED_GMI

    # Fill old/unavailiable
    outdated_mask = (current_date - atlas_updated[lat_lon]) > TTL
    atlas_updated[lat_lon[outdated_mask]] = current_date
    atlas_es[lat_lon[outdated_mask]] = es
    atlas_owner[lat_lon[outdated_mask]] = OWNED_TELSEM

    base_dir = f'{SOURCE_DIR}/{year:04}/{month:02}/{day:02}/'
    os.makedirs(base_dir, exist_ok=True)
    np.save(f'{base_dir}/atlas_es.npy', atlas_es)
    np.save(f'{base_dir}/atlas_updated.npy', atlas_updated)
    np.save(f'{base_dir}/atlas_owner.npy', atlas_owner)
