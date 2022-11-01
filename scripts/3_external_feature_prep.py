import pandas as pd
from matplotlib import pyplot as plt
import os
import matplotlib
matplotlib.use('agg')

# Day-ahead available and actual (ex-post) features from ENTSO-E data
actual_cols = ['load', 'gen_biomass', 'gen_lignite', 'gen_coal_gas', 'gen_gas',
               'gen_hard_coal', 'gen_oil', 'gen_oil_shale', 'gen_fossil_peat',
               'gen_geothermal', 'gen_pumped_hydro', 'gen_run_off_hydro',
               'gen_reservoir_hydro', 'gen_marine', 'gen_nuclear', 'gen_other_renew',
               'gen_solar', 'gen_waste', 'gen_wind_off', 'gen_wind_on', 'gen_other']

forecast_cols = ['load_day_ahead', 'scheduled_gen_total', 'prices_day_ahead',
                 'solar_day_ahead', 'wind_off_day_ahead', 'wind_on_day_ahead']


# Areas inlcuding "country" areas
areas = ['CE']

# Time zones of frequency recordings
tzs = {'CE': 'CET'}


for area in areas:
    print('Processing external features from', area)

    # Setup folder paths to raw input data
    folder = './data/{}/'.format(area)
    doc_folder = folder + 'documentation_of_data_download/'
    if not os.path.exists(doc_folder):
        os.makedirs(doc_folder)

    # Load the pre-processed external features
    data = pd.read_hdf(folder + 'raw_input_data.h5')

    # Inspection of data distribution as histograms
    fig, ax = plt.subplots(figsize=(len(data.columns), len(data.columns)))
    data.hist(log=True, ax=ax, bins=100)
    plt.tight_layout()
    plt.savefig(doc_folder + 'raw_data_histograms.svg', bbox_inches='tight')
    plt.close()

    # Split into forecast and actual data and convert index to local timezone of frequency data
    area_actual_cols = actual_cols + [i for i in data.columns if (i.endswith('_cross_border_flow')) | (i.endswith('_import_export_total'))]
    area_forecast_cols = forecast_cols  # + [i for i in data.columns if (i.endswith('_import_export_day_ahead'))]
    input_forecast = data.loc[:, data.columns.intersection(area_forecast_cols)]
    input_actual = data.loc[:, data.columns.intersection(area_actual_cols)]
    input_actual.index = input_actual.index.tz_convert(tzs[area])
    input_forecast.index = input_forecast.index.tz_convert(tzs[area])

    #  Additional engineered features #
    # Time
    input_forecast['month'] = input_forecast.index.month
    input_forecast['weekday'] = input_forecast.index.weekday
    input_forecast['hour'] = input_forecast.index.hour

    # Total generation
    input_actual['total_gen'] = input_actual.filter(regex='^gen').sum(axis='columns')


    # Ramps of load and total generation
    input_forecast['load_ramp_day_ahead'] = input_forecast.load_day_ahead.diff()
    input_actual['load_ramp'] = input_actual.load.diff()
    input_forecast['total_gen_ramp_day_ahead'] = input_forecast.scheduled_gen_total.diff()
    input_actual['total_gen_ramp'] = input_actual.total_gen.diff()

    # Ramps of generaton types
    if 'wind_off_day_ahead' in input_forecast.columns:
        input_forecast['wind_off_ramp_day_ahead'] = input_forecast.wind_off_day_ahead.diff()
    input_forecast['wind_on_ramp_day_ahead'] = input_forecast.wind_on_day_ahead.diff()
    if 'solar_day_ahead' in input_forecast.columns:
        input_forecast['solar_ramp_day_ahead'] = input_forecast.solar_day_ahead.diff()
    gen_ramp_cols = input_actual.filter(regex='^gen').columns.str[4:] + '_ramp'
    input_actual[gen_ramp_cols] = input_actual.filter(regex='^gen').diff()

    # Price Ramps
    input_forecast['price_ramp_day_ahead'] = input_forecast.prices_day_ahead.diff()

    # Flow ramps
    import_export_total_prefix = [x.replace('_import_export_total', '') for x in data.filter(regex='import_export_total$').columns.values]
    import_export_day_ahead_prefix = [x.replace('_import_export_day_ahead', '') for x in data.filter(regex='import_export_day_ahead$').columns.values]
    cross_border_flow_prefix = [x.replace('_cross_border_flow', '') for x in data.filter(regex='cross_border_flow$').columns.values]
    for i in import_export_total_prefix:
        input_actual[i+'_import_export_total_ramp'] = input_actual[i+'_import_export_total'].diff()
    for i in cross_border_flow_prefix:
        input_actual[i+'_cross_border_flow_ramp'] = input_actual[i+'_cross_border_flow'].diff()

    # Forecast errors
    input_actual['forecast_error_total_gen'] = input_forecast.scheduled_gen_total - input_actual.total_gen
    input_actual['forecast_error_load'] = input_forecast.load_day_ahead - input_actual.load
    input_actual['forecast_error_load_ramp'] = input_forecast.load_ramp_day_ahead - input_actual.load_ramp
    input_actual['forecast_error_total_gen_ramp'] = input_forecast.total_gen_ramp_day_ahead - input_actual.total_gen_ramp
    input_actual['forecast_error_wind_on_ramp'] = input_forecast.wind_on_ramp_day_ahead - input_actual.wind_on_ramp
    input_actual['forecast_error_wind_on'] = input_forecast.wind_on_day_ahead - input_actual.gen_wind_on
    if 'wind_off_day_ahead' in input_forecast.columns:
        input_actual['forecast_error_wind_off'] = input_forecast.wind_off_day_ahead - input_actual.gen_wind_off
        input_actual['forecast_error_wind_off_ramp'] = input_forecast.wind_off_ramp_day_ahead - input_actual.wind_off_ramp
    if 'solar_day_ahead' in input_forecast.columns:
        input_actual['forecast_error_solar_ramp'] = input_forecast.solar_ramp_day_ahead - input_actual.solar_ramp
        input_actual['forecast_error_solar'] = input_forecast.solar_day_ahead - input_actual.gen_solar


    for i in set(import_export_total_prefix).intersection(cross_border_flow_prefix):
        input_actual[i+'_unscheduled_flow'] = input_actual[i+'_import_export_total'] - input_actual[i+'_cross_border_flow']
        input_actual[i+'_unscheduled_flow_ramp'] = input_actual[i+'_import_export_total_ramp'] - input_actual[i+'_cross_border_flow_ramp']

    # Save data
    input_actual.to_hdf(folder + 'input_actual.h5', key='df')
    input_forecast.to_hdf(folder + 'input_forecast.h5', key='df')
