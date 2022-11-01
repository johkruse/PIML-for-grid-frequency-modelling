cd /path/to/local/ENTSO-E/data

sftp e.mail@institute.org@sftp-transparency.entsoe.eu  << EOF

cd TP_export
echo "hallo"

# Actual load
cd ActualTotalLoad_6.1.A
get 201[5-9]*.csv

# Load forecast
cd ..
cd DayAheadTotalLoadForecast_6.1.B
get 201[5-9]*.csv

# Generation forecast
cd ..
cd DayAheadAggregatedGeneration_14.1.C
get 201[5-9]*.csv

# Generation per type
cd ..
cd AggregatedGenerationPerType_16.1.B_C
get 201[5-9]*.csv

# Wind/solar forecast
cd ..
cd DayAheadGenerationForecastForWindAndSolar_14.1.D
get 201[5-9]*.csv

# Day ahead prices
cd ..
cd DayAheadPrices_12.1.D
get 201[5-9]*.csv

# #Commercial Schedules Forecast
# cd ..
# cd DayAheadCommercialSchedules_12.1.F
# get 201[5-9]*.csv

#Commercial Schedules
cd ..
cd TotalCommercialSchedules_12.1.F
get 201[5-9]*.csv

#Physical Flows
cd ..
cd PhysicalFlows_12.1.G
get 201[5-9]*.csv


EOF