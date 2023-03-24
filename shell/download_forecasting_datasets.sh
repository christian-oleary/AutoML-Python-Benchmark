#!/bin/bash

mkdir -p ./data/forecasting

# Webpage URL format: https://zenodo.org/record/4659727

curl https://zenodo.org/record/4659727/files/australian_electricity_demand_dataset.zip?download=1 --output ./data/forecasting/australian_electricity_demand_dataset.zip
curl https://zenodo.org/record/5121965/files/bitcoin_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/bitcoin_dataset_with_missing_values.zip
curl https://zenodo.org/record/5122101/files/bitcoin_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/bitcoin_dataset_without_missing_values.zip
curl https://zenodo.org/record/4656022/files/car_parts_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/car_parts_dataset_with_missing_values.zip
curl https://zenodo.org/record/4656021/files/car_parts_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/car_parts_dataset_without_missing_values.zip
curl https://zenodo.org/record/4656042/files/cif_2016_dataset.zip?download=1 --output ./data/forecasting/cif_2016_dataset.zip
curl https://zenodo.org/record/4656009/files/covid_deaths_dataset.zip?download=1 --output ./data/forecasting/covid_deaths_dataset.zip
curl https://zenodo.org/record/4663809/files/covid_mobility_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/covid_mobility_dataset_without_missing_values.zip
curl https://zenodo.org/record/4654802/files/dominick_dataset.zip?download=1 --output ./data/forecasting/dominick_dataset.zip
curl https://zenodo.org/record/4656069/files/elecdemand_dataset.zip?download=1 --output ./data/forecasting/elecdemand_dataset.zip

curl https://zenodo.org/record/4656140/files/electricity_hourly_dataset.zip?download=1 --output ./data/forecasting/electricity_hourly_dataset.zip
curl https://zenodo.org/record/4656141/files/electricity_weekly_dataset.zip?download=1 --output ./data/forecasting/electricity_weekly_dataset.zip
curl https://zenodo.org/record/4654833/files/fred_md_dataset.zip?download=1 --output ./data/forecasting/fred_md_dataset.zip
curl https://zenodo.org/record/4656014/files/hospital_dataset.zip?download=1 --output ./data/forecasting/hospital_dataset.zip
curl https://zenodo.org/record/4656664/files/kaggle_web_traffic_weekly_dataset.zip?download=1 --output ./data/forecasting/kaggle_web_traffic_weekly_dataset.zip
curl https://zenodo.org/record/4656080/files/kaggle_web_traffic_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/kaggle_web_traffic_dataset_with_missing_values.zip
curl https://zenodo.org/record/4656075/files/kaggle_web_traffic_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/kaggle_web_traffic_dataset_without_missing_values.zip
curl https://zenodo.org/record/7370977/files/web_traffic_extended_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/kaggle_web_traffic_extended_dataset_with_missing_values.zip
curl https://zenodo.org/record/7371038/files/web_traffic_extended_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/kaggle_web_traffic_extended_dataset_without_missing_values.zip
curl https://zenodo.org/record/4656756/files/kdd_cup_2018_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/kdd_cup_2018_dataset_without_missing_values.zip

curl https://zenodo.org/record/4656072/files/london_smart_meters_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/london_smart_meters_dataset_with_missing_values.zip
curl https://zenodo.org/record/4656091/files/london_smart_meters_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/london_smart_meters_dataset_without_missing_values.zip
curl https://zenodo.org/record/4656159/files/m1_monthly_dataset.zip?download=1 --output ./data/forecasting/m1_monthly_dataset.zip
curl https://zenodo.org/record/4656154/files/m1_quarterly_dataset.zip?download=1 --output ./data/forecasting/m1_quarterly_dataset.zip
curl https://zenodo.org/record/4656193/files/m1_yearly_dataset.zip?download=1 --output ./data/forecasting/m1_yearly_dataset.zip
curl https://zenodo.org/record/4656298/files/m3_monthly_dataset.zip?download=1 --output ./data/forecasting/m3_monthly_dataset.zip
curl https://zenodo.org/record/4656335/files/m3_other_dataset.zip?download=1 --output ./data/forecasting/m3_other_dataset.zip
curl https://zenodo.org/record/4656262/files/m3_quarterly_dataset.zip?download=1 --output ./data/forecasting/m3_quarterly_dataset.zip
curl https://zenodo.org/record/4656222/files/m3_yearly_dataset.zip?download=1 --output ./data/forecasting/m3_yearly_dataset.zip
curl https://zenodo.org/record/4656548/files/m4_daily_dataset.zip?download=1 --output ./data/forecasting/m4_daily_dataset.zip

curl https://zenodo.org/record/4656589/files/m4_hourly_dataset.zip?download=1 --output ./data/forecasting/m4_hourly_dataset.zip
curl https://zenodo.org/record/4656480/files/m4_monthly_dataset.zip?download=1 --output ./data/forecasting/m4_monthly_dataset.zip
curl https://zenodo.org/record/4656410/files/m4_quarterly_dataset.zip?download=1 --output ./data/forecasting/m4_quarterly_dataset.zip
curl https://zenodo.org/record/4656522/files/m4_weekly_dataset.zip?download=1 --output ./data/forecasting/m4_weekly_dataset.zip
curl https://zenodo.org/record/4656379/files/m4_yearly_dataset.zip?download=1 --output ./data/forecasting/m4_yearly_dataset.zip
curl https://zenodo.org/record/4656110/files/nn5_daily_dataset_with_missing_values.zip?download=1.zip --output ./data/forecasting/nn5_daily_dataset_with_missing_values.zip
curl https://zenodo.org/record/4656117/files/nn5_daily_dataset_without_missing_values.zip?download=1.zip --output ./data/forecasting/nn5_daily_dataset_without_missing_values.zip
curl https://zenodo.org/record/4656125/files/nn5_weekly_dataset.zip?download=1 --output ./data/forecasting/nn5_weekly_dataset.zip
curl https://zenodo.org/record/5184708/files/oikolab_weather_dataset.zip?download=1 --output ./data/forecasting/oikolab_weather_dataset.zip
curl https://zenodo.org/record/4656626/files/pedestrian_counts_dataset.zip?download=1 --output ./data/forecasting/pedestrian_counts_dataset.zip

curl https://zenodo.org/record/5122114/files/rideshare_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/rideshare_dataset_with_missing_values.zip
curl https://zenodo.org/record/5122232/files/rideshare_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/rideshare_dataset_without_missing_values.zip
curl https://zenodo.org/record/4656058/files/saugeenday_dataset.zip?download=1 --output ./data/forecasting/saugeenday_dataset.zip
curl https://zenodo.org/record/4656027/files/solar_4_seconds_dataset.zip?download=1 --output ./data/forecasting/solar_4_seconds_dataset.zip
curl https://zenodo.org/record/4656144/files/solar_10_minutes_dataset.zip?download=1 --output ./data/forecasting/solar_10_minutes_dataset.zip
curl https://zenodo.org/record/4656151/files/solar_weekly_dataset.zip?download=1 --output ./data/forecasting/solar_weekly_dataset.zip
curl https://zenodo.org/record/4654773/files/sunspot_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/sunspot_dataset_with_missing_values.zip
curl https://zenodo.org/record/4654722/files/sunspot_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/sunspot_dataset_without_missing_values.zip
curl https://zenodo.org/record/5129073/files/temperature_rain_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/temperature_rain_dataset_with_missing_values.zip
curl https://zenodo.org/record/5129091/files/temperature_rain_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/temperature_rain_dataset_without_missing_values.zip

curl https://zenodo.org/record/4656096/files/tourism_monthly_dataset.zip?download=1 --output ./data/forecasting/tourism_monthly_dataset.zip
curl https://zenodo.org/record/4656093/files/tourism_quarterly_dataset.zip?download=1 --output ./data/forecasting/tourism_quarterly_dataset.zip
curl https://zenodo.org/record/4656132/files/traffic_hourly_dataset.zip?download=1 --output ./data/forecasting/traffic_hourly_dataset.zip
curl https://zenodo.org/record/4656135/files/traffic_weekly_dataset.zip?download=1 --output ./data/forecasting/traffic_weekly_dataset.zip
curl https://zenodo.org/record/4656103/files/tourism_yearly_dataset.zip?download=1 --output ./data/forecasting/tourism_yearly_dataset.zip
curl https://zenodo.org/record/4656049/files/us_births_dataset.zip?download=1 --output ./data/forecasting/us_births_dataset.zip
curl https://zenodo.org/record/5122535/files/vehicle_trips_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/vehicle_trips_dataset_with_missing_values.zip
curl https://zenodo.org/record/5122537/files/vehicle_trips_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/vehicle_trips_dataset_without_missing_values.zip
curl https://zenodo.org/record/4654822/files/weather_dataset.zip?download=1 --output ./data/forecasting/weather_dataset.zip
curl https://zenodo.org/record/4656032/files/wind_4_seconds_dataset.zip?download=1 --output ./data/forecasting/wind_4_seconds_dataset.zip

curl https://zenodo.org/record/4654909/files/wind_farms_minutely_dataset_with_missing_values.zip?download=1 --output ./data/forecasting/wind_farms_minutely_dataset_with_missing_values.zip
curl https://zenodo.org/record/4654858/files/wind_farms_minutely_dataset_without_missing_values.zip?download=1 --output ./data/forecasting/wind_farms_minutely_dataset_without_missing_values.zip
