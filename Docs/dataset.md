

2 datasets


1. NSW Roads Traffic Volume Counts API

station_key,date,year,month,day_of_week,public_holiday,school_holiday,daily_total,hour_00,hour_01,hour_02,hour_03,hour_04,hour_05,hour_06,hour_07,hour_08,hour_09,hour_10,hour_11,hour_12,hour_13,hour_14,hour_15,hour_16,hour_17,hour_18,hour_19,hour_20,hour_21,hour_22,hour_23

Link: https://data.nsw.gov.au/data/dataset/2-nsw-roads-traffic-volume-counts-api/resource/d7e887dd-93f1-4417-acb2-0c815a4211af


File 1. Road Traffic Counts Hourly Permanent (API Generated CSVs) 
Path: datasets\Traffic_TimesOfDay\road_traffic_counts_hourly_permanent\




File 2. road_traffic_counts_station_reference (combine with File 1 to convert station_key to lat and long)

station_key,name,road_name,full_name,common_road_name,secondary_name,road_name_base,road_name_type,road_on_type,lane_count,road_classification_type,suburb,post_code,wgs84_latitude,wgs84_longitude,quality_rating

Path: datasets\Traffic_TimesOfDay\road_traffic_counts_station_reference.csv




2. Weather Data Including 3 things and will be combined with the traffic data

Weather data from the geographically closest weather station will be sourced from the Bureau of Meteorology [3]. This will include maximum and minimum daily temperatures, daily rainfall, and total daily solar radiation.

File 3:

By Suburb name

1.csv:Product code,Bureau of Meteorology station number,Year,Month,Day,Rainfall amount (millimetres),Period over which rainfall was measured (days),Quality

2.csv:Product code,Bureau of Meteorology station number,Year,Month,Day,Daily global solar exposure (MJ/m*m)

3.csv:Product code,Bureau of Meteorology station number,Year,Month,Day,Minimum temperature (Degree C),Days of accumulation of minimum temperature,Quality

4.csv:Product code,Bureau of Meteorology station number,Year,Month,Day,Maximum temperature (Degree C),Days of accumulation of maximum temperature,Quality

https://www.bom.gov.au/climate/data/

Path: datasets\Weather_Beuro_Meterology_PerDay\Substation_x\[Suburb from File 1]


File 4:

C0, NO, NO2, PM2.5, PM10 air quaility data per day based on suburb

https://www.airquality.nsw.gov.au/air-quality-data-services/data-download-facility

path: Weather_AQ/[Data Type]











optionallll

3. train data
https://opendata.transport.nsw.gov.au/data/dataset/train-station-entries-and-exits-data

path