# Data

## carbonmonitor_china_daily.csv

Daily fossil CO2 emissions for China by sector, in Mt CO2/day, from 2019-01-01 to 2026-03-31
(2,647 days x 6 sectors). Columns: `country`, `date` (DD/MM/YYYY), `sector`, `value`.

Source: Carbon Monitor, https://carbonmonitor.org. The rows for China were extracted unchanged from
the global daily dataset (`carbonmonitor-global_datas_2026-05-22.csv`, downloaded 2026-05-22).
Carbon Monitor data are distributed under the Creative Commons Attribution 4.0 International
license (CC BY 4.0).

Liu, Z., Ciais, P., Deng, Z. et al. Near-real-time monitoring of global CO2 emissions reveals the
effects of the COVID-19 pandemic. Nature Communications 11, 5172 (2020).

## cn_official_holidays.csv

Statutory holidays (`type = holiday`) and adjusted working days (`type = workday`) in China for
2019-2026, compiled from the annual holiday arrangement notices of the General Office of the State
Council. Columns: `date`, `type`, `festival`, `notice_year`. The 2020 Spring Festival entry includes
the extension to 2 February 2020.
