# Phase 4A Findings

Session 2 reran the long-horizon attribution matrix on Qwen3-8B 4-bit with the frozen Phase 3 repair path. The run used commit `dfdd1fca655851707840b2127ebbc0aa9cc7509b` and generation-path hash `7072860e2ace8afe`.

## Claims

- **Confirmed:** the original 7-stage latent collapse was caused by duplicate chat-templated task/prompt re-encoding into the latent KV cache. `C1_phase3_exact` was 3/15 = 0.200 while `C2_dedup` was 11/15 = 0.733; Fisher p=0.0092. Median decode cache length fell from 2828 to 532. `C2_dedup` is statistically indistinguishable from `B_textmas` (12/15 = 0.800, Fisher p=1.0000) while using 0 decoded coordination tokens.
- **Directional, not yet confirmed:** latent steps added +0.333 over the stage-text-only cache (`C2_dedup` 11/15 = 0.733 vs `C3_no_latent` 6/15 = 0.400), but Fisher p=0.1394; the n=30 confirmation run decides whether this becomes a claim.
- **No evidence decoded anchors help:** `C5_anchor` was 8/15 = 0.533, worse than `C2_dedup` by -0.200 (Fisher p=0.4497). C5 median cache length was 1258, about 2.4x C2, consistent with anchors re-polluting the cache.

## By Variant

| Mode | Variant | Runs | Final pass | Wilson CI | First-attempt pass | Median cache len |
|---|---|---:|---:|---|---:|---:|
| A_single | - | 15 | 1.000 | [0.796, 1.000] | 1.000 | 0 |
| B_textmas | - | 15 | 0.800 | [0.548, 0.930] | 0.800 | 0 |
| C_latentmas | C1_phase3_exact | 15 | 0.200 | [0.070, 0.452] | 0.200 | 2828 |
| C_latentmas | C2_dedup | 15 | 0.733 | [0.480, 0.891] | 0.667 | 532 |
| C_latentmas | C3_no_latent | 15 | 0.400 | [0.198, 0.643] | 0.333 | 504 |
| C_latentmas | C5_anchor | 15 | 0.533 | [0.301, 0.752] | 0.533 | 1258 |

## By Family

| Mode | Variant | Family | Runs | Final pass | First-attempt pass | Median cache len |
|---|---|---|---:|---:|---:|---:|
| A_single | - | campaign_roi | 5 | 1.000 | 1.000 | 0 |
| A_single | - | orders_kpi | 5 | 1.000 | 1.000 | 0 |
| A_single | - | sensor_quality | 5 | 1.000 | 1.000 | 0 |
| B_textmas | - | campaign_roi | 5 | 1.000 | 1.000 | 0 |
| B_textmas | - | orders_kpi | 5 | 0.600 | 0.600 | 0 |
| B_textmas | - | sensor_quality | 5 | 0.800 | 0.800 | 0 |
| C_latentmas | C1_phase3_exact | campaign_roi | 5 | 0.000 | 0.000 | 2828 |
| C_latentmas | C1_phase3_exact | orders_kpi | 5 | 0.000 | 0.000 | 3227 |
| C_latentmas | C1_phase3_exact | sensor_quality | 5 | 0.600 | 0.600 | 2725 |
| C_latentmas | C2_dedup | campaign_roi | 5 | 0.800 | 0.800 | 532 |
| C_latentmas | C2_dedup | orders_kpi | 5 | 0.600 | 0.400 | 595 |
| C_latentmas | C2_dedup | sensor_quality | 5 | 0.800 | 0.800 | 501 |
| C_latentmas | C3_no_latent | campaign_roi | 5 | 0.200 | 0.200 | 504 |
| C_latentmas | C3_no_latent | orders_kpi | 5 | 0.200 | 0.000 | 567 |
| C_latentmas | C3_no_latent | sensor_quality | 5 | 0.800 | 0.800 | 473 |
| C_latentmas | C5_anchor | campaign_roi | 5 | 0.600 | 0.600 | 1258 |
| C_latentmas | C5_anchor | orders_kpi | 5 | 0.000 | 0.000 | 1357 |
| C_latentmas | C5_anchor | sensor_quality | 5 | 1.000 | 1.000 | 1222 |

## Failure Classes

| Mode | Variant | Failure type | Rows |
|---|---|---|---:|
| A_single | - | passed | 15 |
| B_textmas | - | passed | 12 |
| B_textmas | - | runtime_bug | 1 |
| B_textmas | - | semantic_scorer_slip | 2 |
| C_latentmas | C1_phase3_exact | passed | 3 |
| C_latentmas | C1_phase3_exact | runtime_bug | 8 |
| C_latentmas | C1_phase3_exact | semantic_scorer_slip | 4 |
| C_latentmas | C2_dedup | passed | 11 |
| C_latentmas | C2_dedup | runtime_bug | 2 |
| C_latentmas | C2_dedup | semantic_scorer_slip | 2 |
| C_latentmas | C3_no_latent | passed | 6 |
| C_latentmas | C3_no_latent | runtime_bug | 6 |
| C_latentmas | C3_no_latent | semantic_scorer_slip | 3 |
| C_latentmas | C5_anchor | passed | 8 |
| C_latentmas | C5_anchor | runtime_bug | 7 |

## C5 Anchor Forensics

Anchors were greedy, <=24-token decoded stage summaries appended as raw continuation text. The table dumps the per-run anchor-quality classification and pass/fail outcome.

| Task | Repeat | Passed | Cache len | Anchor quality | Quality counts | Anchor dump |
|---|---:|---:|---:|---|---|---|
| campaign_roi_long | 1 | True | 1258 | wrong | {'faithful': 2, 'vague': 1, 'wrong': 4} | Stage 1/7: Read campaigns.csv and channels.csv.; Follow the authoritative task specification for this stage.; Stage 3/7: Join channel names on channel_id.Stage 4/7: Compute CTR, CVR; Stage 4/7: Compute CTR, CVR, and ROI.Stage 5/7: Aggregate channel revenue; Stage 5/7: Aggregate channel revenue and identify the top channel by revenue.Stage 6/7: Fit a; Stage 6/7: Fit a simple numpy polynomial line predicting revenue from spend and predict revenue for spend 25; Stage 7/7: Write campaign_long_report.json with rows_clean, total_profit, top_channel_by_revenue, |
| campaign_roi_long | 2 | True | 1258 | wrong | {'faithful': 2, 'vague': 1, 'wrong': 4} | Stage 1/7: Read campaigns.csv and channels.csv.; Follow the authoritative task specification for this stage.; Stage 3/7: Join channel names on channel_id.Stage 4/7: Compute CTR, CVR; Stage 4/7: Compute CTR, CVR, and ROI.Stage 5/7: Aggregate channel revenue; Stage 5/7: Aggregate channel revenue and identify the top channel by revenue.Stage 6/7: Fit a; Stage 6/7: Fit a simple numpy polynomial line predicting revenue from spend and predict revenue for spend 25; Stage 7/7: Write campaign_long_report.json with rows_clean, total_profit, top_channel_by_revenue, |
| campaign_roi_long | 3 | True | 1258 | wrong | {'faithful': 2, 'vague': 1, 'wrong': 4} | Stage 1/7: Read campaigns.csv and channels.csv.; Follow the authoritative task specification for this stage.; Stage 3/7: Join channel names on channel_id.Stage 4/7: Compute CTR, CVR; Stage 4/7: Compute CTR, CVR, and ROI.Stage 5/7: Aggregate channel revenue; Stage 5/7: Aggregate channel revenue and identify the top channel by revenue.Stage 6/7: Fit a; Stage 6/7: Fit a simple numpy polynomial line predicting revenue from spend and predict revenue for spend 25; Stage 7/7: Write campaign_long_report.json with rows_clean, total_profit, top_channel_by_revenue, |
| campaign_roi_long | 4 | False | 1258 | wrong | {'faithful': 2, 'vague': 1, 'wrong': 4} | Stage 1/7: Read campaigns.csv and channels.csv.; Follow the authoritative task specification for this stage.; Stage 3/7: Join channel names on channel_id.Stage 4/7: Compute CTR, CVR; Stage 4/7: Compute CTR, CVR, and ROI.Stage 5/7: Aggregate channel revenue; Stage 5/7: Aggregate channel revenue and identify the top channel by revenue.Stage 6/7: Fit a; Stage 6/7: Fit a simple numpy polynomial line predicting revenue from spend and predict revenue for spend 25; Stage 7/7: Write campaign_long_report.json with rows_clean, total_profit, top_channel_by_revenue, |
| campaign_roi_long | 5 | False | 1258 | wrong | {'faithful': 2, 'vague': 1, 'wrong': 4} | Stage 1/7: Read campaigns.csv and channels.csv.; Follow the authoritative task specification for this stage.; Stage 3/7: Join channel names on channel_id.Stage 4/7: Compute CTR, CVR; Stage 4/7: Compute CTR, CVR, and ROI.Stage 5/7: Aggregate channel revenue; Stage 5/7: Aggregate channel revenue and identify the top channel by revenue.Stage 6/7: Fit a; Stage 6/7: Fit a simple numpy polynomial line predicting revenue from spend and predict revenue for spend 25; Stage 7/7: Write campaign_long_report.json with rows_clean, total_profit, top_channel_by_revenue, |
| orders_kpi_long | 1 | False | 1357 | wrong | {'faithful': 5, 'wrong': 2} | Read orders.csv and customers.csv into DataFrames, preserving all columns and handling missing values as specified.; Coerce numeric columns in orders.csv to float/int, drop rows with missing units, and retain all non-numeric columns; Compute gross_revenue as units * unit_price, net_revenue as (units * (unit_price - discount_amount)),; Join orders with customers on customer_id to include region and tier informationJoin orders with customers on customer_id to include region and; Aggregate net_revenue by customer_id and count customers with net_revenue >= 30Aggregate net_revenue by customer; Fit LinearRegression on net_revenue using units, unit_price, and discount_rate as features, then compute training MAE; Write one-row orders_long_report.csv with columns rows_clean, total_net_revenue, total_margin, top_region, high |
| orders_kpi_long | 2 | False | 1357 | wrong | {'faithful': 5, 'wrong': 2} | Read orders.csv and customers.csv into DataFrames, preserving all columns and handling missing values as specified.; Coerce numeric columns in orders.csv to float/int, drop rows with missing units, and retain all non-numeric columns; Compute gross_revenue as units * unit_price, net_revenue as (units * (unit_price - discount_amount)),; Join orders with customers on customer_id to include region and tier informationJoin orders with customers on customer_id to include region and; Aggregate net_revenue by customer_id and count customers with net_revenue >= 30Aggregate net_revenue by customer; Fit LinearRegression on net_revenue using units, unit_price, and discount_rate as features, then compute training MAE; Write one-row orders_long_report.csv with columns rows_clean, total_net_revenue, total_margin, top_region, high |
| orders_kpi_long | 3 | False | 1357 | wrong | {'faithful': 5, 'wrong': 2} | Read orders.csv and customers.csv into DataFrames, preserving all columns and handling missing values as specified.; Coerce numeric columns in orders.csv to float/int, drop rows with missing units, and retain all non-numeric columns; Compute gross_revenue as units * unit_price, net_revenue as (units * (unit_price - discount_amount)),; Join orders with customers on customer_id to include region and tier informationJoin orders with customers on customer_id to include region and; Aggregate net_revenue by customer_id and count customers with net_revenue >= 30Aggregate net_revenue by customer; Fit LinearRegression on net_revenue using units, unit_price, and discount_rate as features, then compute training MAE; Write one-row orders_long_report.csv with columns rows_clean, total_net_revenue, total_margin, top_region, high |
| orders_kpi_long | 4 | False | 1357 | wrong | {'faithful': 5, 'wrong': 2} | Read orders.csv and customers.csv into DataFrames, preserving all columns and handling missing values as specified.; Coerce numeric columns in orders.csv to float/int, drop rows with missing units, and retain all non-numeric columns; Compute gross_revenue as units * unit_price, net_revenue as (units * (unit_price - discount_amount)),; Join orders with customers on customer_id to include region and tier informationJoin orders with customers on customer_id to include region and; Aggregate net_revenue by customer_id and count customers with net_revenue >= 30Aggregate net_revenue by customer; Fit LinearRegression on net_revenue using units, unit_price, and discount_rate as features, then compute training MAE; Write one-row orders_long_report.csv with columns rows_clean, total_net_revenue, total_margin, top_region, high |
| orders_kpi_long | 5 | False | 1357 | wrong | {'faithful': 5, 'wrong': 2} | Read orders.csv and customers.csv into DataFrames, preserving all columns and handling missing values as specified.; Coerce numeric columns in orders.csv to float/int, drop rows with missing units, and retain all non-numeric columns; Compute gross_revenue as units * unit_price, net_revenue as (units * (unit_price - discount_amount)),; Join orders with customers on customer_id to include region and tier informationJoin orders with customers on customer_id to include region and; Aggregate net_revenue by customer_id and count customers with net_revenue >= 30Aggregate net_revenue by customer; Fit LinearRegression on net_revenue using units, unit_price, and discount_rate as features, then compute training MAE; Write one-row orders_long_report.csv with columns rows_clean, total_net_revenue, total_margin, top_region, high |
| sensor_quality_long | 1 | True | 1222 | wrong | {'faithful': 4, 'wrong': 3} | Stage 1/7: Read readings.csv and sensor_meta.csv.; Stage 2/7: Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.; Stage 3/7: Join sensor metadata.Stage 3/7: Join sensor metadata.Stage 4/; Stage 4/7: Parse timestamp and derive hour.Stage 4/7: Parse timestamp and derive hour.; Stage 5/7: Compute adjusted_temp, adjusted_pressure, and alert where adjusted_temp > 76 or adjusted; Stage 6/7: Aggregate site-hour alerts and write sensor_long_hourly_summary.csv.Stage 6/7:; Stage 7/7: Write sensor_long_report.json with rows_clean, total_alerts, worst_site, peak_hour |
| sensor_quality_long | 2 | True | 1222 | wrong | {'faithful': 4, 'wrong': 3} | Stage 1/7: Read readings.csv and sensor_meta.csv.; Stage 2/7: Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.; Stage 3/7: Join sensor metadata.Stage 3/7: Join sensor metadata.Stage 4/; Stage 4/7: Parse timestamp and derive hour.Stage 4/7: Parse timestamp and derive hour.; Stage 5/7: Compute adjusted_temp, adjusted_pressure, and alert where adjusted_temp > 76 or adjusted; Stage 6/7: Aggregate site-hour alerts and write sensor_long_hourly_summary.csv.Stage 6/7:; Stage 7/7: Write sensor_long_report.json with rows_clean, total_alerts, worst_site, peak_hour |
| sensor_quality_long | 3 | True | 1222 | wrong | {'faithful': 4, 'wrong': 3} | Stage 1/7: Read readings.csv and sensor_meta.csv.; Stage 2/7: Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.; Stage 3/7: Join sensor metadata.Stage 3/7: Join sensor metadata.Stage 4/; Stage 4/7: Parse timestamp and derive hour.Stage 4/7: Parse timestamp and derive hour.; Stage 5/7: Compute adjusted_temp, adjusted_pressure, and alert where adjusted_temp > 76 or adjusted; Stage 6/7: Aggregate site-hour alerts and write sensor_long_hourly_summary.csv.Stage 6/7:; Stage 7/7: Write sensor_long_report.json with rows_clean, total_alerts, worst_site, peak_hour |
| sensor_quality_long | 4 | True | 1222 | wrong | {'faithful': 4, 'wrong': 3} | Stage 1/7: Read readings.csv and sensor_meta.csv.; Stage 2/7: Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.; Stage 3/7: Join sensor metadata.Stage 3/7: Join sensor metadata.Stage 4/; Stage 4/7: Parse timestamp and derive hour.Stage 4/7: Parse timestamp and derive hour.; Stage 5/7: Compute adjusted_temp, adjusted_pressure, and alert where adjusted_temp > 76 or adjusted; Stage 6/7: Aggregate site-hour alerts and write sensor_long_hourly_summary.csv.Stage 6/7:; Stage 7/7: Write sensor_long_report.json with rows_clean, total_alerts, worst_site, peak_hour |
| sensor_quality_long | 5 | True | 1222 | wrong | {'faithful': 4, 'wrong': 3} | Stage 1/7: Read readings.csv and sensor_meta.csv.; Stage 2/7: Coerce raw_temp/raw_pressure, keep ok readings, and drop rows missing raw_temp.; Stage 3/7: Join sensor metadata.Stage 3/7: Join sensor metadata.Stage 4/; Stage 4/7: Parse timestamp and derive hour.Stage 4/7: Parse timestamp and derive hour.; Stage 5/7: Compute adjusted_temp, adjusted_pressure, and alert where adjusted_temp > 76 or adjusted; Stage 6/7: Aggregate site-hour alerts and write sensor_long_hourly_summary.csv.Stage 6/7:; Stage 7/7: Write sensor_long_report.json with rows_clean, total_alerts, worst_site, peak_hour |

Anchor quality vs outcome:

| Outcome | Runs | Median cache len | Anchor qualities |
|---|---:|---:|---|
| True | 8 | 1222 | {'wrong': 8} |
| False | 7 | 1357 | {'wrong': 7} |

Interpretation: C5 failures were not driven by empty or degenerate code. The anchor text often contained duplicated/truncated stage fragments, and the added decoded text roughly doubled-to-tripled the C2 cache length, matching the cache-pollution mechanism.
