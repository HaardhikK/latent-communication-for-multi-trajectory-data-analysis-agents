# Phase 4A Findings

Session 2 reran the long-horizon attribution matrix on Qwen3-8B 4-bit with the frozen Phase 3 repair path. The run used commit `dfdd1fca655851707840b2127ebbc0aa9cc7509b` and generation-path hash `7072860e2ace8afe`.

## Claims

- **Confirmed:** the original 7-stage latent collapse was caused by duplicate chat-templated task/prompt re-encoding into the latent KV cache. `C1_phase3_exact` was 3/15 = 0.200 while `C2_dedup` was 11/15 = 0.733; Fisher p=0.0092. Median decode cache length fell from 2828 to 532. `C2_dedup` is statistically indistinguishable from `B_textmas` (12/15 = 0.800, Fisher p=1.0000) while using 0 decoded coordination tokens.
- **Not confirmed at n=30:** latent steps remained directionally positive but did not reach significance. After Session 3 pooling, `C2_dedup` was 25/30 = 0.833 and `C3_no_latent` was 19/30 = 0.633 (delta +0.200, Fisher p=0.1432; first-attempt p=0.4118). Report this as no confirmed latent-step contribution at this sample size.
- **Greedy anchors harmed this implementation:** after Session 3 pooling, `C5_anchor` was 17/30 = 0.567 versus `C2_dedup` at 25/30 = 0.833 (delta -0.267, Fisher p=0.0470; first-attempt p=0.2789). This does not test well-formed grounding. The greedy <=24-token anchors parroted duplicated/truncated stage text into the cache and acted as a medium pollution dose: C5 median cache length was 1258, about 2.4x C2.

## Session 3 Confirmation

Session 3 added repeats 6-10 for `C2_dedup`, `C3_no_latent`, and `C5_anchor`, using the exact Session 2 generation commit/hash (`dfdd1fca655851707840b2127ebbc0aa9cc7509b`, `7072860e2ace8afe`). The imported zip had no model weights, HF cache, or token-like strings; hidden-signal smoke and latent tool-roundtrip both passed.

| Variant | Runs | Final pass | Wilson CI | First-attempt pass | Median cache len | Failure rows |
|---|---:|---:|---|---:|---:|---|
| C2_dedup | 30 | 0.833 | [0.664, 0.927] | 0.733 | 532 | 3 runtime, 2 semantic |
| C3_no_latent | 30 | 0.633 | [0.455, 0.781] | 0.600 | 504 | 7 runtime, 4 semantic |
| C5_anchor | 30 | 0.567 | [0.392, 0.726] | 0.567 | 1258 | 12 runtime, 1 semantic |

Pre-registered decision: `C2_dedup` versus `C3_no_latent` at n=30 gives Fisher p=0.1432, so the latent-step contribution is not confirmed. The stronger follow-up is the per-stage execute-observe-continue benchmark, where latent vectors have more work to do than in this one-execution planning horizon.

## Cache Dose-Response

Among latent-step variants, final pass rate falls as duplicated decoded/chat-templated cache text increases. This is a dose-response signal for cache pollution, not for latent steps in isolation: `C3_no_latent` sits off the curve with a short cache but weaker accuracy, so cache composition and whether latent steps are present both matter.

| Variant | Latent steps? | Median cache_len_at_decode | Final pass | Interpretation |
|---|---:|---:|---:|---|
| C2_dedup | yes | 532 | 0.833 | Clean latent-step cache, best observed C path |
| C5_anchor | yes | 1258 | 0.567 | Medium decoded-anchor pollution dose, significantly worse than C2 |
| C1_phase3_exact | yes | 2828 | 0.200 | Heavy duplicate full-task/chat-template pollution |
| C3_no_latent | no | 504 | 0.633 | Short cache but no latent-step updates; off the pollution curve |

<svg xmlns="http://www.w3.org/2000/svg" width="620" height="310" viewBox="0 0 620 310" role="img" aria-label="Final pass rate versus median cache length for Phase 4A latent variants">
  <rect x="0" y="0" width="620" height="310" fill="white"/>
  <line x1="70" y1="250" x2="570" y2="250" stroke="#444" stroke-width="1.5"/>
  <line x1="70" y1="40" x2="70" y2="250" stroke="#444" stroke-width="1.5"/>
  <text x="300" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Median cache_len_at_decode</text>
  <text x="18" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 18 150)">Final pass rate</text>
  <text x="70" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">500</text>
  <text x="254" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">1250</text>
  <text x="438" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">2000</text>
  <text x="561" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">3000</text>
  <text x="50" y="250" text-anchor="end" font-family="Arial, sans-serif" font-size="11">0.0</text>
  <text x="50" y="145" text-anchor="end" font-family="Arial, sans-serif" font-size="11">0.5</text>
  <text x="50" y="40" text-anchor="end" font-family="Arial, sans-serif" font-size="11">1.0</text>
  <polyline points="78,75 256,131 563,208" fill="none" stroke="#1f77b4" stroke-width="2.5"/>
  <circle cx="78" cy="75" r="6" fill="#1f77b4"/>
  <text x="90" y="70" font-family="Arial, sans-serif" font-size="12">C2 0.833</text>
  <circle cx="256" cy="131" r="6" fill="#1f77b4"/>
  <text x="268" y="126" font-family="Arial, sans-serif" font-size="12">C5 0.567</text>
  <circle cx="563" cy="208" r="6" fill="#1f77b4"/>
  <text x="470" y="203" font-family="Arial, sans-serif" font-size="12">C1 0.200</text>
  <circle cx="70" cy="117" r="6" fill="#d62728"/>
  <text x="84" y="118" font-family="Arial, sans-serif" font-size="12">C3 off-curve 0.633</text>
  <text x="70" y="24" font-family="Arial, sans-serif" font-size="14" font-weight="bold">Cache pollution dose-response among latent-step variants</text>
</svg>

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

Anchors were greedy, <=24-token decoded stage summaries appended as raw continuation text. This implementation parroted duplicated/truncated stage text into the cache and should be read as a medium pollution-dose test, not as a test of well-formed grounding. The table dumps the per-run anchor-quality classification and pass/fail outcome. Because anchor decoding was greedy and deterministic, anchor content was identical across repeats within each family; repeats are therefore not independent with respect to anchor text content, even though code generation still used the normal repeat seeds.

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

Interpretation: C5 failures were not driven by empty or degenerate code. The anchor text often contained duplicated/truncated stage fragments, and the added decoded text roughly doubled-to-tripled the C2 cache length, matching the cache-pollution mechanism. The orders-family anchors are the clearest concrete failure: they injected the wrong formula `net_revenue = units * (unit_price - discount_amount)` instead of the task formula `units * unit_price * (1 - discount_rate)`, and all 5 orders C5 runs failed. The correct conclusion is narrow: this greedy-anchor implementation significantly harmed C5 versus C2 (Fisher p=0.0470), but it does not rule out a future, well-formed grounding channel.

## Phase 4C Xlong Attempt 1

Kaggle Version #8 ran the first 9-stage ceiling matrix on Qwen3-8B 4-bit, single visible Tesla T4, commit `a3e4457671d432cdef6e4b88a83d4b66d633cf75`, generation-path hash `6ce3d3c4492384d2`. The result zip imported cleanly: dependency/GPU guard passed, hidden-signal smoke passed, latent tool-roundtrip passed, `60/60` rows completed, and the zip audit found no HF cache, model weights, or token-like secrets.

Pre-registered A-gate failed: `A_single` was 4/15 = 0.267, far below the required `>=13/15`. Therefore this run is **not interpretable** as a latent-vs-text ceiling comparison. The mode rows are recorded only as diagnostics:

| Mode | Variant | Runs | Final pass | First-attempt pass | Median cache len |
|---|---|---:|---:|---:|---:|
| A_single | - | 15 | 0.267 | 0.267 | 0 |
| B_textmas | - | 15 | 0.400 | 0.333 | 0 |
| C_latentmas | C1_phase3_exact | 15 | 0.400 | 0.067 | 4259 |
| C_latentmas | C2_dedup | 15 | 0.067 | 0.067 | 679 |

Failure inspection showed mostly high-partial-credit semantic slips rather than empty/invalid code: sensor missed only the global `mean_alert_rate` formula, campaign often used mean row CTR instead of global CTR and sometimes returned channel_id instead of channel name, and one orders run missed the inclusive high-value customer count. The xlong task contract was therefore clarified/simplified while preserving 9 dependent stages. Version #8 B/C gaps should not be cited.
