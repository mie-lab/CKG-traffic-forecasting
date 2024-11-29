# raw_data Directory
This directory is intended to store the project's raw data. As the data files are not included in the repository, please follow the instructions below to download and place them here.

## Download Instructions
Download the raw data from the following link(s):
- [raw_data.zip](https://figshare.com/s/efbfc48381385ea48b83)

## Directory Structure
We provide test datasets for running the proposed CKG-GNN model, located in `./raw_data/speed_test_data/`. The test dataset was collected on 23/03/2022, from 00:00 to 23:59, with a 10-minute interval. File details and formats:

- **`speed_test_data.geo`**: Describes geographic entity information, including:
  - `geo_id`, `type`, `coordinates` of road segments.
  
- **`speed_test_data.rel`**: Describes geographic relationships between road segments:
  - Columns: `rel_id`, `type`, `origin_id`, `destination_id`, `link_weight`.
  - `origin_id` and `destination_id` belong to `geo_id`. `link_weight` defines edge weights (1 for adjacent, 0 for non-adjacent).

- **`speed_test_data.dyna`**: Describes traffic speed data over time:
  - Columns: `dyna_id`, `type`, `time`, `entity_id`, `traffic_speed`.

## Notes
- If the data is not placed correctly, the code may raise errors during execution.