# CKG-Traffic-Forecasting

## Introduction
We propose a **context-aware knowledge graph (CKG)** framework to effectively model and embed the spatio-temporal relationships inherent in diverse urban contexts. This framework facilitates the integration of various context datasets into machine-readable formats. Then, we strategically combine the proposed CKG framework with GNN through dual-view multi-head self-attention (MHSA) techniques to forecast traffic speed. This integration ensures that CKG-based context representations can be effectively incorporated into traffic forecasting models, leveraging the strengths of both knowledge graphs and advanced neural network architectures to improve predictive performance.

---

## Core codes
Here, we introduce the main files for the proposed CKGGNN model.

- **`run_model.py`**: Script to run the CKGGNN model. Key parameters:
  - `--task`: `"traffic_state_pred"` - Used for traffic speed prediction.
  - `--dataset`: `"speed_test_data"` - Refers to the test dataset name. Replace with your custom dataset.
  - `--model`: `"CKGGNN"` - Refers to the proposed context-based traffic forecasting model. Can be replaced with other models like TGCN, STGCN, or DCRNN.
  - `--batch_size`: `"16"` - Batch size for running the experiment.
  - `--max_epoch`: `"300"` - Maximum number of epochs for running the experiment.

- **`libcity/model/traffic_speed_prediction/CKGGNN.py`**: Implementation of the proposed CKGGNN model.
- **`libcity/pipeline/pipeline.py`**: Contains the complete pipeline for the CKGGNN model.
- **`libcity/pipeline/embedkg_template.py`**: Script for constructing context-based knowledge graphs (CKG).
- **`libcity/data/dataset/traffic_state_contextkg_dataset.py`**: Script for loading datasets.

---

## Speed dataset
We provide test datasets for running the proposed CKGGNN model, located in `./raw_data/speed_test_data/`. The test dataset was collected on 23/03/2022, from 00:00 to 23:59, with a 10-minute interval. File details and formats:

- **`speed_test_data.geo`**: Describes geographic entity information, including:
  - `geo_id`, `type`, `coordinates` of road segments.
  
- **`speed_test_data.rel`**: Describes geographic relationships between road segments:
  - Columns: `rel_id`, `type`, `origin_id`, `destination_id`, `link_weight`.
  - `origin_id` and `destination_id` belong to `geo_id`. `link_weight` defines edge weights (1 for adjacent, 0 for non-adjacent).

- **`speed_test_data.dyna`**: Describes traffic speed data over time:
  - Columns: `dyna_id`, `type`, `time`, `entity_id`, `traffic_speed`.

---

## Context dataset
Spatial and temporal context datasets related to the speed dataset are located in `./kg_data/`.

### Spatial unit
- **`kg_spatial_pickle_dict.pickle`**: Incorporates spatial context data.
  - *Road*: `(road, adjacentToRoad, road)`
  - *POI*: `(poi, locatedInBuffer[Dist], road)` and `(poi, hasType, poiType)`
  - *Land use*: `(land, intersectWithBuffer[Dist], road)` and `(land, hasType, landType)`
  - *Spatial link*: `(road, spatiallyLink[Order], road)`

### Temporal unit
- **`kg_temporal_pickle_dict.pickle`**: Incorporates temporal context data, including:
  - *Time*: `(road, hasHour, hour)` and `(road, hasDay, day)`
  - *Traffic jams*: `(road, hasJam[PastMins], jam)`
  - *Weather*: `(road, hasTprt[PastMins], tprt)`,  `(road, hasRain[PastMins], rain)`, and `(road, hasWind[PastMins], wind)`
  - *Temporal link*: `(road, temporallyLink[Temp][Link], [Temp])`

---

## Running
To test the CKGGNN model with the provided dataset, run the `run_model.py` script:

```bash
python run_model.py --task traffic_state_pred --dataset speed_test_data --model CKGGNN --batch_size 16 --max_epoch 300
```

## Related materials
The code repository includes several Python scripts provided by LibCity, organized into the following directories:

- **`./libcity/config`**: Configuration files for various models and experiments.
- **`./libcity/data`**: Scripts for handling and preprocessing datasets.
- **`./libcity/evaluator`**: Modules for evaluating model performance.
- **`./libcity/executor`**: Scripts to execute experiments and manage workflows.
- **`./libcity/model`**: Model implementations, including CKGGNN and others.
- **`./libcity/pipeline`**: End-to-end pipelines for running the CKGGNN model.

For a detailed tutorial and further documentation, please visit the [LibCity Documentation](https://bigscity-libcity-docs.readthedocs.io/en/latest/get_started/introduction.html).
