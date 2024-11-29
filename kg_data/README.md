# kg_data Directory
This directory is intended to store the project's kg data. As the data files are not included in the repository, please follow the instructions below to download and place them here.

## Download Instructions
Download the kg data from the following link(s):
- [kg_data.zip](https://figshare.com/s/efbfc48381385ea48b83)

## Directory Structure
Spatial and temporal context datasets related to the speed dataset are located in `./kg_data/`.

### Spatial unit
- **`kg_spatial_pickle_dict.pickle`**: Incorporates spatial context data.
  - *Road*: `(road, adjacentToRoad, road)`
  - *POI*: `(poi, locatedInBuffer[Dist], road)` and `(poi, hasType, poiType)`
  - *Land use*: `(land, intersectWithBuffer[Dist], road)` and `(land, hasType, landType)`
  - *Spatial link*: `(road, spatiallyLink[Order], road)`

### Temporal unit
- **`kg_temporal_pickle_dict.pickle`**: Incorporates temporal context data.
  - *Time*: `(road, hasHour, hour)` and `(road, hasDay, day)`
  - *Traffic jams*: `(road, hasJam[PastMins], jam)`
  - *Weather*: `(road, hasTprt[PastMins], tprt)`,  `(road, hasRain[PastMins], rain)`, and `(road, hasWind[PastMins], wind)`
  - *Temporal link*: `(road, temporallyLink[Temp][Link], [Temp])`

## Notes
- If the data is not placed correctly, the code may raise errors during execution.