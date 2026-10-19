## rel-f1: Formula 1 motorsport database: races, drivers, constructors, circuits, race results, qualifying, and championship standings.

Validation cutoff 2005-01-01, test cutoff 2010-01-01.

| Table | Rows | Primary key | Time column | Foreign keys |
|---|---:|---|---|---|
| `races` | 1,101 | raceId | date | circuitId -> circuits |
| `qualifying` | 9,815 | qualifyId | date | raceId -> races, driverId -> drivers, constructorId -> constructors |
| `constructor_standings` | 13,051 | constructorStandingsId | date | raceId -> races, constructorId -> constructors |
| `standings` | 34,124 | driverStandingsId | date | raceId -> races, driverId -> drivers |
| `constructors` | 211 | constructorId | none | none |
| `drivers` | 857 | driverId | none | none |
| `constructor_results` | 12,290 | constructorResultsId | date | raceId -> races, constructorId -> constructors |
| `circuits` | 77 | circuitId | none | none |
| `results` | 26,080 | resultId | date | raceId -> races, driverId -> drivers, constructorId -> constructors |

### Task `driver-top3`: Predict if each driver will qualify in the top-3 for a race within the next 1 month.

Type binary_classification; entity `drivers.driverId`; label `qualifying`; horizon 30 days.

| Split | Rows | Prediction dates | Positive rate |
|---|---:|---:|---:|
| train | 1,353 | 59 | 0.171 |
| val | 588 | 27 | 0.202 |
| test | 726 | held out | held out |
