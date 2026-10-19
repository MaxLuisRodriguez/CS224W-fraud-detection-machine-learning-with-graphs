## rel-avito: Avito online classifieds: users, ads, search queries, and impression / click / visit streams.

Validation cutoff 2015-05-08, test cutoff 2015-05-14.

| Table | Rows | Primary key | Time column | Foreign keys |
|---|---:|---|---|---|
| `VisitStream` | 6,454,562 | none | ViewDate | UserID -> UserInfo, AdID -> AdsInfo |
| `AdsInfo` | 5,960,558 | AdID | none | LocationID -> Location, CategoryID -> Category |
| `SearchStream` | 9,254,702 | none | SearchDate | SearchID -> SearchInfo, AdID -> AdsInfo |
| `SearchInfo` | 2,579,289 | SearchID | SearchDate | UserID -> UserInfo, LocationID -> Location, CategoryID -> Category |
| `Category` | 68 | CategoryID | none | none |
| `PhoneRequestsStream` | 302,974 | none | PhoneRequestDate | UserID -> UserInfo, AdID -> AdsInfo |
| `UserInfo` | 98,250 | UserID | none | none |
| `Location` | 3,512 | LocationID | none | none |

### Task `user-clicks`: Predict whether the each customer will click on more than one ads in the next 4 days.

Type binary_classification; entity `UserInfo.UserID`; label `num_click`; horizon 4 days.

| Split | Rows | Prediction dates | Positive rate |
|---|---:|---:|---:|
| train | 59,454 | 3 | 0.039 |
| val | 21,183 | 1 | 0.035 |
| test | 47,996 | held out | held out |
