## rel-event: Event recommendation: users, events, attendance records, and social and interest signals.

Validation cutoff 2012-11-21, test cutoff 2012-11-29.

| Table | Rows | Primary key | Time column | Foreign keys |
|---|---:|---|---|---|
| `user_friends` | 30,386,403 | none | none | user -> users, friend -> users |
| `event_interest` | 15,398 | none | timestamp | event -> events, user -> users |
| `event_attendees` | 11,245,010 | none | start_time | event -> events, user_id -> users |
| `users` | 38,209 | user_id | joinedAt | none |
| `events` | 3,137,972 | event_id | start_time | user_id -> users |

### Task `user-ignore`: Predict whether a user will ignore more than 2 event invitations in the next 7 days.

Type binary_classification; entity `users.user`; label `target`; horizon 7 days.

| Split | Rows | Prediction dates | Positive rate |
|---|---:|---:|---:|
| train | 19,239 | 22 | 0.169 |
| val | 2,013 | 1 | 0.111 |
| test | 1,958 | held out | held out |
