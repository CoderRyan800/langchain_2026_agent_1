# Litter Box Monitor — Simulation Report

## 1. Run Summary

| Parameter | Value |
|-----------|-------|
| Total simulated events | 20 |
| Visits found in DB | 20 |
| Seeded anomalous events | 3 |
| DB path | `/Users/ryanmukai/Documents/github/langchain_2026_agent_1/data/litterbox.db` |

---

## 2. Identity Accuracy

| Outcome | Count | % |
|---------|-------|----|
| Correctly identified | 14 | 70% |
| Wrong identity | 3 | 15% |
| Unidentified | 3 | 15% |

### Per-cat breakdown

| Cat | Visits | Correct | Wrong | Unidentified |
|-----|--------|---------|-------|--------------|
| Anna | 4 | 4 | 0 | 0 |
| Marina | 5 | 4 | 1 | 0 |
| Luna | 6 | 5 | 0 | 1 |
| Natasha | 5 | 1 | 2 | 2 |

---

## 3. Weight Accuracy

> Cat weight is derived as `weight_entry_g − weight_pre_g`.
> Error = |measured cat weight − true cat weight|.

| Cat | True weight (g) | Mean error (g) | Std dev (g) | Samples |
|-----|----------------|----------------|-------------|---------|
| Anna | 3,200 | 21.0 | 11.6 | 4 |
| Marina | 4,000 | 22.8 | 18.3 | 5 |
| Luna | 5,000 | 32.2 | 15.4 | 6 |
| Natasha | 5,500 | 12.6 | 7.2 | 5 |

---

## 4. Waste Weight

> Waste weight = `weight_exit_g − weight_pre_g`.

| Cat | Mean waste deposited (g) | Std dev (g) | Samples |
|-----|--------------------------|-------------|---------|
| Anna | 106.0 | 26.0 | 4 |
| Marina | 61.0 | 41.4 | 5 |
| Luna | 81.0 | 43.9 | 6 |
| Natasha | 55.0 | 30.8 | 5 |

---

## 5. Sensor Coverage

| Sensor | Present | Null | Coverage |
|--------|---------|------|----------|
| Ammonia (NH₃) | 18 | 2 | 90% |
| Methane (CH₄) | 17 | 3 | 85% |

---

## 6. Anomaly Detection

> 🌱 = seeded by simulator  ⚠️ = flagged by agent health analysis

| Outcome | Count |
|---------|-------|
| Seeded anomalies detected (true positives) | 0 |
| Seeded anomalies missed (false negatives) | 3 |
| Non-seeded events flagged (false positives) | 0 |

---

## 7. Null Sensor Handling

✅ All events with null sensor readings completed without errors.

---

## 8. Raw Event Table

| # | Sim time | Cat | True wt | Pre | Entry wt | Exit wt | Waste | NH₃ | CH₄ | Seed | DB# | Identified |
|---|----------|-----|--------:|----:|---------:|--------:|------:|----:|----:|------|-----|------------|
| 1 | 2026-03-14T12:14 | Natasha | 5,500 | 1,997 | 7,492 | 2,057 | 55 | 54 | null | — | 17 | Unknown |
| 2 | 2026-03-14T16:51 | Marina | 4,000 | 2,032 | 6,025 | 2,084 | 50 | 6 | 26 | — | 18 | Marina |
| 3 | 2026-03-14T19:48 | Luna | 5,000 | 1,986 | 6,980 | 2,070 | 83 | 43 | 6 | — | 19 | Luna |
| 4 | 2026-03-15T07:44 | Marina | 4,000 | 1,998 | 6,024 | 2,031 | 38 | null | 24 | — | 20 | Marina |
| 5 | 2026-03-15T15:48 | Luna | 5,000 | 2,011 | 6,966 | 2,108 | 78 | 51 | 34 | — | 21 | Luna |
| 6 | 2026-03-16T07:22 | Luna | 5,000 | 1,997 | 6,956 | 2,016 | 34 | 21 | null | — | 22 | Luna |
| 7 | 2026-03-16T15:46 | Natasha | 5,500 | 2,001 | 7,515 | 2,043 | 55 | 16 | 38 | — | 23 | Marina |
| 8 | 2026-03-17T07:35 | Natasha | 5,500 | 1,986 | 7,461 | 2,009 | 45 | 14 | 40 | — | 24 | Natasha |
| 9 | 2026-03-17T12:36 | Marina | 4,000 | 1,984 | 5,955 | 2,095 | 92 | 7 | 11 | — | 25 | Marina |
| 10 | 2026-03-17T15:02 | Luna | 5,000 | 1,971 | 6,988 | 2,126 | 109 | 41 | 37 | — | 26 | Luna |
| 11 | 2026-03-17T19:49 | Natasha | 5,500 | 1,985 | 7,491 | 2,023 | 52 | 54 | 9 | — | 27 | Unknown |
| 12 | 2026-03-18T11:24 | Luna | 5,000 | 1,994 | 7,030 | 2,032 | 38 | null | null | — | 28 | Unknown |
| 13 | 2026-03-18T16:40 | Anna | 3,200 | 1,975 | 5,137 | 2,061 | 68 | 229 | 166 | 🌱 | 29 | Anna |
| 14 | 2026-03-18T20:10 | Luna | 5,000 | 2,008 | 7,056 | 2,101 | 91 | 190 | 91 | 🌱 | 30 | Luna |
| 15 | 2026-03-19T12:44 | Anna | 3,200 | 1,980 | 5,193 | 2,107 | 116 | 15 | 35 | — | 31 | Anna |
| 16 | 2026-03-19T15:38 | Natasha | 5,500 | 1,989 | 7,476 | 2,101 | 85 | 47 | 31 | — | 32 | Anna |
| 17 | 2026-03-19T19:34 | Marina | 4,000 | 1,999 | 5,999 | 2,104 | 59 | 51 | 2 | — | 33 | Marina |
| 18 | 2026-03-20T07:10 | Marina | 4,000 | 2,006 | 6,058 | 2,010 | 38 | 9 | 31 | — | 34 | Anna |
| 19 | 2026-03-20T12:17 | Anna | 3,200 | 2,016 | 5,241 | 2,091 | 79 | 182 | 153 | 🌱 | 35 | Anna |
| 20 | 2026-03-20T15:43 | Anna | 3,200 | 2,040 | 5,248 | 2,176 | 120 | 29 | 5 | — | 36 | Anna |
