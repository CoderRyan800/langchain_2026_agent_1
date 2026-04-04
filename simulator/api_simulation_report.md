# Litter Box API Simulation Report
**Simulation date:** 2026-04-04  
**Random seed:** 42  
**Cats registered:** 5  
**Visits per cat:** 10  
**Total visits:** 50  
**Injected anomalies:** 3  

---

## 1. Cat Registration
Each cat was enrolled using the **first** (lexicographically sorted) image in their `images/cats/<cat>/` folder.  The database was cleared before enrollment.

| Cat | Body Mass | Registration Image | Pool Size | Result |
|-----|----------:|--------------------:|----------:|--------|
| Anna       | 3.2 kg | `images/cats/anna/001_c25e6c.jpg` | 5 images | Registered reference image #1 for 'Anna'. Stored at: ap |
| Marina     | 4.0 kg | `images/cats/marina/001_aad380.jpg` | 5 images | Registered reference image #1 for 'Marina'. Stored at:  |
| Luna       | 5.5 kg | `images/cats/luna/001_a150db.jpg` | 6 images | Registered reference image #1 for 'Luna'. Stored at: ap |
| Natasha    | 5.0 kg | `images/cats/natasha/001_4c58dd.jpg` | 7 images | Registered reference image #1 for 'Natasha'. Stored at: |
| Whiskers   | 6.0 kg | `images/cats/whiskers/001_b0c0d4.jpg` | 3 images | Registered reference image #1 for 'Whiskers'. Stored at |

---

## 2. Sensor Model
The litter box is equipped with a weight scale and camera.

| Parameter | Value |
|-----------|-------|
| Box baseline weight | 2000 g |
| Pre-entry noise σ   | 20 g |
| Entry noise σ       | 30 g |
| Exit noise σ        | 20 g |
| Waste range         | 30–120 g |
| Ammonia normal      | 5.0–60.0 ppb |
| Ammonia anomaly     | 150.0–350.0 ppb |
| Methane normal      | 0.0–40.0 ppb |
| Methane anomaly     | 80.0–180.0 ppb |

Seeded anomalies: **Anna visit 8** (elevated CH₄), **Luna visit 3** (elevated NH₃ + CH₄), **Whiskers visit 5** (elevated NH₃).

---

## 3. Visit Log — Image Used, Cat ID, and Sensor Data
| Seq | Cat | #  | Image Used | Anom? | Identified As | Sim Score | Pre (g) | Entry (g) | Exit (g) | Waste (g) | NH₃ | CH₄ | Health |
|----:|-----|----|------------|:-----:|--------------|----------:|-------:|----------:|---------:|----------:|----:|----:|--------|
|   1 | Luna       |  1 | `001_a150db.jpg` |       | Luna         |       n/a |    2024 |      7519 |     2039 |        41 |  14.1 |  21.1 | ⚠️ |
|   2 | Luna       |  5 | `IMG_7633.jpeg` |       | Luna         |       n/a |    2005 |      7508 |     2049 |        60 |  31.7 |  21.6 | ⚠️ |
|   3 | Luna       |  8 | `IMG_7996.jpeg` |       | Unknown      |      0.87 |    2021 |      7546 |     2094 |        61 |  52.4 |   9.9 | ⚠️ |
|   4 | Luna       |  7 | `IMG_7633.jpeg` |       | Luna         |       n/a |    2024 |      7549 |     2052 |        62 |   8.0 |  26.1 | ⚠️ |
|   5 | Natasha    |  4 | `IMG_8003.jpeg` |       | Unknown      |      0.90 |    2050 |      7046 |     2129 |        73 |  55.6 |  21.2 | ⚠️ |
|   6 | Anna       |  8 | `IMG_7989.jpeg` |  ⚠️   | Anna         |       n/a |    2015 |      5221 |     2144 |       114 | 279.6 | 140.9 | ⚠️ |
|   7 | Anna       |  3 | `IMG_7991.jpeg` |       | Anna         |       n/a |    1981 |      5197 |     2059 |        55 |  13.8 |  16.9 | ⚠️ |
|   8 | Marina     |  6 | `IMG_7490.jpeg` |       | Anna         |       n/a |    1998 |      5998 |     2025 |        61 |  21.1 |   6.3 | ⚠️ |
|   9 | Natasha    |  6 | `IMG_8002.jpeg` |       | Unknown      |      0.80 |    1983 |      6984 |     2096 |        86 |  28.1 |  23.3 | ⚠️ |
|  10 | Anna       |  7 | `IMG_7989.jpeg` |       | Anna         |       n/a |    2039 |      5234 |     2131 |       108 |  50.9 |   6.5 | ⚠️ |
|  11 | Anna       |  1 | `001_c25e6c.jpg` |       | Anna         |       n/a |    2016 |      5220 |     2063 |        50 |  54.1 |   3.5 | ⚠️ |
|  12 | Natasha    |  8 | `IMG_8001.jpeg` |       | Natasha      |       n/a |    2014 |      6994 |     2082 |        37 |  30.2 |  39.9 | ⚠️ |
|  13 | Natasha    |  7 | `IMG_8002.jpeg` |       | Unknown      |      0.80 |    1994 |      7015 |     2078 |        90 |  12.2 |  25.8 | ⚠️ |
|  14 | Anna       | 10 | `IMG_7989.jpeg` |       | Anna         |       n/a |    2045 |      5274 |     2067 |        51 |   6.8 |  12.6 | ⚠️ |
|  15 | Whiskers   |  4 | `002_32dbc2.jpg` |       | Whiskers     |       n/a |    1978 |      8004 |     2025 |        42 |  58.9 |   6.5 | ⚠️ |
|  16 | Whiskers   |  2 | `001_b0c0d4.jpg` |       | Unknown      |      1.00 |    1960 |      7959 |     2003 |        55 |  16.6 |  13.7 | ⚠️ |
|  17 | Luna       |  4 | `001_a150db.jpg` |       | Luna         |       n/a |    1987 |      7469 |     2029 |        50 |   8.9 |  25.2 | ⚠️ |
|  18 | Marina     |  7 | `001_aad380.jpg` |       | Anna         |       n/a |    2039 |      6021 |     2080 |        54 |  55.2 |  34.8 | ⚠️ |
|  19 | Natasha    |  5 | `001_4c58dd.jpg` |       | Natasha      |       n/a |    2008 |      7005 |     2030 |        47 |  17.9 |   4.8 | ⚠️ |
|  20 | Marina     |  1 | `IMG_7193.jpeg` |       | Marina       |       n/a |    2040 |      6067 |     2118 |        81 |  32.5 |  35.4 | ⚠️ |
|  21 | Marina     |  2 | `IMG_7311.jpeg` |       | Marina       |       n/a |    1986 |      5996 |     2092 |        97 |  34.6 |  29.9 | ⚠️ |
|  22 | Marina     |  4 | `001_aad380.jpg` |       | Anna         |       n/a |    2000 |      6010 |     2101 |        91 |  37.8 |  15.4 | ⚠️ |
|  23 | Whiskers   |  6 | `002_32dbc2.jpg` |       | Whiskers     |       n/a |    2019 |      8004 |     2054 |        41 |  22.0 |  36.0 | ⚠️ |
|  24 | Natasha    |  1 | `IMG_8004.jpeg` |       | Marina       |       n/a |    2007 |      6992 |     2086 |        74 |   8.2 |  15.2 | ⚠️ |
|  25 | Natasha    |  2 | `IMG_8001.jpeg` |       | Natasha      |       n/a |    2020 |      7015 |     2126 |        71 |  28.3 |  38.3 | ⚠️ |
|  26 | Whiskers   |  8 | `002_b2e960.jpg` |       | Unknown      |      0.55 |    2021 |      8021 |     2026 |        51 |  13.9 |  32.0 | ⚠️ |
|  27 | Marina     |  9 | `IMG_7490.jpeg` |       | Anna         |       n/a |    2024 |      6004 |     2112 |        74 |  59.2 |  32.3 | ⚠️ |
|  28 | Anna       |  6 | `IMG_7991.jpeg` |       | Anna         |       n/a |    1988 |      5209 |     2023 |        47 |   8.8 |  26.5 | ⚠️ |
|  29 | Anna       |  9 | `IMG_7988.jpeg` |       | Anna         |       n/a |    1985 |      5180 |     2051 |        72 |  42.8 |   8.8 | ⚠️ |
|  30 | Luna       |  9 | `IMG_7633.jpeg` |       | Luna         |       n/a |    1989 |      7485 |     1996 |        47 |   9.1 |  32.3 | ⚠️ |
|  31 | Anna       |  4 | `IMG_7989.jpeg` |       | Anna         |       n/a |    1975 |      5217 |     2077 |        60 |  10.1 |   3.9 | ⚠️ |
|  32 | Whiskers   |  3 | `002_b2e960.jpg` |       | Unknown      |      0.55 |    1990 |      7972 |     2096 |        74 |  10.1 |  16.9 | ⚠️ |
|  33 | Natasha    | 10 | `IMG_7998.jpeg` |       | Natasha      |       n/a |    1983 |      6948 |     2049 |        57 |  59.1 |  32.3 | ⚠️ |
|  34 | Whiskers   |  1 | `002_b2e960.jpg` |       | Unknown      |      0.55 |    2037 |      8040 |     2070 |        57 |  12.4 |   4.6 | ⚠️ |
|  35 | Natasha    |  3 | `IMG_8004.jpeg` |       | Marina       |       n/a |    1970 |      6954 |     1999 |        44 |  36.9 |  21.7 | ⚠️ |
|  36 | Whiskers   |  7 | `001_b0c0d4.jpg` |       | Whiskers     |       n/a |    2011 |      7999 |     2091 |        52 |  53.6 |  16.3 | ⚠️ |
|  37 | Natasha    |  9 | `001_4c58dd.jpg` |       | Natasha      |       n/a |    2012 |      7021 |     2104 |        54 |  53.4 |  14.8 | ⚠️ |
|  38 | Anna       |  5 | `IMG_7989.jpeg` |       | Anna         |       n/a |    1971 |      5138 |     2015 |        96 |  25.8 |  22.1 | ⚠️ |
|  39 | Marina     |  8 | `IMG_7193.jpeg` |       | Marina       |       n/a |    2000 |      6019 |     2028 |        48 |  25.6 |   6.5 | ⚠️ |
|  40 | Luna       |  2 | `IMG_7994.jpeg` |       | Luna         |       n/a |    1959 |      7441 |     2044 |        79 |  45.1 |   8.0 | ⚠️ |
|  41 | Luna       |  6 | `IMG_7996.jpeg` |       | Unknown      |      0.87 |    1999 |      7535 |     2062 |        73 |  48.2 |  32.3 | ⚠️ |
|  42 | Whiskers   |  9 | `001_b0c0d4.jpg` |       | Unknown      |      1.00 |    2003 |      7998 |     2076 |        96 |  49.1 |  34.6 | ⚠️ |
|  43 | Anna       |  2 | `IMG_7990.jpeg` |       | Anna         |       n/a |    1970 |      5183 |     2023 |        51 |  38.1 |  22.4 | ⚠️ |
|  44 | Whiskers   |  5 | `002_32dbc2.jpg` |  ⚠️   | Unknown      |      1.00 |    1980 |      7970 |     2048 |        31 | 330.8 | 134.6 | ⚠️ |
|  45 | Whiskers   | 10 | `001_b0c0d4.jpg` |       | Whiskers     |       n/a |    2040 |      8034 |     2115 |        40 |  53.0 |  34.3 | ⚠️ |
|  46 | Marina     | 10 | `IMG_7175.jpeg` |       | Anna         |       n/a |    2012 |      6070 |     2142 |       115 |   9.7 |  19.4 | ⚠️ |
|  47 | Luna       |  3 | `IMG_7992 2.jpeg` |  ⚠️   | Luna         |       n/a |    1976 |      7503 |     2058 |        64 | 199.6 |  86.4 | ⚠️ |
|  48 | Luna       | 10 | `IMG_7994.jpeg` |       | Luna         |       n/a |    2009 |      7545 |     2105 |        79 |   5.8 |   3.7 | ⚠️ |
|  49 | Marina     |  3 | `IMG_7311.jpeg` |       | Marina       |       n/a |    2016 |      5998 |     2090 |        50 |  10.0 |   1.9 | ⚠️ |
|  50 | Marina     |  5 | `IMG_7490.jpeg` |       | Anna         |       n/a |    2025 |      6024 |     2144 |       117 |  44.6 |  27.3 | ⚠️ |

---

## 4. Identification Summary
| Cat | Visits | Identified (tentative) | Unknown | Avg Score |
|-----|-------:|-----------------------:|--------:|----------:|
| Anna       |     10 |                     10 |       0 |         — |
| Marina     |     10 |                     10 |       0 |         — |
| Luna       |     10 |                      8 |       2 |      0.87 |
| Natasha    |     10 |                      7 |       3 |      0.83 |
| Whiskers   |     10 |                      4 |       6 |      0.78 |

---

## 5. Per-Cat Visit Summaries (from API)
The following text is returned directly by `agent.get_visits_by_cat()`.

### Anna
```
Visits for 'Anna' (16 total):
  #50 [tentative] sim=0.90 | 2026-04-04T23:26:50.212869 → 2026-04-04T23:26:58.187011
  #46 [tentative] sim=0.93 | 2026-04-04T23:26:01.897517 → 2026-04-04T23:26:10.491135
  #43 [tentative] sim=0.93 | 2026-04-04T23:25:49.581961 → 2026-04-04T23:25:53.616982
  #38 [tentative] sim=0.96 | 2026-04-04T23:25:05.421171 → 2026-04-04T23:25:10.062082
  #31 [tentative] sim=0.96 | 2026-04-04T23:24:29.874714 → 2026-04-04T23:24:34.857923
  #29 [tentative] sim=1.00 | 2026-04-04T23:24:14.108619 → 2026-04-04T23:24:18.336178
  #28 [tentative] sim=0.95 | 2026-04-04T23:24:06.354707 → 2026-04-04T23:24:10.826161
  #27 [tentative] sim=0.90 | 2026-04-04T23:23:53.478601 → 2026-04-04T23:24:02.877014
  #22 [tentative] sim=0.93 | 2026-04-04T23:23:20.902642 → 2026-04-04T23:23:31.127701
  #18 [tentative] sim=0.93 | 2026-04-04T23:22:43.154132 → 2026-04-04T23:22:52.872691
  #14 [tentative] sim=0.96 | 2026-04-04T23:22:20.511296 → 2026-04-04T23:22:25.684005
  #11 [tentative] sim=1.00 | 2026-04-04T23:21:58.903184 → 2026-04-04T23:22:03.145294
  #10 [tentative] sim=0.96 | 2026-04-04T23:21:51.699530 → 2026-04-04T23:21:56.111180
  #8 [tentative] sim=0.90 | 2026-04-04T23:21:36.639642 → 2026-04-04T23:21:45.251788
  #7 [tentative] sim=0.95 | 2026-04-04T23:21:29.757164 → 2026-04-04T23:21:33.711935
  #6 [tentative] sim=0.96 | 2026-04-04T23:21:22.920996 → 2026-04-04T23:21:26.961106
```

### Marina
```
Visits for 'Marina' (6 total):
  #49 [tentative] sim=0.92 | 2026-04-04T23:26:43.406120 → 2026-04-04T23:26:47.444197
  #39 [tentative] sim=0.92 | 2026-04-04T23:25:12.981445 → 2026-04-04T23:25:17.537771
  #35 [tentative] sim=0.87 | 2026-04-04T23:24:47.210230 → 2026-04-04T23:24:51.770372
  #24 [tentative] sim=0.87 | 2026-04-04T23:23:37.687627 → 2026-04-04T23:23:42.703879
  #21 [tentative] sim=0.92 | 2026-04-04T23:23:12.797566 → 2026-04-04T23:23:17.814846
  #20 [tentative] sim=0.92 | 2026-04-04T23:23:03.876421 → 2026-04-04T23:23:09.520990
```

### Luna
```
Visits for 'Luna' (8 total):
  #48 [tentative] sim=0.82 | 2026-04-04T23:26:28.898101 → 2026-04-04T23:26:40.686808
  #47 [tentative] sim=0.84 | 2026-04-04T23:26:13.569003 → 2026-04-04T23:26:26.508685
  #40 [tentative] sim=0.82 | 2026-04-04T23:25:20.266409 → 2026-04-04T23:25:31.156781
  #30 [tentative] sim=1.00 | 2026-04-04T23:24:21.411490 → 2026-04-04T23:24:25.711694
  #17 [tentative] sim=1.00 | 2026-04-04T23:22:35.894723 → 2026-04-04T23:22:39.872482
  #4 [tentative] sim=1.00 | 2026-04-04T23:20:57.747501 → 2026-04-04T23:21:02.550050
  #2 [tentative] sim=1.00 | 2026-04-04T23:20:32.550921 → 2026-04-04T23:20:36.642753
  #1 [tentative] sim=1.00 | 2026-04-04T23:20:24.136559 → 2026-04-04T23:20:29.217863
```

### Natasha
```
Visits for 'Natasha' (5 total):
  #37 [tentative] sim=1.00 | 2026-04-04T23:24:57.952176 → 2026-04-04T23:25:02.079585
  #33 [tentative] sim=1.00 | 2026-04-04T23:24:38.230347 → 2026-04-04T23:24:42.741181
  #25 [tentative] sim=0.92 | 2026-04-04T23:23:46.287005 → 2026-04-04T23:23:50.084025
  #19 [tentative] sim=1.00 | 2026-04-04T23:22:56.202842 → 2026-04-04T23:23:00.515861
  #12 [tentative] sim=0.92 | 2026-04-04T23:22:05.504830 → 2026-04-04T23:22:09.729622
```

### Whiskers
```
Visits for 'Whiskers' (4 total):
  #45 [tentative] sim=1.00 | 2026-04-04T23:25:59.382487 → 2026-04-04T23:26:00.944314
  #36 [tentative] sim=1.00 | 2026-04-04T23:24:54.458807 → 2026-04-04T23:24:55.911153
  #23 [tentative] sim=1.00 | 2026-04-04T23:23:34.000599 → 2026-04-04T23:23:35.597038
  #15 [tentative] sim=1.00 | 2026-04-04T23:22:28.615499 → 2026-04-04T23:22:29.970767
```

---

## 6. Health Warnings
> **Disclaimer:** All health findings are preliminary outputs from an automated vision model and must be reviewed by a licensed veterinarian before any clinical action is taken.

### Visits flagged as anomalous by the system

```
No anomalous visits on record.
```

### Seeded anomaly detail
| Cat | Visit # | NH₃ (ppb) | CH₄ (ppb) | System health flag | Identified as |
|-----|--------:|----------:|----------:|-------------------|---------------|
| Anna       |       8 |     279.6 |     140.9 | ⚠️ yes | Anna |
| Whiskers   |       5 |     330.8 |     134.6 | ⚠️ yes | Unknown |
| Luna       |       3 |     199.6 |      86.4 | ⚠️ yes | Luna |

---

## 7. Unconfirmed Identities
All visits use tentative (unconfirmed) IDs until a human calls `confirm_identity()`.  Visits with `Unknown` identity need manual review.

```
50 unconfirmed visit(s):
  #50: ~Anna (sim=0.90) at 2026-04-04T23:26:50.212869
  #49: ~Marina (sim=0.92) at 2026-04-04T23:26:43.406120
  #48: ~Luna (sim=0.82) at 2026-04-04T23:26:28.898101
  #47: ~Luna (sim=0.84) at 2026-04-04T23:26:13.569003
  #46: ~Anna (sim=0.93) at 2026-04-04T23:26:01.897517
  #45: ~Whiskers (sim=1.00) at 2026-04-04T23:25:59.382487
  #44: Unknown (sim=1.00) at 2026-04-04T23:25:56.242986
  #43: ~Anna (sim=0.93) at 2026-04-04T23:25:49.581961
  #42: Unknown (sim=1.00) at 2026-04-04T23:25:46.502204
  #41: Unknown (sim=0.87) at 2026-04-04T23:25:35.453409
  #40: ~Luna (sim=0.82) at 2026-04-04T23:25:20.266409
  #39: ~Marina (sim=0.92) at 2026-04-04T23:25:12.981445
  #38: ~Anna (sim=0.96) at 2026-04-04T23:25:05.421171
  #37: ~Natasha (sim=1.00) at 2026-04-04T23:24:57.952176
  #36: ~Whiskers (sim=1.00) at 2026-04-04T23:24:54.458807
  #35: ~Marina (sim=0.87) at 2026-04-04T23:24:47.210230
  #34: Unknown (sim=0.55) at 2026-04-04T23:24:46.059887
  #33: ~Natasha (sim=1.00) at 2026-04-04T23:24:38.230347
  #32: Unknown (sim=0.55) at 2026-04-04T23:24:36.699911
  #31: ~Anna (sim=0.96) at 2026-04-04T23:24:29.874714
  #30: ~Luna (sim=1.00) at 2026-04-04T23:24:21.411490
  #29: ~Anna (sim=1.00) at 2026-04-04T23:24:14.108619
  #28: ~Anna (sim=0.95) at 2026-04-04T23:24:06.354707
  #27: ~Anna (sim=0.90) at 2026-04-04T23:23:53.478601
  #26: Unknown (sim=0.55) at 2026-04-04T23:23:52.640806
  #25: ~Natasha (sim=0.92) at 2026-04-04T23:23:46.287005
  #24: ~Marina (sim=0.87) at 2026-04-04T23:23:37.687627
  #23: ~Whiskers (sim=1.00) at 2026-04-04T23:23:34.000599
  #22: ~Anna (sim=0.93) at 2026-04-04T23:23:20.902642
  #21: ~Marina (sim=0.92) at 2026-04-04T23:23:12.797566
  #20: ~Marina (sim=0.92) at 2026-04-04T23:23:03.876421
  #19: ~Natasha (sim=1.00) at 2026-04-04T23:22:56.202842
  #18: ~Anna (sim=0.93) at 2026-04-04T23:22:43.154132
  #17: ~Luna (sim=1.00) at 2026-04-04T23:22:35.894723
  #16: Unknown (sim=1.00) at 2026-04-04T23:22:31.940306
  #15: ~Whiskers (sim=1.00) at 2026-04-04T23:22:28.615499
  #14: ~Anna (sim=0.96) at 2026-04-04T23:22:20.511296
  #13: Unknown (sim=0.80) at 2026-04-04T23:22:12.152530
  #12: ~Natasha (sim=0.92) at 2026-04-04T23:22:05.504830
  #11: ~Anna (sim=1.00) at 2026-04-04T23:21:58.903184
  #10: ~Anna (sim=0.96) at 2026-04-04T23:21:51.699530
  #9: Unknown (sim=0.80) at 2026-04-04T23:21:48.629145
  #8: ~Anna (sim=0.90) at 2026-04-04T23:21:36.639642
  #7: ~Anna (sim=0.95) at 2026-04-04T23:21:29.757164
  #6: ~Anna (sim=0.96) at 2026-04-04T23:21:22.920996
  #5: Unknown (sim=0.90) at 2026-04-04T23:21:05.134722
  #4: ~Luna (sim=1.00) at 2026-04-04T23:20:57.747501
  #3: Unknown (sim=0.87) at 2026-04-04T23:20:40.450912
  #2: ~Luna (sim=1.00) at 2026-04-04T23:20:32.550921
  #1: ~Luna (sim=1.00) at 2026-04-04T23:20:24.136559
```

---

## 8. Sensor Statistics by Cat
| Cat | Body Mass | Mean entry (g) | Expected entry (g) | Error | Mean waste (g) | Mean NH₃ | Mean CH₄ |
|-----|----------:|---------------:|-------------------:|------:|---------------:|---------:|---------:|
| Anna       | 3.2 kg |           5207 |               5200 | +7 g |           70.4 |     53.1 |     26.4 |
| Marina     | 4.0 kg |           6021 |               6000 | +21 g |           78.8 |     33.0 |     20.9 |
| Luna       | 5.5 kg |           7510 |               7500 | +10 g |           61.6 |     42.3 |     26.7 |
| Natasha    | 5.0 kg |           6997 |               7000 | -3 g |           63.3 |     33.0 |     23.7 |
| Whiskers   | 6.0 kg |           8000 |               8000 | +0 g |           53.9 |     62.0 |     33.0 |

---
_Generated by `simulator/run_api_simulation.py` using the `LitterboxAgent` Python API._
