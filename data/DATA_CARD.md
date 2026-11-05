# Data card — synthetic card-transaction stream

**Generator:** `src/data_generation.py` (`generate_transactions(seed=42)`)
**Regenerate:** `python -m src.pipeline` writes `data/transactions.csv`. That
file is git-ignored, about 20 MB, and identical bit-for-bit for the same seed
and library versions. `data/sample_transactions.csv` is a committed
1,000-row sample.

## Size (seed 42)

| | value |
|---|---|
| customers | 3,000 |
| period | 90 days (2025-01-01 → 2025-03-31) |
| transactions | 234,291 |
| fraudulent | 2,343 (1.000%) |

## Columns (raw)

| column | type | description |
|---|---|---|
| `transaction_id` | int | row id in time order |
| `customer_id` | int | card holder (0–2999) |
| `timestamp` | datetime | authorisation time |
| `t_seconds` | float | seconds since 2025-01-01 00:00 |
| `amount` | float | USD, ≥ 0.50 |
| `merchant_category` | str | one of 10 categories (below) |
| `is_online` | 0/1 | card-not-present (online) vs. card-present |
| `distance_km` | float | geolocation distance from the card holder's home (IP geolocation for online purchases) |
| `is_fraud` | 0/1 | label |
| `fraud_scenario` | str | `none`, `cnp_spree`, `card_testing`, `stolen_card`, `mimic`. **For analysis only. Never a model input.** |

## Generative model

### Customers
For each customer *c*:
- spend multiplier `m_c ~ LogNormal(0, 0.5)`
- category preferences `π_c ~ Dirichlet(12 · popularity)`
- daily rate `λ_c ~ LogNormal(log 0.75, 0.5)`
- chronotype shift `h_c ~ N(0, 1.5)` hours
- online propensity `o_c ~ LogNormal(0, 0.4)`
- home-noise radius `r_c ~ LogNormal(log 5, 0.4)` km
- with probability 0.15, one trip lasting 3–10 days at 200–3,000 km

### Legitimate transactions
`N_c ~ Poisson(90 λ_c)`. Each transaction is generated as follows:
- day ~ U{0..89}
- hour: with probability 0.88, `N(14 + h_c, 3.5)`; otherwise U(0, 24)
- category ~ `π_c`
- amount = `median_cat · m_c · LogNormal(0, 0.6)`
- online ~ Bernoulli(`p_online_cat · o_c`)
- distance: card-present `LogNormal(log r_c, 0.8)`; online `LogNormal(log 15, 1.1)`; card-present during a trip, about the trip distance

| category | popularity | median $ | P(online) |
|---|---|---|---|
| grocery | 0.22 | 45 | 0.04 |
| restaurant | 0.17 | 30 | 0.06 |
| gas_transport | 0.14 | 40 | 0.03 |
| retail_apparel | 0.09 | 60 | 0.25 |
| online_retail | 0.12 | 50 | 0.95 |
| electronics | 0.03 | 150 | 0.40 |
| travel | 0.03 | 300 | 0.70 |
| entertainment | 0.08 | 35 | 0.30 |
| health | 0.08 | 40 | 0.05 |
| gift_cards_digital | 0.04 | 40 | 0.80 |

### Fraud
Compromise events hit a uniformly random customer on a random day. The
start hour is night (00:00–05:59) with probability 0.55; otherwise it is
uniform over the day. Each event emits a burst:

| scenario | P | burst size | gap | categories | amount | channel / distance |
|---|---|---|---|---|---|---|
| `cnp_spree` | 0.45 | 1+Poisson(2.5) | Exp(15 min) | 75% electronics / gift cards / online retail / travel | `median·m_c·LogNormal(log 3, 0.6)` | 90% online; 65% far (`LogNormal(log 800, 1.0)`) |
| `card_testing` | 0.20 | 3+Poisson(3) | Exp(3 min) | gift cards / online retail / entertainment | U($0.5, $5); 40% end with one large "cash-out" | online, `LogNormal(log 600, 1.2)` |
| `stolen_card` | 0.20 | 1+Poisson(1.5) | Exp(45 min) | 70% electronics / apparel / gas / grocery | `median·m_c·LogNormal(log 2.2, 0.6)` | card-present, `LogNormal(log 120, 0.9)` |
| `mimic` | 0.15 | 1+Poisson(0.8) | Exp(6 h) | the victim's own `π_c` | `median·m_c·LogNormal(log 1.3, 0.6)` | the victim's own channel mix, near home, daytime hours |

Events are added (and the last one is truncated) until fraud makes up
exactly `0.01 · N / (1 − 0.01)` of legitimate rows, which puts the fraud
share at 1.000%. Scenario shares in *rows* differ from the event shares
above, because card-testing bursts are long and mimic bursts are short.

### Known simplifications
- No merchant identities, devices or chargeback delay; labels are clean and
  available immediately.
- Stationary behaviour: no concept drift, except that the history available
  for behavioural features grows over time.
- Fraud scenarios are generated independently; there are no adaptive adversaries.
