# <YYYYMMDD-short-name>

> Copy this `_template/` directory to start a new experiment:
> `cp -r experiments/_template experiments/20260701-my-idea`

## Objective

One paragraph: what question does this experiment answer, and against which
baseline (e.g. `20260624-sb3-baselines`)?

## Hypotheses

- H1: ...
- H2: ...

## Trials

One row per trial. A trial = one model/config variant; each runs across seeds,
writing artifacts to `trials/<id>/out/s<seed>/` and a row to `../ledger.jsonl`.

| trial | description | status |
|-------|-------------|--------|
| 001_baseline | starting-point CNN policy stub | todo |

## Result

Fill in after running: numbers, plots, and the takeaway vs. the baseline.
