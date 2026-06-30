# FastClose

Tools for closing the books faster and verifying tax returns add up.

## S-Corp Year-over-Year Reconciliation

`reconciliation.html` is a self-contained web tool that reconciles two consecutive years of an S-Corporation (Form 1120-S) return. Enter the key figures for the prior and current year (or load the built-in demo) and it verifies that each return is internally consistent and that the two years connect.

### Checks performed

**Internal consistency (each year)**
- Total income = gross profit (gross receipts − returns − COGS) + other income
- Ordinary business income (line 21) = total income − total deductions
- Schedule L: asset detail foots to total assets; liability & equity detail foots to total (beginning and ending)
- Schedule L balances: assets = liabilities + equity (beginning and ending)
- Schedule M-1: book income + additions − subtractions = income per return
- Schedule M-1 income per return ties to ordinary business income (review-level — separately stated items can explain a difference)
- Retained earnings roll-forward: beginning + book income − distributions = ending (review-level)
- Schedule M-2 AAA roll-forward: beginning + income + other additions − distributions − reductions = ending

**Year-over-year continuity**
- Every Schedule L line and total: prior-year ending = current-year beginning
- AAA: prior-year ending = current-year beginning

**Reasonableness**
- Flags large year-over-year swings in gross receipts, total income, deductions, ordinary income, and total assets for review

Each check reports the expected value, actual value, the difference, and a plain-language explanation. A configurable rounding tolerance (default $1) absorbs whole-dollar rounding.

### Usage

Open `reconciliation.html` in a browser — no build step or server required. Use **Load clean demo** to see a return that ties out, or **Load demo with errors** to see how discrepancies are surfaced. Returns can be saved/loaded as JSON via **Export/Import JSON**.

### Reconciliation engine

The logic lives in `js/recon-engine.js` as a pure, dependency-free module that runs both in the browser and in Node. `js/sample-data.js` holds the demo datasets.

### Tests

```
node tests/recon-engine.test.js
```

The suite verifies that a fully tying dataset passes with zero failures and that seeded errors (broken continuity, footing, and balance-sheet errors) are detected.
