# Area C eval set: IRS notice classification (synthetic)

**Everything in this folder is synthetic.** The taxpayers, businesses, addresses, EINs, amounts and dates are
invented by `gen/gen_notices.py`; no real person, client or company appears. The IRS letterhead, notice numbers
and public IRS phone lines are reproduced only so the letters read like real notices. SSNs are printed masked as
`XXX-XX-NNNN` (last 4 visible), the way redacted client copies look. No real client data is used, so nothing here
falls under IRC §7216.

Regenerate at any time (reruns are byte-identical; notice *n* is seeded with `random.Random(n)`):

```
python gen/gen_notices.py
```

## Files

| File | What it is |
|---|---|
| `notice01.txt` .. `notice25.txt` | plain-text IRS-style notice letters, 120-320 words each (letterhead, notice number, tax year, notice date as "Month D, YYYY", SSN/EIN line, "What you need to do" section, amount and deadline where applicable, phone line) |
| `ground_truth.json` | one record per notice, keyed `notice01`.., with exactly the `notice_fields` from `eval/fields.json` plus an unscored `rubric_reason` string explaining each value |

Ground-truth fields and canonical formats (`eval/score.py` normalizes model output to the same forms before exact match):

| Field | Format | Notes |
|---|---|---|
| `notice_type` | enum, e.g. `CP2000` | from `fields.json` `notice_enums.notice_type` |
| `tax_year` | `2025` | the tax year the notice is about (not the notice date's year) |
| `notice_date` | `YYYY-MM-DD` | printed on the letter as "April 29, 2026" |
| `amount_due` | `4812.37` | the amount the notice asks the taxpayer to pay; `0.00` for refunds, refund holds, identity letters, CP2501 initial contact and CP259 missing return. A refund amount is never an amount due. |
| `response_deadline` | `YYYY-MM-DD` or `NONE` | see "Deadline styles" |
| `action_required` | enum | from the rubric below |
| `urgency` | enum | from the rubric below |

## Coverage

25 notices, 22 distinct notice types (CP2000 x2, CP14 x2 and LT11 x2 are repeated with the other deadline style):

CP2000, CP2501, CP14, CP501, CP503, CP504, LT11, CP49, CP12, CP11, CP22A, CP23, CP161, CP75, 4883C, 5071C,
CP3219A, CP90, CP259, CP71C, CP21B, CP05.

Individuals get a masked SSN; the two business notices (CP161 to an S corporation, CP259 to a partnership) show a
full synthetic EIN. 4883C and 5071C are written as IRS "Letters" (`Letter number LTR 4883C`, `Letter date`).

## Deadline styles

| Style | Count | How the letter states it | Ground truth |
|---|---|---|---|
| explicit | 10 | "Pay $3,401.83 by May 8, 2026." | that date |
| relative | 11 | "within 30 days from the date of this notice", "within 60 days of the date of this notice", "no later than 90 days from the date of this notice", "by the due date, which is 21 days from the notice date" | `notice_date + N` **calendar** days, computed with `datetime.timedelta` in the generator. No business-day or weekend adjustment is applied (this matches the instruction the eval prompt gives the model). |
| none | 4 | CP49, CP21B, CP71C, CP05 | `NONE` |

Relative windows used: CP503 10 days; CP14 / CP501 / CP11 / CP22A / CP23 / CP161 21 days; CP2000 / CP2501 /
CP504 / LT11 / CP75 / 4883C / 5071C / CP90 / CP259 30 days; CP12 60 days; CP3219A 90 days. Relative-style letters
never print the computed date, so the model has to do the arithmetic.

Deliberate hard cases:

* **CP05** mentions the IRS's own "within 60 days" review window but sets no deadline for the taxpayer -> `NONE`, `NO_ACTION`.
* **CP71C** has an amount due but says "pay immediately" and "doesn't set a new payment due date" -> amount due, `NONE`.
* **CP12** is a refund increase (`amount_due` `0.00`, `NO_ACTION`) whose only deadline is the 60-day window to dispute the change -> a date, not `NONE`.
* **CP2501** and **CP75** mention dollar figures (unreported income, refund being held) that are not amounts due -> `0.00`.
* **CP3219A** total is deficiency + penalty; the letter says interest is charged in addition.

## Rubric (applied in code; identical to `NOTICE_RUBRIC` in `eval/prompts.py`)

| urgency | notice types |
|---|---|
| CRITICAL | LT11, CP90, CP504, CP3219A (levy / seizure / Tax Court clock) |
| HIGH | CP2000, CP2501, CP14, CP503, CP161, CP22A, CP23, CP75 |
| MEDIUM | 4883C, 5071C, CP501, CP11, CP259, CP71C, CP05 |
| LOW | CP49, CP12, CP21B |

| action_required | notice types |
|---|---|
| PAY | CP14, CP501, CP503, CP504, CP161, CP22A, CP23, CP71C, CP11 |
| RESPOND_WITH_DOCUMENTATION | CP2000, CP2501, CP75 |
| REQUEST_HEARING | LT11, CP90 (Collection Due Process hearing, Form 12153), CP3219A (petition the Tax Court) |
| VERIFY_IDENTITY | 4883C, 5071C |
| FILE_RETURN | CP259 |
| NO_ACTION | CP49, CP12, CP21B, CP05 |

The generator asserts that every ground-truth value is in the `fields.json` enums, that each explicit deadline date
and each relative window literally appears in its letter, that no relative-style letter leaks the computed date,
that each amount due is printed, and that every letter is 120-320 words.

## Scoring

```
python eval/run_notices.py --runner <...> --name <model>      # writes results/pred_notices_<model>.json
python eval/score.py --area notices --pred results/pred_notices_<model>.json --name <model>
```

Field-level exact match after normalization; a missing field or invalid JSON counts as wrong on every field.
`rubric_reason` is documentation only and is not scored.
