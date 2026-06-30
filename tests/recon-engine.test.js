'use strict';
/* Dependency-free test runner: `node tests/recon-engine.test.js` */
var R = require('../js/recon-engine.js');
var sample = require('../js/sample-data.js');

var passed = 0, failed = 0;
function ok(cond, label) {
  if (cond) { passed++; console.log('  ✓ ' + label); }
  else { failed++; console.error('  ✗ ' + label); }
}
function find(checks, id, scope) {
  return checks.filter(function (c) { return c.id === id && (scope == null || c.scope === scope); })[0];
}

console.log('Clean dataset reconciles with no failures:');
var clean = R.reconcile(sample.clean);
ok(clean.passedAll === true, 'passedAll is true');
ok(clean.summary.failed === 0, 'zero failed checks');
ok(clean.summary.warnings === 0, 'zero review/warning checks');
ok(clean.summary.passed === clean.summary.total, 'every check passed');
ok(clean.checks.length > 30, 'a substantive number of checks ran (' + clean.checks.length + ')');

console.log('Seeded errors are detected:');
var bad = R.reconcile(sample.withErrors);
ok(bad.passedAll === false, 'passedAll is false');
ok(bad.summary.failed > 0, 'at least one failed check (' + bad.summary.failed + ')');
ok(find(bad.checks, 'yoy.bs.cash').status === 'fail', 'cash continuity break detected');
ok(find(bad.checks, 'yoy.aaa').status === 'fail', 'AAA continuity break detected');
ok(find(bad.checks, 'income.ordinary', '2024').status === 'fail', 'income footing break detected');
ok(find(bad.checks, 'bs.balances.ending', '2024').status === 'fail', 'ending balance sheet out of balance detected');

console.log('Unit: balance sheet footing helpers:');
ok(R.assetTotal(sample.clean.priorYear.balanceSheet.ending) === 360000, 'assetTotal foots');
ok(R.liabEquityTotal(sample.clean.priorYear.balanceSheet.ending) === 360000, 'liabEquityTotal foots');

console.log('Unit: per-year income checks:');
var yc = R.checkYear({
  year: 2099,
  income: { grossReceipts: 100, returnsAllowances: 0, cogs: 40, otherIncome: 10, totalIncome: 70, totalDeductions: 20, ordinaryBusinessIncome: 50 }
}, 1);
ok(find(yc, 'income.footing').status === 'pass', 'income footing passes when it ties');
ok(find(yc, 'income.ordinary').status === 'pass', 'ordinary income passes when it ties');
var ycBad = R.checkYear({
  year: 2099,
  income: { grossReceipts: 100, returnsAllowances: 0, cogs: 40, otherIncome: 10, totalIncome: 99, totalDeductions: 20, ordinaryBusinessIncome: 50 }
}, 1);
ok(find(ycBad, 'income.footing').status === 'fail', 'income footing fails when off');

console.log('Unit: tolerance is respected:');
var tInput = {
  currentYear: {
    year: 2099,
    income: { grossReceipts: 100, returnsAllowances: 0, cogs: 40, otherIncome: 10, totalIncome: 71, totalDeductions: 20, ordinaryBusinessIncome: 51 }
  }
};
ok(find(R.reconcile(Object.assign({}, tInput, { options: { tolerance: 1 } })).checks, 'income.footing').status === 'pass', '$1 diff passes at tolerance 1');
ok(find(R.reconcile(Object.assign({}, tInput, { options: { tolerance: 0 } })).checks, 'income.footing').status === 'fail', '$1 diff fails at tolerance 0');

console.log('Unit: variance flags large swings:');
var v = R.checkVariance(
  { year: 2023, income: { grossReceipts: 100 }, balanceSheet: { ending: {} } },
  { year: 2024, income: { grossReceipts: 1000 }, balanceSheet: { ending: {} } },
  0.5
);
ok(find(v, 'var.income.grossReceipts').status === 'warning', '10x revenue swing flagged for review');

console.log('Unit: money() formatting:');
ok(R.money(2500) === '$2,500', 'positive money formats with comma');
ok(R.money(-1500) === '($1,500)', 'negative money formats in parentheses');
ok(R.money(0) === '$0', 'zero formats as $0');

console.log('');
console.log(passed + ' passed, ' + failed + ' failed');
process.exit(failed === 0 ? 0 : 1);
