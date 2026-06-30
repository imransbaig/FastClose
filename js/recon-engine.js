/*
 * FastClose — S-Corp (Form 1120-S) Year-over-Year Reconciliation Engine
 * Pure and dependency-free. Runs in the browser (global FastCloseRecon) and in
 * Node (module.exports) so the same logic powers both the UI and the tests.
 */
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();
  } else {
    root.FastCloseRecon = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var DEFAULT_TOLERANCE = 1; // dollars; Form 1120-S is filed in whole dollars

  function n(v) {
    if (v === null || v === undefined || v === '') return 0;
    var x = typeof v === 'number' ? v : parseFloat(String(v).replace(/[$,\s]/g, ''));
    return isNaN(x) ? 0 : x;
  }
  function round2(x) { return Math.round(x * 100) / 100; }

  function money(x) {
    var v = n(x);
    var neg = v < 0;
    var s = Math.abs(round2(v)).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
    return neg ? '($' + s + ')' : '$' + s;
  }
  function pctStr(p) {
    return (p >= 0 ? '+' : '') + (p * 100).toFixed(1) + '%';
  }

  function makeCheck(c) {
    var expected = n(c.expected);
    var actual = n(c.actual);
    var difference = round2(actual - expected);
    var tol = c.tolerance == null ? DEFAULT_TOLERANCE : c.tolerance;
    var withinTol = Math.abs(difference) <= tol;
    var severity = c.severity || 'error';
    var status;
    if (withinTol) status = 'pass';
    else status = (severity === 'warning' || severity === 'info') ? 'warning' : 'fail';
    return {
      id: c.id,
      category: c.category,
      title: c.title,
      scope: c.scope || '',
      expected: round2(expected),
      actual: round2(actual),
      difference: difference,
      tolerance: tol,
      status: status,
      severity: severity,
      message: typeof c.message === 'function' ? c.message(status, difference) : (c.message || '')
    };
  }

  var ASSET_FIELDS = ['cash', 'accountsReceivable', 'inventories', 'otherCurrentAssets', 'netFixedAssets', 'otherAssets'];
  var LE_FIELDS = ['accountsPayable', 'otherCurrentLiabilities', 'longTermDebt', 'otherLiabilities', 'capitalStock', 'additionalPaidInCapital', 'retainedEarnings'];

  function assetTotal(bs) {
    bs = bs || {};
    return ASSET_FIELDS.reduce(function (s, k) { return s + n(bs[k]); }, 0);
  }
  function liabEquityTotal(bs) {
    bs = bs || {};
    return LE_FIELDS.reduce(function (s, k) { return s + n(bs[k]); }, 0) - n(bs.lessTreasuryStock);
  }

  function checkYear(y, tol) {
    y = y || {};
    var checks = [];
    var label = String(y.year == null ? '' : y.year);
    var inc = y.income || {};
    var bs = y.balanceSheet || {};
    var beg = bs.beginning || {};
    var end = bs.ending || {};
    var m1 = y.scheduleM1 || {};
    var aaa = y.aaa || {};
    var distributions = n(y.distributions);

    var grossProfit = n(inc.grossReceipts) - n(inc.returnsAllowances) - n(inc.cogs);
    checks.push(makeCheck({
      id: 'income.footing', category: 'Income Statement', scope: label, tolerance: tol,
      title: 'Total income = gross profit + other income',
      expected: grossProfit + n(inc.otherIncome), actual: n(inc.totalIncome),
      message: function (st, d) {
        return st === 'pass'
          ? 'Total income ties to gross profit plus other income.'
          : 'Total income is off by ' + money(d) + ' from gross profit (' + money(grossProfit) + ') plus other income (' + money(n(inc.otherIncome)) + ').';
      }
    }));

    checks.push(makeCheck({
      id: 'income.ordinary', category: 'Income Statement', scope: label, tolerance: tol,
      title: 'Ordinary business income = total income − total deductions',
      expected: n(inc.totalIncome) - n(inc.totalDeductions), actual: n(inc.ordinaryBusinessIncome),
      message: function (st, d) {
        return st === 'pass'
          ? 'Line 21 ordinary business income ties to total income minus total deductions.'
          : 'Ordinary business income is off by ' + money(d) + '.';
      }
    }));

    [['beginning', beg], ['ending', end]].forEach(function (pair) {
      var when = pair[0], b = pair[1];
      checks.push(makeCheck({
        id: 'bs.assets.footing.' + when, category: 'Balance Sheet (Schedule L)', scope: label, tolerance: tol,
        title: 'Total assets foot (' + when + ' of year)',
        expected: assetTotal(b), actual: n(b.totalAssets),
        message: function (st, d) { return st === 'pass' ? 'Asset detail sums to total assets.' : 'Total assets do not foot; off by ' + money(d) + '.'; }
      }));
      checks.push(makeCheck({
        id: 'bs.liabeq.footing.' + when, category: 'Balance Sheet (Schedule L)', scope: label, tolerance: tol,
        title: 'Total liabilities + equity foot (' + when + ' of year)',
        expected: liabEquityTotal(b), actual: n(b.totalLiabilitiesEquity),
        message: function (st, d) { return st === 'pass' ? 'Liability & equity detail sums to total.' : 'Total liabilities & equity do not foot; off by ' + money(d) + '.'; }
      }));
      checks.push(makeCheck({
        id: 'bs.balances.' + when, category: 'Balance Sheet (Schedule L)', scope: label, tolerance: tol,
        title: 'Balance sheet balances (' + when + ' of year): assets = liabilities + equity',
        expected: n(b.totalAssets), actual: n(b.totalLiabilitiesEquity),
        message: function (st, d) { return st === 'pass' ? 'Assets equal liabilities plus equity.' : 'Balance sheet is out of balance by ' + money(d) + '.'; }
      }));
    });

    checks.push(makeCheck({
      id: 'm1.recon', category: 'Schedule M-1', scope: label, tolerance: tol,
      title: 'M-1: book income + additions − subtractions = income per return',
      expected: n(m1.netIncomePerBooks) + n(m1.additions) - n(m1.subtractions), actual: n(m1.incomePerReturn),
      message: function (st, d) { return st === 'pass' ? 'Schedule M-1 reconciles book income to the return.' : 'Schedule M-1 does not tie; off by ' + money(d) + '.'; }
    }));
    checks.push(makeCheck({
      id: 'm1.tiesToReturn', category: 'Schedule M-1', scope: label, tolerance: tol, severity: 'warning',
      title: 'M-1 income per return ties to ordinary business income',
      expected: n(inc.ordinaryBusinessIncome), actual: n(m1.incomePerReturn),
      message: function (st, d) {
        return st === 'pass'
          ? 'Income per return agrees with ordinary business income.'
          : 'Income per return differs from ordinary business income by ' + money(d) + '. This can be legitimate when separately stated items exist — review Schedule K.';
      }
    }));

    checks.push(makeCheck({
      id: 're.rollforward', category: 'Equity Roll-forward', scope: label, tolerance: tol, severity: 'warning',
      title: 'Retained earnings: beginning + book income − distributions = ending',
      expected: n(beg.retainedEarnings) + n(m1.netIncomePerBooks) - distributions, actual: n(end.retainedEarnings),
      message: function (st, d) {
        return st === 'pass'
          ? 'Retained earnings roll-forward ties.'
          : 'Retained earnings roll-forward leaves ' + money(d) + ' unexplained (could be prior-period or other equity adjustments).';
      }
    }));

    checks.push(makeCheck({
      id: 'aaa.rollforward', category: 'Schedule M-2 (AAA)', scope: label, tolerance: tol,
      title: 'AAA: beginning + income + other additions − distributions − reductions = ending',
      expected: n(aaa.beginning) + n(aaa.ordinaryIncome) + n(aaa.otherAdditions) - n(aaa.distributions) - n(aaa.otherReductions),
      actual: n(aaa.ending),
      message: function (st, d) { return st === 'pass' ? 'Accumulated Adjustments Account rolls forward correctly.' : 'AAA roll-forward is off by ' + money(d) + '.'; }
    }));

    return checks;
  }

  var BS_LINE_LABELS = {
    cash: 'Cash', accountsReceivable: 'Accounts receivable', inventories: 'Inventories',
    otherCurrentAssets: 'Other current assets', netFixedAssets: 'Net fixed assets', otherAssets: 'Other assets',
    totalAssets: 'Total assets', accountsPayable: 'Accounts payable',
    otherCurrentLiabilities: 'Other current liabilities', longTermDebt: 'Long-term debt',
    otherLiabilities: 'Other liabilities', capitalStock: 'Capital stock',
    additionalPaidInCapital: 'Additional paid-in capital', retainedEarnings: 'Retained earnings',
    lessTreasuryStock: 'Less treasury stock', totalLiabilitiesEquity: 'Total liabilities & equity'
  };

  function checkYoY(prior, current, tol) {
    prior = prior || {}; current = current || {};
    var checks = [];
    var scope = prior.year + ' → ' + current.year;
    var pe = (prior.balanceSheet || {}).ending || {};
    var cb = (current.balanceSheet || {}).beginning || {};
    Object.keys(BS_LINE_LABELS).forEach(function (k) {
      checks.push(makeCheck({
        id: 'yoy.bs.' + k, category: 'Year-over-Year Continuity', scope: scope, tolerance: tol,
        title: BS_LINE_LABELS[k] + ': prior-year ending = current-year beginning',
        expected: n(pe[k]), actual: n(cb[k]),
        message: function (st, d) {
          return st === 'pass'
            ? BS_LINE_LABELS[k] + ' carries forward correctly.'
            : BS_LINE_LABELS[k] + ' does not carry forward: prior ending ' + money(n(pe[k])) + ' vs current beginning ' + money(n(cb[k])) + ' (off ' + money(d) + ').';
        }
      }));
    });

    var pa = prior.aaa || {}, ca = current.aaa || {};
    checks.push(makeCheck({
      id: 'yoy.aaa', category: 'Year-over-Year Continuity', scope: scope, tolerance: tol,
      title: 'AAA: prior-year ending = current-year beginning',
      expected: n(pa.ending), actual: n(ca.beginning),
      message: function (st, d) { return st === 'pass' ? 'AAA carries forward correctly.' : 'AAA does not carry forward; off by ' + money(d) + '.'; }
    }));

    return checks;
  }

  function checkVariance(prior, current, threshold) {
    prior = prior || {}; current = current || {};
    threshold = threshold == null ? 0.5 : threshold;
    var scope = prior.year + ' → ' + current.year;
    var pi = prior.income || {}, ci = current.income || {};
    var pbe = (prior.balanceSheet || {}).ending || {}, cbe = (current.balanceSheet || {}).ending || {};
    var metrics = [
      ['income.grossReceipts', 'Gross receipts', n(pi.grossReceipts), n(ci.grossReceipts)],
      ['income.totalIncome', 'Total income', n(pi.totalIncome), n(ci.totalIncome)],
      ['income.totalDeductions', 'Total deductions', n(pi.totalDeductions), n(ci.totalDeductions)],
      ['income.ordinary', 'Ordinary business income', n(pi.ordinaryBusinessIncome), n(ci.ordinaryBusinessIncome)],
      ['bs.totalAssets', 'Total assets', n(pbe.totalAssets), n(cbe.totalAssets)]
    ];
    var checks = [];
    metrics.forEach(function (m) {
      var id = m[0], lbl = m[1], a = m[2], b = m[3];
      var pct = a === 0 ? (b === 0 ? 0 : 1) : (b - a) / Math.abs(a);
      var flagged = Math.abs(pct) > threshold;
      checks.push({
        id: 'var.' + id, category: 'Reasonableness', scope: scope,
        title: lbl + ' year-over-year change',
        expected: round2(a), actual: round2(b), difference: round2(b - a),
        tolerance: null,
        status: flagged ? 'warning' : 'pass',
        severity: 'info',
        message: lbl + ' changed ' + pctStr(pct) + ' (' + money(a) + ' → ' + money(b) + ')' + (flagged ? ' — large swing, confirm this is expected.' : '.')
      });
    });
    return checks;
  }

  function summarize(checks) {
    var s = { total: checks.length, passed: 0, failed: 0, warnings: 0, byCategory: {} };
    checks.forEach(function (c) {
      if (c.status === 'pass') s.passed++;
      else if (c.status === 'fail') s.failed++;
      else s.warnings++;
      var cat = s.byCategory[c.category] || (s.byCategory[c.category] = { total: 0, passed: 0, failed: 0, warnings: 0 });
      cat.total++;
      if (c.status === 'pass') cat.passed++;
      else if (c.status === 'fail') cat.failed++;
      else cat.warnings++;
    });
    return s;
  }

  function reconcile(input) {
    input = input || {};
    var opts = input.options || {};
    var tol = opts.tolerance == null ? DEFAULT_TOLERANCE : opts.tolerance;
    var threshold = opts.varianceThreshold == null ? 0.5 : opts.varianceThreshold;
    var prior = input.priorYear;
    var current = input.currentYear;
    var checks = [];

    if (prior) checks = checks.concat(checkYear(prior, tol));
    if (current) checks = checks.concat(checkYear(current, tol));
    if (prior && current) {
      checks = checks.concat(checkYoY(prior, current, tol));
      checks = checks.concat(checkVariance(prior, current, threshold));
    }

    var summary = summarize(checks);
    return {
      generatedFor: { priorYear: prior ? prior.year : null, currentYear: current ? current.year : null },
      tolerance: tol,
      varianceThreshold: threshold,
      checks: checks,
      summary: summary,
      passedAll: summary.failed === 0
    };
  }

  return {
    DEFAULT_TOLERANCE: DEFAULT_TOLERANCE,
    reconcile: reconcile,
    checkYear: checkYear,
    checkYoY: checkYoY,
    checkVariance: checkVariance,
    assetTotal: assetTotal,
    liabEquityTotal: liabEquityTotal,
    money: money,
    _n: n
  };
});
