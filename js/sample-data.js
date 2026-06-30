/*
 * FastClose — sample two-year S-Corp dataset for the reconciliation demo.
 * `clean` ties out perfectly; `withErrors` seeds several classic mistakes.
 * Shared by reconciliation.html and the test suite.
 */
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();
  } else {
    root.FastCloseSampleData = factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var prior = {
    year: 2023,
    income: {
      grossReceipts: 1000000, returnsAllowances: 0, cogs: 600000, otherIncome: 5000,
      totalIncome: 405000, totalDeductions: 300000, ordinaryBusinessIncome: 105000
    },
    balanceSheet: {
      beginning: {
        cash: 60000, accountsReceivable: 40000, inventories: 90000, otherCurrentAssets: 0,
        netFixedAssets: 120000, otherAssets: 0, totalAssets: 310000,
        accountsPayable: 30000, otherCurrentLiabilities: 0, longTermDebt: 90000, otherLiabilities: 0,
        capitalStock: 10000, additionalPaidInCapital: 50000, retainedEarnings: 130000,
        lessTreasuryStock: 0, totalLiabilitiesEquity: 310000
      },
      ending: {
        cash: 100000, accountsReceivable: 50000, inventories: 95000, otherCurrentAssets: 0,
        netFixedAssets: 115000, otherAssets: 0, totalAssets: 360000,
        accountsPayable: 35000, otherCurrentLiabilities: 0, longTermDebt: 80000, otherLiabilities: 0,
        capitalStock: 10000, additionalPaidInCapital: 50000, retainedEarnings: 185000,
        lessTreasuryStock: 0, totalLiabilitiesEquity: 360000
      }
    },
    scheduleM1: { netIncomePerBooks: 105000, additions: 0, subtractions: 0, incomePerReturn: 105000 },
    aaa: { beginning: 80000, ordinaryIncome: 105000, otherAdditions: 0, distributions: 50000, otherReductions: 0, ending: 135000 },
    distributions: 50000
  };

  var current = {
    year: 2024,
    income: {
      grossReceipts: 1200000, returnsAllowances: 0, cogs: 700000, otherIncome: 6000,
      totalIncome: 506000, totalDeductions: 350000, ordinaryBusinessIncome: 156000
    },
    balanceSheet: {
      beginning: {
        cash: 100000, accountsReceivable: 50000, inventories: 95000, otherCurrentAssets: 0,
        netFixedAssets: 115000, otherAssets: 0, totalAssets: 360000,
        accountsPayable: 35000, otherCurrentLiabilities: 0, longTermDebt: 80000, otherLiabilities: 0,
        capitalStock: 10000, additionalPaidInCapital: 50000, retainedEarnings: 185000,
        lessTreasuryStock: 0, totalLiabilitiesEquity: 360000
      },
      ending: {
        cash: 150000, accountsReceivable: 56000, inventories: 110000, otherCurrentAssets: 0,
        netFixedAssets: 125000, otherAssets: 0, totalAssets: 441000,
        accountsPayable: 40000, otherCurrentLiabilities: 0, longTermDebt: 70000, otherLiabilities: 0,
        capitalStock: 10000, additionalPaidInCapital: 50000, retainedEarnings: 271000,
        lessTreasuryStock: 0, totalLiabilitiesEquity: 441000
      }
    },
    scheduleM1: { netIncomePerBooks: 156000, additions: 0, subtractions: 0, incomePerReturn: 156000 },
    aaa: { beginning: 135000, ordinaryIncome: 156000, otherAdditions: 0, distributions: 70000, otherReductions: 0, ending: 221000 },
    distributions: 70000
  };

  var currentErrors = JSON.parse(JSON.stringify(current));
  currentErrors.balanceSheet.beginning.cash = 99000;          // continuity break vs prior ending (100000)
  currentErrors.aaa.beginning = 130000;                       // AAA continuity break vs prior ending (135000)
  currentErrors.income.ordinaryBusinessIncome = 150000;       // income footing break (should be 156000)
  currentErrors.balanceSheet.ending.totalAssets = 440000;     // out of balance (should be 441000)

  return {
    clean: { priorYear: prior, currentYear: current },
    withErrors: { priorYear: prior, currentYear: currentErrors }
  };
});
