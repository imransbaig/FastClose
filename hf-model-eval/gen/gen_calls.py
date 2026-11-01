#!/usr/bin/env python3
"""Generate the five synthetic CPA <-> client phone-call scripts for area B (speech-to-text + call-log summary).

    python3 gen/gen_calls.py            # writes data/calls/callNN.script.json, callNN.ref.txt, ground_truth.json
    python3 gen/gen_calls.py --check    # regenerate in memory and validate only (nothing written)

Audio is synthesized later from the script files by gen/tts_synth.py (Kokoro TTS), which is why every number,
dollar amount, date and year is written as WORDS the way people say them ("forty-five hundred dollars",
"January fifteenth, twenty twenty-six", "form ten ninety-nine", "K one", "S corp"). The reference transcript
(callNN.ref.txt) is the turn texts joined with single spaces; eval/score.py runs it and the ASR output through
the Whisper English normalizer before WER, which turns both spellings into the same digit tokens.

Synthetic identities only (invented people, businesses, funds). Deterministic: everything for call N comes from
random.Random(N), so a rerun produces byte-identical files.

Ground truth (data/calls/ground_truth.json) uses exactly the call_summary_fields from eval/fields.json, in
canonical form (name UPPERCASE, money "4500.00", date YYYY-MM-DD, enums), plus unscored extras
action_items / word_count / duration_estimate_sec. The CPA ends every call with an explicit recap that states
exactly one dollar amount, exactly one full date (month, day and year) and the CPA's own next action, so that
key_amount / key_date / follow_up_action_code are unambiguous.
"""
import argparse
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "calls"
FIELDS = json.load(open(ROOT / "eval" / "fields.json"))
GT_FIELDS = FIELDS["call_summary_fields"]
ENUMS = FIELDS["call_enums"]

WORDS_MIN, WORDS_MAX = 450, 650   # 3 - 4.5 minutes at ~2.5 words/sec
WORDS_PER_SEC = 2.5
FIRM = "Baig Tax Advisory"

# ----------------------------------------------------------------------------
# numbers / dates as spoken words (TTS input and ASR reference)
# ----------------------------------------------------------------------------
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
        10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth", 16: "sixteenth",
        17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth", 30: "thirtieth"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


def num_words(n):
    """0 <= n < 1,000,000,000 -> 'one hundred eighty-five thousand' (no 'and', hyphen inside 21-99)."""
    n = int(n)
    if n < 0:
        return "minus " + num_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 1_000_000:
        return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return num_words(n // 1_000_000) + " million" + ("" if n % 1_000_000 == 0 else " " + num_words(n % 1_000_000))


def dollars(n, casual=True):
    """4500 -> 'forty-five hundred dollars' (casual, only for 1,100-9,900 in whole hundreds), 9500 -> 'nine thousand five hundred dollars'."""
    n = int(round(n))
    if casual and 1100 <= n <= 9900 and n % 100 == 0 and (n // 100) % 10 != 0:
        return f"{num_words(n // 100)} hundred dollars"
    return f"{num_words(n)} dollars"


def ordinal(day):
    if day in ORDS:
        return ORDS[day]
    return TENS[day // 10] + "-" + ORDS[day % 10]


def year_words(y):
    if 2000 <= y <= 2009:
        return "two thousand" + ("" if y == 2000 else " " + ONES[y - 2000])
    return f"{num_words(y // 100)} {num_words(y % 100)}"   # 2026 -> 'twenty twenty-six'


def date_words(d):
    return f"{MONTHS[d.month - 1]} {ordinal(d.day)}, {year_words(d.year)}"


def canon_money(n):
    return f"{float(n):.2f}"


def pick(rnd, seq):
    return seq[rnd.randrange(len(seq))]


def round_to(x, step):
    return int(round(x / step)) * step


def floor_to(x, step):
    return int(x // step) * step


# ----------------------------------------------------------------------------
# synthetic identity pools (all invented; spoken-friendly spellings)
# ----------------------------------------------------------------------------
NAMES = {
    1: [("Karen", "Miller"), ("Brian", "Foster"), ("Dana", "Walsh"), ("Scott", "Mason")],
    2: [("David", "Brooks"), ("Amy", "Grant"), ("Kevin", "Stone"), ("Jill", "Reed")],
    3: [("Susan", "Carter"), ("Paul", "Hunter"), ("Laura", "Bell"), ("Tom", "Ward")],
    4: [("Mark", "Hayes"), ("Nancy", "Cole"), ("James", "Porter"), ("Ellen", "Lane")],
    5: [("Linda", "Porter"), ("Greg", "Turner"), ("Megan", "Hart"), ("Alan", "Fox")],
}
SCORP_BIZ = [("Hillcrest Plumbing", "the furnace and water heater calls", "the new van"),
             ("Summit Auto Repair", "the winter tire changeovers", "the new lift"),
             ("Brightwater Landscaping", "the holiday lighting jobs", "the new trailer"),
             ("Keystone Dental Lab", "the year-end crown orders", "the new scanner")]
BROKERS = ["Pinecrest Brokerage", "Castle Rock Investments", "Blue Harbor Securities", "Granite Bay Wealth"]
PARTNERSHIPS = [("Riverbend Holdings", "Lakeside Apartment Fund Two", "Mark"),
                ("Oakline Properties", "Meridian Storage Partners Three", "Dana"),
                ("Harbor Street Partners", "Copper Creek Multifamily Fund", "Chris"),
                ("Stonefield Rentals", "Northwind Industrial Fund Four", "Lisa")]
RENTAL_STREETS = ["Willow Street", "Maple Avenue", "Birch Lane", "Cedar Court"]
SOLE_PROPS = [("graphic design business", "Creative", "the new computers"),
              ("marketing consulting practice", "Consulting", "the new laptop and camera gear"),
              ("bookkeeping practice", "Bookkeeping", "the new office furniture"),
              ("web development business", "Digital", "the new workstation")]


# ----------------------------------------------------------------------------
# scenario 1: S corp owner, Q4 estimate, due January 15
# ----------------------------------------------------------------------------
def call01(rnd):
    first, last = pick(rnd, NAMES[1])
    biz, december_reason, purchase = pick(rnd, SCORP_BIZ)
    salary = rnd.choice([65000, 70000, 72000, 75000, 80000])
    net_nov = rnd.choice([160000, 170000, 185000, 195000, 210000])
    prior = rnd.choice([3600, 3800, 4000, 4200, 4400])
    q4 = prior + rnd.choice([300, 400, 500, 600])          # 3,900 - 5,000: whole hundreds -> 'forty-five hundred dollars'
    due = dt.date(2026, 1, 15)
    penalty_rate = rnd.choice([7, 8])
    turns = [
        ("CPA", f"Thanks for calling {FIRM}, how can I help you?"),
        ("CLIENT", f"Hi, this is {first} {last}, I own {biz}. I got your voicemail about the fourth-quarter estimated payment and I wanted to go over it before the holidays."),
        ("CPA", f"Perfect timing, {first}. I just finished going through your November numbers from QuickBooks, so let's walk through it. First, a quick reminder on how this works for an S corp. The company itself doesn't pay the income tax. The profit flows through to you on the K one, and you pay it on your personal return, so the estimates are personal payments, not company payments."),
        ("CLIENT", "Right, that's what we did last year."),
        ("CPA", f"Exactly. Now, your salary from the company is {dollars(salary)} this year, and you've had withholding on that through payroll, which is good. The piece that isn't covered by withholding is the pass-through profit. Through the end of November, {biz} is sitting at about {dollars(net_nov)} of net income, and December is usually a strong month for you with {december_reason}."),
        ("CLIENT", f"Yeah, December's been busy already. We also took a bigger distribution in October for {purchase}."),
        ("CPA", f"I saw that. The distribution itself isn't taxable as long as you have basis, which you do, but the profit it came out of is. So here's where I landed. You made three estimated payments this year, April, June and September, at {dollars(prior)} each. Those covered the first three quarters pretty well. For the fourth quarter, with the stronger second half, I'm recommending {dollars(q4)}."),
        ("CLIENT", "Okay. Is that just federal, or does that include the state?"),
        ("CPA", "That's federal only. Your state estimate is separate, and I'll have that figure for you next week once I see the December payroll. The federal one is the time sensitive piece."),
        ("CLIENT", "And when is that one due?"),
        ("CPA", f"{date_words(due)}. If it's late, the IRS charges an underpayment penalty, which right now is basically interest at about {num_words(penalty_rate)} percent, so it's not the end of the world, but there's no reason to pay it."),
        ("CLIENT", "Got it. How do I actually pay it? Last year I mailed a check and it took forever to show up on my account."),
        ("CPA", "I'd use IRS Direct Pay this time. It's on the IRS website, you choose estimated tax, tax year twenty twenty-five, form ten forty E S, and it pulls straight from your checking account. You get a confirmation number right away. Save that confirmation and email me a screenshot so I have it for the return."),
        ("CLIENT", "Okay, I can do that. Do I need to pay from my personal account or the business account?"),
        ("CPA", "Personal account, since it's your personal tax. If you want to move the money out of the business first, book it as a distribution to you, not as an expense. That's a common mistake and it messes up the books."),
        ("CLIENT", "Right, I remember you mentioning that."),
        ("CPA", "One more thing while I have you. Your reasonable compensation is a little on the low side for the profit level you're at this year. We don't need to fix anything for twenty twenty-five, but let's plan to bump the salary in January. I'll put that on the agenda for our planning meeting."),
        ("CLIENT", "Sounds good. Anything else you need from me?"),
        ("CPA", f"Not right now. So to recap: your fourth-quarter federal estimated payment is {dollars(q4)}, due {date_words(due)}. I'll prepare the estimate voucher with the exact figure and email it to you by the end of this week."),
        ("CLIENT", "Great, thank you. Have a good holiday."),
        ("CPA", f"You too, {first}. Talk soon."),
    ]
    gt = {"client_name": f"{first} {last}".upper(), "entity_type": "S_CORP", "tax_year_discussed": "2025",
          "topic_code": "ESTIMATED_PAYMENTS", "key_amount": canon_money(q4), "key_date": due.isoformat(),
          "follow_up_action_code": "PREPARE_ESTIMATE"}
    actions = [f"CPA: prepare the fourth-quarter federal 1040-ES voucher for ${q4:,} and email it by the end of the week",
               f"Client: pay ${q4:,} via IRS Direct Pay (tax year 2025, 1040-ES) from the personal account by {due.isoformat()} and email the confirmation screenshot",
               "CPA: send the state fourth-quarter estimate next week after December payroll",
               "Client: book any cash moved out of the business as a shareholder distribution, not an expense",
               "CPA: add a reasonable-compensation salary increase to the January planning-meeting agenda"]
    return turns, gt, actions


# ----------------------------------------------------------------------------
# scenario 2: individual, CP2000 for unreported brokerage proceeds, respond-by date
# ----------------------------------------------------------------------------
def call02(rnd):
    first, last = pick(rnd, NAMES[2])
    broker = pick(rnd, BROKERS)
    tax_year = 2024                                        # a CP2000 for a 2024 return arrives in 2026: realistic timing
    proceeds = rnd.choice([26000, 28000, 31000, 34000, 37000, 42000])
    gain = rnd.choice([3000, 3500, 4000, 4500, 5000])
    basis = proceeds - gain
    proposed = round_to(proceeds * 0.27, 100)              # tax at 22% + accuracy penalty + interest, as the IRS computes it on gross proceeds
    expect = round_to(gain * 0.15 + rnd.choice([30, 40, 50, 60]), 10)
    notice_date = dt.date(2026, 9, rnd.randint(8, 25))
    respond_by = notice_date + dt.timedelta(days=30)
    turns = [
        ("CPA", f"{FIRM}, how can I help you?"),
        ("CLIENT", f"Hi, this is {first} {last}. I got a letter from the IRS yesterday and it says I owe {dollars(proposed)}. I'm kind of panicking."),
        ("CPA", f"Okay, take a breath, {first}, these are very common. Can you read me the notice number? It should be in the top right corner."),
        ("CLIENT", f"It says C P two thousand, and the tax year is {year_words(tax_year)}."),
        ("CPA", f"Good, that's what I expected. A C P two thousand is not a bill and it's not an audit. It's an automated matching notice. The IRS computer compared the income reported on your {year_words(tax_year)} return against the forms that third parties sent them, and something didn't line up. Does it name a payer?"),
        ("CLIENT", f"Yeah, it says {broker}, and there's a number next to it, {dollars(proceeds)}."),
        ("CPA", f"That's your ten ninety-nine B, the stock sales. Here's what happened. When you sold those shares, the broker reported the gross proceeds of {dollars(proceeds)} to the IRS. The return we filed didn't include that sale, because the consolidated statement from {broker} didn't come in with the rest of your documents."),
        ("CLIENT", "I thought I uploaded everything."),
        ("CPA", f"I went back through your folder this morning. We had the interest and dividend pages from {broker}, but not the B section with the sales. It happens, especially when the broker sends a corrected statement in March. The important part is that the IRS is treating the entire {dollars(proceeds)} as profit, because they don't know what you paid for the shares. That's why the proposed tax is so high."),
        ("CLIENT", "Right, but I didn't make anywhere near that. I bought most of it years ago."),
        ("CPA", "Exactly, and that's our response. Do you remember roughly what you paid?"),
        ("CLIENT", f"The statement I found says the cost basis was around {dollars(basis)}."),
        ("CPA", f"Then your actual gain is about {dollars(gain)}, not {num_words(proceeds)}. Since you held it more than a year, that's a long-term gain taxed at fifteen percent for you. So the real tax is a few hundred dollars, plus a little interest for being late. I'm going to say expect to owe about {dollars(expect)} when it's all said and done."),
        ("CLIENT", f"That's a lot better than {num_words(proposed // 100)} hundred."),
        ("CPA", "It is. Now, the way we handle this matters. We do not file an amended return. The notice has a response form. We check the box that says we partially agree, and we attach a form eighty-nine forty-nine showing the sale with the proceeds, the cost basis and the actual gain, plus a copy of the brokerage statement. The IRS recalculates and sends you a corrected notice with the smaller amount."),
        ("CLIENT", "Okay. What do you need from me?"),
        ("CPA", f"Two things. First, log into {broker} and download the full {year_words(tax_year)} tax statement, the one with the realized gain and loss section. Second, sign the response form where I flag it. I'll send it through the portal. Please don't call the IRS yourself, and don't pay the {num_words(proposed // 100)} hundred."),
        ("CLIENT", "What's the deadline on this? The letter says thirty days."),
        ("CPA", f"Right. The notice is dated {date_words(notice_date)}, so the response is due {date_words(respond_by)}. I want it out the door at least a week before that, by certified mail, so we have proof of delivery."),
        ("CLIENT", "Okay. And if we miss it?"),
        ("CPA", "Then they issue a statutory notice of deficiency and your only option is Tax Court, which is a much bigger headache. So let's not miss it. Can you get me the statement by Friday?"),
        ("CLIENT", "Yes, I'll do it tonight."),
        ("CPA", f"Perfect. So to recap: expect to owe about {dollars(expect)}, not the amount on the notice, and the response is due {date_words(respond_by)}. I'll prepare the written response with the form eighty-nine forty-nine and send it to you to sign this week."),
        ("CLIENT", "Thank you, I feel a lot better."),
        ("CPA", f"Anytime, {first}. Talk soon."),
    ]
    gt = {"client_name": f"{first} {last}".upper(), "entity_type": "INDIVIDUAL", "tax_year_discussed": str(tax_year),
          "topic_code": "IRS_NOTICE", "key_amount": canon_money(expect), "key_date": respond_by.isoformat(),
          "follow_up_action_code": "RESPOND_TO_NOTICE"}
    actions = [f"CPA: prepare the CP2000 written response (partially agree) with Form 8949 and the {broker} statement; send for signature this week",
               f"Client: download the full {tax_year} {broker} tax statement with the realized gain/loss section by Friday",
               "Client: sign the CP2000 response form in the portal",
               f"CPA: mail the response by certified mail at least a week before {respond_by.isoformat()}",
               f"Client: do not pay the proposed ${proposed:,} and do not call the IRS; expect a corrected notice for about ${expect:,}"]
    return turns, gt, actions


# ----------------------------------------------------------------------------
# scenario 3: partnership, missing K-1 from a fund, file Form 7004 extension by March 16, 2026
# ----------------------------------------------------------------------------
def call03(rnd):
    first, last = pick(rnd, NAMES[3])
    pship, fund, partner = pick(rnd, PARTNERSHIPS)
    rental_income = rnd.choice([80000, 90000, 100000, 110000, 120000])
    set_aside = rnd.choice([8000, 8500, 9000, 9500, 10000, 11000, 12000])
    due = dt.date(2026, 3, 16)                             # March 15, 2026 is a Sunday
    turns = [
        ("CPA", f"{FIRM}, how can I help you?"),
        ("CLIENT", f"Hi, this is {first} {last}. I'm the managing member of {pship}. I'm calling about getting an extension for the partnership return, because we still don't have the K one from the fund."),
        ("CPA", f"Hi {first}. Yes, I've been watching for that one too. This is the {fund} investment, right?"),
        ("CLIENT", "Right. I emailed their investor relations twice. The last reply said their K ones won't go out until the end of April because their own accountant is behind."),
        ("CPA", f"That's pretty typical for a real estate syndication. They're a partnership themselves, so they extend, and then everybody downstream has to extend too. Here's the situation. {pship}'s own return, the form ten sixty-five, is due in March. Normally that's the fifteenth, but this year the fifteenth falls on a Sunday, so the deadline moves to Monday, {date_words(due)}."),
        ("CLIENT", "Okay."),
        ("CPA", f"Without the fund's K one, I can't finish {pship}'s return, because that income has to flow through to you and {partner} on your own K ones. So the right move is to file an extension for {pship}. That's a form seventy oh four. It's automatic, there's no reason needed, and it gives us until the middle of September to file."),
        ("CLIENT", "Does the partnership owe anything with the extension?"),
        ("CPA", f"No. {pship} is a Florida partnership, so there's no entity-level tax. The partnership itself pays nothing. The extension is just a filing, not a payment. What it does do is protect you from the late filing penalty, which for a partnership is charged per partner, per month, so it adds up fast even with two partners."),
        ("CLIENT", f"Right. And what about our personal returns? We can't file those either without {pship}'s K ones."),
        ("CPA", f"Correct, and that's the part I want you to plan for. You and {partner} will both need to extend your personal returns in April. A personal extension is also automatic, but it only extends the filing, not the payment. Any tax you owe for twenty twenty-five still has to be paid by the April deadline, or interest starts running."),
        ("CLIENT", "So how do we know what to pay if we don't have the numbers?"),
        ("CPA", f"We estimate. I have everything for {pship} except the fund. The rental side is looking like about {dollars(rental_income)} of net income this year, split between the two of you. The fund has historically thrown off a small loss on paper because of depreciation, so I'm not expecting it to add tax. Based on that, I'd have you set aside {dollars(set_aside, casual=False)} for your personal extension payment in April. I'll do a separate number for {partner}."),
        ("CLIENT", "Okay, that's manageable. Do you need anything from me to file the extension?"),
        ("CPA", f"Nothing for the federal one. I have {pship}'s tax identification number and address on file. I will need you to forward me whatever the fund sends, even if it's just an estimate letter. Some funds send a preliminary summary before the actual K one, and that helps me tighten up the numbers."),
        ("CLIENT", "I'll forward you the last email from them today. It has an estimate in it, I think."),
        ("CPA", "That would be great. And once the real K one lands, upload it to the portal and I'll finish both the partnership and the personal returns at the same time, usually within two weeks."),
        ("CLIENT", "Sounds good. Are we going to get hit with any penalty for extending?"),
        ("CPA", "No. An extension filed on time has no penalty and no interest attached to it. It's just a date change. A lot of our partnership clients extend every year for exactly this reason."),
        ("CLIENT", "Okay, that's a relief."),
        ("CPA", f"So to recap: I'm filing the extension for {pship} by Monday, {date_words(due)}, which moves the partnership return out to September. Nothing is due with it, but set aside {dollars(set_aside, casual=False)} for your personal extension payment in April. I'll file the partnership extension this week and send you the confirmation."),
        ("CLIENT", "Thanks so much."),
        ("CPA", f"You're welcome, {first}. Talk soon."),
    ]
    gt = {"client_name": f"{first} {last}".upper(), "entity_type": "PARTNERSHIP", "tax_year_discussed": "2025",
          "topic_code": "EXTENSION", "key_amount": canon_money(set_aside), "key_date": due.isoformat(),
          "follow_up_action_code": "FILE_EXTENSION"}
    actions = [f"CPA: file Form 7004 for {pship} by {due.isoformat()} and send the client the confirmation",
               f"Client: forward the latest email / estimate letter from {fund} today",
               f"Client: set aside ${set_aside:,} for the personal extension payment in April",
               f"CPA: compute a separate April set-aside figure for {partner}",
               f"Client: upload the {fund} K-1 to the portal when it arrives; CPA then finishes the 1065 and both personal returns"]
    return turns, gt, actions


# ----------------------------------------------------------------------------
# scenario 4: individual selling a rental, tax to set aside, Q4 estimate deadline, CPA schedules a meeting
# ----------------------------------------------------------------------------
def call04(rnd):
    first, last = pick(rnd, NAMES[4])
    street = pick(rnd, RENTAL_STREETS)
    sale = rnd.choice([345000, 365000, 385000, 410000, 440000])
    purchase = round_to(sale * rnd.choice([0.58, 0.62, 0.66]), 5000)
    roof = rnd.choice([9000, 11000, 12000, 14000])
    bought = rnd.choice([2013, 2014, 2015])
    depreciation = round_to(purchase * rnd.choice([0.27, 0.29, 0.31]), 1000)
    costs = round_to(sale * 0.057, 1000)
    basis = purchase + roof - depreciation
    gain = sale - costs - basis
    ltcg = gain - depreciation
    tax = depreciation * 0.25 + ltcg * 0.15 + gain * 0.038 + gain * 0.05      # recapture, LTCG, NIIT, ~5% state
    set_aside = round_to(tax * 1.03, 1000)                                       # 'a little conservative'
    closing_day = rnd.randint(14, 26)                                            # November 2025
    due = dt.date(2026, 1, 15)
    turns = [
        ("CPA", f"{FIRM}, how can I help you?"),
        ("CLIENT", f"Hi, this is {first} {last}. I'm under contract to sell my rental on {street} and I wanted to know how much I should set aside for taxes before I spend any of it."),
        ("CPA", f"Smart call, {first}, that's exactly the right question to ask before closing, not after. When is the closing?"),
        ("CLIENT", f"The {ordinal(closing_day)} of this month, so a couple of weeks out. The sale price is {dollars(sale)}."),
        ("CPA", f"Okay. And remind me, you bought that one in {year_words(bought)} for around {dollars(purchase)}?"),
        ("CLIENT", f"That's right. And I put a new roof on it a few years ago, that was about {dollars(roof)}."),
        ("CPA", f"Good, I have the roof on the depreciation schedule. So let me walk through how the gain works, because a rental is taxed differently than your house. There are three pieces. The first piece is the depreciation you've taken over the years. You've been depreciating the building and the roof since {year_words(bought)}, and the total on the schedule is about {dollars(depreciation)}. When you sell, the IRS recaptures that, and it's taxed at a maximum of twenty-five percent."),
        ("CLIENT", "Even though I never actually got that money?"),
        ("CPA", f"Right, you got it as a deduction each year, so it comes back around at the sale. The second piece is the rest of the gain. Your adjusted basis is the purchase price plus the roof, minus the depreciation, which is about {dollars(basis)}. Take the sale price, subtract selling costs, which with commissions are probably around {dollars(costs)}, and subtract the basis, and the total gain is somewhere around {dollars(gain)}."),
        ("CLIENT", "Okay."),
        ("CPA", f"The {num_words(depreciation // 1000)} thousand of depreciation is recaptured at twenty-five percent, and the remaining {dollars(ltcg)} is long-term capital gain at fifteen percent for your bracket. The third piece is the net investment income tax, three point eight percent, which kicks in because the sale pushes your income over the threshold this year. And then there's the state on top."),
        ("CLIENT", "So what's the number?"),
        ("CPA", f"Rough math, federal and state together, I'd set aside {dollars(set_aside)} from the proceeds. That's a little conservative, which is what you want. If it comes in lower when we do the return, you get it back as a refund."),
        ("CLIENT", "That's more than I was hoping, but better to know now."),
        ("CPA", f"Here's the other important part. Because the sale closes in November, that tax belongs to twenty twenty-five, and you don't want to wait until April to pay it. You should make an estimated payment for the fourth quarter by {date_words(due)}. If you don't, there's an underpayment penalty, even if you pay the full amount in April."),
        ("CLIENT", "Got it. Is there any way to avoid the tax? My brother-in-law mentioned a ten thirty-one exchange."),
        ("CPA", "A ten thirty-one exchange can defer all of it, but it has to be set up before closing, with a qualified intermediary holding the money, and you'd need to identify a replacement rental within forty-five days. With closing a couple of weeks out it's possible but tight, and only if you actually want to buy another rental. Do you?"),
        ("CLIENT", "Honestly, no. I'm done being a landlord."),
        ("CPA", "Then I wouldn't force it. Paying the tax and being done is a perfectly good answer. What I'd like to do is sit down with you the week after closing, go through the settlement statement line by line, lock in the actual gain, and get the estimate voucher ready. Can you send me the closing disclosure as soon as the title company issues it?"),
        ("CLIENT", "Yes, they said I'd have it a couple of days before closing."),
        ("CPA", f"Great. So to recap: set aside {dollars(set_aside)} from the sale proceeds for the tax, and plan to pay that as an estimated payment by {date_words(due)}. I'll schedule a meeting with you for the week after closing to go through the final numbers."),
        ("CLIENT", "Perfect, thanks for walking me through it."),
        ("CPA", f"Happy to, {first}. Congratulations on the sale."),
    ]
    gt = {"client_name": f"{first} {last}".upper(), "entity_type": "INDIVIDUAL", "tax_year_discussed": "2025",
          "topic_code": "PROPERTY_SALE", "key_amount": canon_money(set_aside), "key_date": due.isoformat(),
          "follow_up_action_code": "SCHEDULE_MEETING"}
    actions = [f"CPA: schedule a meeting for the week after the November {closing_day} closing to review the settlement statement and finalize the gain",
               f"Client: set aside ${set_aside:,} of the sale proceeds for federal and state tax",
               f"Client: pay the set-aside as a fourth-quarter 2025 estimated payment by {due.isoformat()}",
               "Client: send the closing disclosure as soon as the title company issues it",
               "CPA: prepare the fourth-quarter estimate voucher after the meeting; no 1031 exchange (client does not want a replacement rental)"]
    return turns, gt, actions


# ----------------------------------------------------------------------------
# scenario 5: sole proprietor year-end planning, SEP IRA amount and April 15 deadline, CPA sends organizer
# ----------------------------------------------------------------------------
def call05(rnd):
    first, last = pick(rnd, NAMES[5])
    biz_kind, suffix, equipment = pick(rnd, SOLE_PROPS)
    biz = f"{last} {suffix}"
    profit_nov = rnd.choice([120000, 125000, 135000, 150000, 165000])
    profit = profit_nov + 5000
    growth = rnd.choice([15, 20, 25])
    se_half = 0.9235 * profit * 0.153 / 2                    # all below the 2025 Social Security wage base
    sep_max = 0.20 * (profit - se_half)
    sep_plan = floor_to(sep_max, 500) - 500                  # 'leave a little cushion'
    savings = round_to(sep_plan * 0.29, 500)
    equipment_cost = rnd.choice([8000, 9500, 11000, 12500])
    due = dt.date(2026, 4, 15)
    turns = [
        ("CPA", f"{FIRM}, how can I help you?"),
        ("CLIENT", f"Hi, this is {first} {last}. I run the {biz_kind}, {biz}. You asked me to call before the end of the year to talk about planning, and I wanted to ask about the SEP IRA."),
        ("CPA", f"Hi {first}, yes, thanks for calling. I pulled your profit and loss through November this morning. Let's start with where the year landed and then go through the SEP and a couple of other items."),
        ("CLIENT", "Okay."),
        ("CPA", f"Through November you're at about {dollars(profit_nov)} of net profit on the Schedule C, and December usually adds a little, so I'm planning around {num_words(profit // 1000)} thousand for the year. That's up about {num_words(growth)} percent from twenty twenty-four, which is great, but it also means more self-employment tax and a higher bracket, so this is a good year to shelter some of it."),
        ("CLIENT", "That's what I figured. How much can I actually put into the SEP?"),
        ("CPA", f"The SEP limit for a sole proprietor is twenty-five percent of compensation, but for a Schedule C it works out to about twenty percent of net profit after the self-employment tax deduction. On {num_words(profit // 1000)} thousand, the maximum is right around {dollars(round_to(sep_max, 100))}. I'd leave a little cushion in case December comes in light, so plan on {dollars(sep_plan, casual=False)}."),
        ("CLIENT", "And that's deductible?"),
        ("CPA", f"Fully deductible on your ten forty. At your bracket, that contribution saves you roughly {dollars(savings)} of federal and state income tax. It doesn't reduce self-employment tax, just income tax, but it's still the best lever you have."),
        ("CLIENT", f"Do I need to do it by December thirty-first? My cash is a little tight right now because of {equipment}."),
        ("CPA", f"No, and that's the nice thing about a SEP. You can fund the twenty twenty-five contribution any time up to the filing deadline, which is {date_words(due)}. If we extend the return you'd have until October, but let's not plan on that. The account just needs to be open, and yours is already open at the brokerage."),
        ("CLIENT", "Oh good, so I can fund it in March when the big invoices come in."),
        ("CPA", "Exactly. Just make sure you tell the brokerage it's a twenty twenty-five contribution when you send it, otherwise they'll code it as twenty twenty-six by default."),
        ("CLIENT", "Right."),
        ("CPA", f"Speaking of {equipment}, how much did you spend?"),
        ("CLIENT", f"About {dollars(equipment_cost)}, in October."),
        ("CPA", "Good, that's fully deductible this year either way, so no need to do anything else before year end. If you were thinking about buying more equipment, I'd actually hold off. You don't need the deduction badly enough to spend money just for the write-off."),
        ("CLIENT", "Okay, I'll hold off then. What about the health insurance? I've been paying that personally."),
        ("CPA", "Keep paying it personally, and send me the total for the year. The self-employed health insurance deduction goes on your personal return, and that's another above-the-line deduction, so it helps the same way the SEP does."),
        ("CLIENT", "Got it. What else do you need from me for the return?"),
        ("CPA", "I'll send your tax organizer out the first week of January through the portal. It'll have a checklist: the year-end profit and loss, the ten ninety-nine forms you receive from clients, the health insurance total, and the SEP confirmation once you fund it. If you get everything back to me by the end of February, we'll file well ahead of April."),
        ("CLIENT", "That works. I should have everything by then."),
        ("CPA", f"So to recap: plan on a SEP IRA contribution of {dollars(sep_plan, casual=False)} for twenty twenty-five, funded no later than {date_words(due)}. I'll send your organizer out the first week of January so we can get started."),
        ("CLIENT", "Perfect, thank you."),
        ("CPA", f"Thanks, {first}. Have a good holiday."),
    ]
    gt = {"client_name": f"{first} {last}".upper(), "entity_type": "SOLE_PROP", "tax_year_discussed": "2025",
          "topic_code": "YEAR_END_PLANNING", "key_amount": canon_money(sep_plan), "key_date": due.isoformat(),
          "follow_up_action_code": "SEND_ORGANIZER"}
    actions = ["CPA: send the 2025 tax organizer through the portal the first week of January",
               f"Client: fund a 2025 SEP IRA contribution of ${sep_plan:,} by {due.isoformat()}, coded as a 2025 contribution",
               "Client: send the total self-employed health insurance premiums paid for the year",
               "Client: return the organizer, year-end P&L, 1099s and SEP confirmation by the end of February",
               "Client: hold off on further equipment purchases before year end"]
    return turns, gt, actions


SCENARIOS = {1: call01, 2: call02, 3: call03, 4: call04, 5: call05}


# ----------------------------------------------------------------------------
# build + validate
# ----------------------------------------------------------------------------
def build(n):
    rnd = random.Random(n)                                   # deterministic per call
    turns, gt, actions = SCENARIOS[n](rnd)
    call_id = f"call{n:02d}"
    script = {"call_id": call_id, "turns": [{"speaker": s, "text": t} for s, t in turns]}
    ref = " ".join(t for _, t in turns)
    wc = len(ref.split())
    record = {f: gt[f] for f in GT_FIELDS}                  # exact field order from fields.json
    record["action_items"] = actions
    record["word_count"] = wc
    record["duration_estimate_sec"] = round(wc / WORDS_PER_SEC, 1)
    validate(call_id, script, ref, record)
    return call_id, script, ref, record


def validate(call_id, script, ref, rec):
    problems = []
    texts = [t["text"] for t in script["turns"]]
    if len(script["turns"]) < 12:
        problems.append("too few turns")
    for t in script["turns"]:
        if t["speaker"] not in ("CPA", "CLIENT"):
            problems.append(f"bad speaker {t['speaker']!r}")
        x = t["text"]
        if re.search(r"\d", x):
            problems.append(f"digit in turn text (numbers must be words): {x[:60]!r}")
        if re.search(r"[\[\]\(\)\{\}<>*_#]", x):
            problems.append(f"stage direction / markup characters in turn: {x[:60]!r}")
        if re.search(r"[^\x00-\x7F]", x):
            problems.append(f"non-ASCII character (emoji / smart quote) in turn: {x[:60]!r}")
        if x != x.strip() or "  " in x:
            problems.append(f"stray whitespace in turn: {x[:60]!r}")
        if not x.endswith((".", "?", "!")):
            problems.append(f"turn does not end with punctuation: {x[-40:]!r}")
    if script["turns"][0]["speaker"] != "CPA" or script["turns"][1]["speaker"] != "CLIENT":
        problems.append("call must open CPA greeting then CLIENT introduction")
    wc = rec["word_count"]
    if not WORDS_MIN <= wc <= WORDS_MAX:
        problems.append(f"word count {wc} outside {WORDS_MIN}-{WORDS_MAX}")
    # ground truth: canonical formats + enums
    if rec["client_name"] != rec["client_name"].upper() or not re.fullmatch(r"[A-Z]+ [A-Z]+", rec["client_name"]):
        problems.append(f"client_name not 'FIRST LAST' uppercase: {rec['client_name']!r}")
    if rec["client_name"].title() not in texts[1]:
        problems.append("client does not state their full name in the first client turn")
    for f in ("entity_type", "topic_code", "follow_up_action_code"):
        if rec[f] not in ENUMS[f]:
            problems.append(f"{f}={rec[f]!r} not in enum")
    if not re.fullmatch(r"\d+\.\d{2}", rec["key_amount"]):
        problems.append(f"key_amount not canonical money: {rec['key_amount']!r}")
    if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", rec["key_date"]) or not re.fullmatch(r"20\d{2}", rec["tax_year_discussed"]):
        problems.append("key_date / tax_year_discussed not canonical")
    dt.date.fromisoformat(rec["key_date"])
    # recap: the last CPA turn containing 'to recap' must state exactly one amount and one full date, matching the GT
    recap = [t for t in texts if "to recap" in t.lower()]
    if len(recap) != 1:
        problems.append("exactly one 'to recap' turn expected")
    else:
        r = recap[0]
        amt_words = dollars(float(rec["key_amount"]))
        amt_words_long = dollars(float(rec["key_amount"]), casual=False)
        if amt_words not in r and amt_words_long not in r:
            problems.append(f"recap does not state the key_amount in words: {amt_words!r}")
        if len(re.findall(r"\bdollars\b", r)) != 1:
            problems.append("recap must contain exactly one dollar amount")
        if date_words(dt.date.fromisoformat(rec["key_date"])) not in r:
            problems.append("recap does not state the key_date in words")
        month_re = r"\b(" + "|".join(MONTHS) + r")\b [a-z-]+, twenty"
        if len(re.findall(month_re, r)) != 1:
            problems.append("recap must contain exactly one full month-day-year date")
    if ref != " ".join(texts):
        problems.append("ref is not the turn texts joined with single spaces")
    if problems:
        raise SystemExit(f"{call_id}: " + "; ".join(problems))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT), help="output directory (default data/calls)")
    ap.add_argument("--check", action="store_true", help="build and validate only, write nothing")
    a = ap.parse_args()
    out = Path(a.out)
    gt = {}
    built = []
    for n in sorted(SCENARIOS):
        call_id, script, ref, rec = build(n)
        gt[call_id] = rec
        built.append((call_id, script, ref))
        print(f"{call_id}: {rec['word_count']} words, ~{rec['duration_estimate_sec']:.0f} s, {len(script['turns'])} turns, "
              f"{rec['client_name']} / {rec['entity_type']} / {rec['topic_code']} / {rec['key_amount']} / {rec['key_date']} / {rec['follow_up_action_code']}")
    if a.check:
        print("check OK, nothing written")
        return
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for call_id, script, ref in built:
        p = out / f"{call_id}.script.json"
        p.write_text(json.dumps(script, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
        written.append(p)
        p = out / f"{call_id}.ref.txt"
        p.write_text(ref + "\n", encoding="utf-8")
        written.append(p)
    p = out / "ground_truth.json"
    p.write_text(json.dumps(gt, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    written.append(p)
    print("wrote", len(written), "files to", out)


if __name__ == "__main__":
    main()
