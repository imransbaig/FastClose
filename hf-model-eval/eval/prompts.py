"""Prompt templates shared by EVERY runner (baseline Claude, open VLMs, open LLMs) so comparisons are fair.
Import: from prompts import doc_prompt, notice_prompt, call_summary_prompt
"""
import json
from pathlib import Path

FIELDS = json.load(open(Path(__file__).resolve().parent / "fields.json"))

DOC_LABELS = {
    "W2": "Form W-2 Wage and Tax Statement", "1099NEC": "Form 1099-NEC", "1099INT": "Form 1099-INT",
    "1099DIV": "Form 1099-DIV", "1099R": "Form 1099-R", "1099B_SUMMARY": "consolidated Form 1099-B summary page",
    "K1_1065": "Schedule K-1 (Form 1065)", "K1_1120S": "Schedule K-1 (Form 1120-S)",
    "IRS_WAGE_INCOME_TRANSCRIPT": "IRS Wage and Income Transcript", "IRS_ACCOUNT_TRANSCRIPT": "IRS Account Transcript",
    "CP2000": "IRS Notice CP2000", "CP14": "IRS Notice CP14", "CP503": "IRS Notice CP503",
}
FIELD_HINTS = {
    "employee_ssn_last4": "the 4 visible digits of the masked SSN (XXX-XX-1234 -> 1234)",
    "recipient_tin_last4": "the 4 visible digits of the masked recipient TIN",
    "partner_tin_last4": "the 4 visible digits of the masked partner TIN",
    "shareholder_tin_last4": "the 4 visible digits of the masked shareholder TIN",
    "taxpayer_ssn_last4": "the 4 visible digits of the masked SSN",
    "box12a_code": "the letter code in box 12a (e.g. D, DD, W)", "box12a_amount": "the amount next to box 12a",
    "box15_state": "two-letter state abbreviation from box 15",
    "box7_distribution_code": "the distribution code in box 7 (e.g. 7, 1, G)",
    "ira_sep_simple_checkbox": "true if the IRA/SEP/SIMPLE box is checked, else false",
    "partner_type": "GENERAL or LIMITED", "profit_share_pct_ending": "ending profit share percent as a number (e.g. 25.00)",
    "ownership_pct": "shareholder ownership percent as a number (e.g. 50.00)",
    "box14a_self_employment_earnings": "box 14 code A amount", "box19a_distributions": "box 19 code A amount",
    "box16d_distributions": "box 16 code D amount", "box4a_guaranteed_payments_services": "box 4a amount",
    "form_count": "how many information forms (W-2, 1099 etc.) are listed on the transcript, as an integer",
    "w2_1_employer_name": "employer name on the FIRST W-2 listed", "w2_1_wages": "wages on the first W-2", "w2_1_fed_withheld": "federal withholding on the first W-2",
    "int_1_payer_name": "payer on the first 1099-INT", "int_1_interest": "interest on the first 1099-INT",
    "nec_1_payer_name": "payer on the first 1099-NEC", "nec_1_amount": "non-employee compensation on the first 1099-NEC",
    "filing_status": "SINGLE, MARRIED FILING JOINT, MARRIED FILING SEPARATE or HEAD OF HOUSEHOLD",
    "tc150_date": "date of transaction code 150", "tc806_withholding": "amount of transaction code 806 as a positive number",
    "tc846_refund_amount": "amount of transaction code 846 as a positive number", "tc846_date": "date of transaction code 846",
    "notice_type": "the notice number printed on the notice, e.g. CP2000", "response_deadline": "the respond-by date",
    "total_proposed_amount_due": "the proposed amount due (tax + penalty + interest)",
    "unreported_payer_name": "the payer whose income the IRS says was not reported", "unreported_amount": "the difference amount for that payer",
    "caller_id": "the Caller ID number printed on the notice", "wash_sale_loss_disallowed": "total wash sale loss disallowed (short-term + long-term)",
}
RULES = ("Rules: money as a plain number string with 2 decimals and no $ or commas (e.g. \"84210.55\"); a value printed in "
         "parentheses is negative (\"-12480.00\"); a blank or zero box is \"0.00\"; dates as YYYY-MM-DD; tax_year as a 4-digit "
         "string; names in UPPERCASE exactly as printed; EIN/TIN as NN-NNNNNNN; booleans as true/false; codes uppercase. "
         "Return ONLY the JSON object, no markdown, no commentary.")


def doc_prompt(doc_type):
    fields = FIELDS["doc_types"][doc_type]
    lines = [f'  "{f}": {FIELD_HINTS.get(f, f.replace("_", " "))}' for f in fields]
    return (f"This image is a scanned {DOC_LABELS[doc_type]} for tax year shown on the form. Extract these fields and return a JSON "
            f"object with exactly these {len(fields)} keys:\n" + "\n".join(lines) + "\n" + RULES)


NOTICE_RUBRIC = (
    "Urgency rubric by notice type: CRITICAL = LT11, CP90, CP504, CP3219A (levy / seizure / Tax Court deadline); "
    "HIGH = CP2000, CP2501, CP14, CP503, CP161, CP22A, CP23, CP75; MEDIUM = 4883C, 5071C, CP501, CP11, CP259, CP71C, CP05; "
    "LOW = CP49, CP12, CP21B.\n"
    "Action rubric: PAY = CP14, CP501, CP503, CP504, CP161, CP22A, CP23, CP71C, CP11; RESPOND_WITH_DOCUMENTATION = CP2000, CP2501, CP75; "
    "REQUEST_HEARING = LT11, CP90, CP3219A; VERIFY_IDENTITY = 4883C, 5071C; FILE_RETURN = CP259; NO_ACTION = CP49, CP12, CP21B, CP05."
)


def notice_prompt(notice_text):
    e = FIELDS["notice_enums"]
    return (
        "Classify the following IRS notice for a CPA firm's intake queue. Return ONLY a JSON object with exactly these keys:\n"
        f'  "notice_type": one of {e["notice_type"]}\n'
        '  "tax_year": the tax year the notice is about, 4-digit string\n'
        '  "notice_date": the notice date as YYYY-MM-DD\n'
        '  "amount_due": amount due as a plain number string with 2 decimals, "0.00" if the notice asks for no payment\n'
        '  "response_deadline": the respond-by / pay-by date as YYYY-MM-DD. If the notice gives a number of days (e.g. "within 30 days from the date of this notice"), add that many calendar days to the notice date. "NONE" if the notice sets no deadline.\n'
        f'  "action_required": one of {e["action_required"]}\n'
        f'  "urgency": one of {e["urgency"]}\n'
        + NOTICE_RUBRIC + "\nNo markdown, no commentary, JSON only.\n\n--- NOTICE ---\n" + notice_text + "\n--- END NOTICE ---"
    )


def call_summary_prompt(transcript):
    e = FIELDS["call_enums"]
    return (
        "Below is a transcript of a phone call between a CPA and a client. Fill the firm's call-log fields. Return ONLY a JSON object with exactly these keys:\n"
        '  "client_name": the client\'s full name in UPPERCASE as they state it\n'
        f'  "entity_type": one of {e["entity_type"]}\n'
        '  "tax_year_discussed": 4-digit string\n'
        f'  "topic_code": one of {e["topic_code"]}\n'
        '  "key_amount": the dollar amount the CPA tells the client to pay or set aside in the closing recap, as a plain number string with 2 decimals\n'
        '  "key_date": the deadline date the CPA states in the closing recap, as YYYY-MM-DD\n'
        f'  "follow_up_action_code": the CPA\'s own next action stated in the recap, one of {e["follow_up_action_code"]}\n'
        "No markdown, no commentary, JSON only.\n\n--- TRANSCRIPT ---\n" + transcript + "\n--- END TRANSCRIPT ---"
    )


def parse_json_object(text):
    """Robustly pull the first JSON object out of a model response. Returns (obj or None, valid: bool)."""
    import re
    if text is None:
        return None, False
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        o = json.loads(t)
        return (o, True) if isinstance(o, dict) else (None, False)
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            return (o, isinstance(o, dict)) if isinstance(o, dict) else (None, False)
        except Exception:
            return None, False
    return None, False


if __name__ == "__main__":
    print(doc_prompt("W2"))
    print()
    print(notice_prompt("<notice text>")[:600])
