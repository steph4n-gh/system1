"""Teach three banking support intents from public, externally labeled requests.

Run: python examples/banking_support.py
The JSON file retains BANKING77's official test split and original labels.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from system1 import ChoiceField, DecisionSchema
from _teaching_demo import run_example


class BankingSupportSchema(DecisionSchema):
    intent = ChoiceField(
        options=['card_arrival', 'lost_or_stolen_card', 'cash_withdrawal_charge'],
        descriptions={
            'card_arrival': 'Delivery and arrival of a bank card already ordered',
            'lost_or_stolen_card': 'A missing, lost, or stolen bank card',
            'cash_withdrawal_charge': 'Fees charged for withdrawing cash',
        },
    )


def main(argv=None):
    return run_example(BankingSupportSchema, 'banking_support', argv)


if __name__ == '__main__':
    main()
