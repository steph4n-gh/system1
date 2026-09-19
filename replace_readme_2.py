import re

with open("README.md", "r") as f:
    text = f.read()

replacements = {
    "Reflex implements": "System 1 implements",
    "Deploy Reflex": "Deploy System 1",
    "**Reflex (L1 Cache Hit)**": "**System 1 (L1 Cache Hit)**",
    "**Reflex (Cold Forward Pass)**": "**System 1 (Cold Forward Pass)**",
    "When Reflex escalates": "When System 1 escalates",
    "Reflex Agent": "System 1 Agent",
    "Reflex is open source": "System 1 is open source",
    "reflex is open source": "System 1 is open source",
}

for old, new in replacements.items():
    text = text.replace(old, new)

with open("README.md", "w") as f:
    f.write(text)

print("README replacements 2 done.")
