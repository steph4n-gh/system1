import re

with open("README.md", "r") as f:
    text = f.read()

replacements = {
    "System 1 implements": "System 1 implements",
    "Deploy System 1": "Deploy System 1",
    "**System 1 (L1 Cache Hit)**": "**System 1 (L1 Cache Hit)**",
    "**System 1 (Cold Forward Pass)**": "**System 1 (Cold Forward Pass)**",
    "When System 1 escalates": "When System 1 escalates",
    "System 1 Agent": "System 1 Agent",
    "System 1 is open source": "System 1 is open source",
    "reflex is open source": "System 1 is open source",
}

for old, new in replacements.items():
    text = text.replace(old, new)

with open("README.md", "w") as f:
    f.write(text)

print("README replacements 2 done.")
