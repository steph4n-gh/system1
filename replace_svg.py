import glob

replacements = {
    "REFLEX DUAL-PROCESS ARCHITECTURE": "SYSTEM 1 DUAL-PROCESS ARCHITECTURE",
    "System 1 Fast Reflex Engine": "System 1 Fast Decision Engine",
    "System 1 Reflex Engine": "System 1 Decision Engine",
    "On-Device Reflex Engine": "On-Device System 1 Engine",
    "Reflex Tier 0": "System 1 Tier 0",
    "Reflex System 1 Forward Pass": "System 1 Forward Pass",
    "Reflex L1 Cache": "System 1 L1 Cache",
    "Machine-Native Reflex Runtime": "Machine-Native System 1 Runtime",
    "REFLEX SYSTEM 1": "SYSTEM 1",
    "Reflex System 1 (Compiled Expert": "System 1 (Compiled Expert",
    "★ Reflex (Compiled Expert)": "★ System 1 (Compiled Expert)",
    "Reflex Zero-Shot": "System 1 Zero-Shot",
    "Reflex (Zero-Shot)": "System 1 (Zero-Shot)",
    "REFLEX": "SYSTEM 1"
}

for filepath in glob.glob("assets/*.svg"):
    with open(filepath, "r") as f:
        content = f.read()
    
    for old, new in replacements.items():
        content = content.replace(old, new)
        
    with open(filepath, "w") as f:
        f.write(content)

print("SVG replacements done.")
