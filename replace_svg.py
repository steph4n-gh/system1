import glob

replacements = {
    "REFLEX DUAL-PROCESS ARCHITECTURE": "SYSTEM 1 DUAL-PROCESS ARCHITECTURE",
    "System 1 Fast System 1 Engine": "System 1 Fast Decision Engine",
    "System 1 System 1 Engine": "System 1 Decision Engine",
    "On-Device System 1 Engine": "On-Device System 1 Engine",
    "System 1 Tier 0": "System 1 Tier 0",
    "System 1 System 1 Forward Pass": "System 1 Forward Pass",
    "System 1 L1 Cache": "System 1 L1 Cache",
    "Machine-Native System 1 Runtime": "Machine-Native System 1 Runtime",
    "REFLEX SYSTEM 1": "SYSTEM 1",
    "System 1 System 1 (Compiled Expert": "System 1 (Compiled Expert",
    "★ System 1 (Compiled Expert)": "★ System 1 (Compiled Expert)",
    "System 1 Zero-Shot": "System 1 Zero-Shot",
    "System 1 (Zero-Shot)": "System 1 (Zero-Shot)",
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
