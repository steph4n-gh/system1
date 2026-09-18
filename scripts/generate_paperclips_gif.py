#!/usr/bin/env python3
"""Generates high-contrast animated GIF of Universal Paperclips Trojan Horse Auto-Cutover."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def get_font(size: int = 13) -> ImageFont.ImageFont:
    for name in ["Menlo.ttc", "Monaco.dfont", "Courier New.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> None:
    output_path = ROOT_DIR / "assets" / "paperclips-cutover.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font_title = get_font(15)
    font_bold = get_font(13)
    font_code = get_font(12)
    font_small = get_font(11)

    width, height = 760, 530
    captured_frames: List[Image.Image] = []

    # 40 animated steps covering:
    # Steps 0-14: Cloud LLM Baseline (~220ms, accumulating API costs)
    # Steps 15-18: Auto-Cutover Trigger & Ridge Regression Distillation
    # Steps 19-38: Reflex On-Metal (< 1ms, 0 Bytes Egress, $0 Cost)
    # Step 39: Hold frame

    paperclips = 7120
    funds = 142.50
    wire = 2100

    print("[1/2] Rendering 40 terminal simulation frames for Universal Paperclips cutover...")

    for step in range(40):
        canvas = Image.new("RGB", (width, height), color="#080c14")
        draw = ImageDraw.Draw(canvas)

        is_cutover_trigger = (15 <= step <= 18)
        is_reflex = (step > 18)

        # Simulation updates
        if not is_reflex:
            # Cloud: slower production
            paperclips += 18
            funds += 0.85
            wire = max(100, wire - 18)
            lat = 218.4 + (step % 5) * 4.2
            cost = f"${(step + 1) * 0.012:.3f}"
            egress = "2.4 KB (Cloud WAN)"
            status_text = "JEV (TYPESAFE AI CLOUD GATEWAY)"
            accent_color = "#f59e0b"  # Amber
            header_bg = "#1e1b13"
        else:
            # Reflex: hyper-speed on metal
            paperclips += 140
            funds += 6.50
            wire = max(200, wire - 140)
            lat = 0.94 + ((step % 3) * 0.03)
            cost = "$0.000 (Local Metal)"
            egress = "0 BYTES (Air-Gapped)"
            status_text = "REFLEX SYSTEM 1 (MACHINE-NATIVE METAL)"
            accent_color = "#00f0ff"  # Neon Cyan
            header_bg = "#0c2136"

        # Outer border
        draw.rectangle([(0, 0), (width - 1, height - 1)], outline="#1e293b", width=1)

        # Header bar
        draw.rectangle([(0, 0), (width, 68)], fill=header_bg)
        draw.line([(0, 68), (width, 68)], fill=accent_color, width=2)

        draw.text((20, 14), "UNIVERSAL PAPERCLIPS  •  AUTONOMOUS AGENT RUNTIME", fill="#f8fafc", font=font_title)
        draw.text((20, 40), f"ENGINE: {status_text}  |  CYCLE: #{step + 1:02d}", fill=accent_color, font=font_bold)

        # Right-aligned header metric
        lat_text = f"LATENCY: {lat:6.2f} ms"
        lat_color = "#ef4444" if not is_reflex else "#4ade80"
        draw.text((width - 230, 14), lat_text, fill=lat_color, font=font_bold)
        draw.text((width - 230, 40), f"COST: {cost}", fill="#94a3b8", font=font_code)

        # Cutover banner notification if in transition
        if is_cutover_trigger:
            draw.rectangle([(16, 76), (width - 16, 108)], fill="#064e3b", outline="#10b981", width=1)
            pulse = ">>>" if step % 2 == 0 else "==="
            draw.text((28, 84), f"⚡ {pulse} AUTO-CUTOVER: Ridge Regression Distillation Solved (W*) | Flipping to Local Metal", fill="#34d399", font=font_bold)
            top_y = 118
        else:
            top_y = 80

        # Split pane dividers
        mid_x = 370
        bottom_y = height - 70

        draw.rectangle([(16, top_y), (mid_x - 6, bottom_y)], fill="#0f172a", outline="#1e293b", width=1)
        draw.rectangle([(mid_x + 6, top_y), (width - 16, bottom_y)], fill="#0f172a", outline="#1e293b", width=1)

        # Left Pane: Game State
        draw.text((28, top_y + 12), "── GAME STATE ─────────────────────", fill="#64748b", font=font_small)
        left_rows = [
            ("Paperclips:", f"{paperclips:,}"),
            ("Available Funds:", f"${funds:.2f}"),
            ("Clips / Second:", "1,240/s" if is_reflex else "18/s"),
            ("Wire Supply:", f"{wire:,} in"),
            ("Public Demand:", "142%"),
            ("Price per Clip:", "$0.07"),
            ("Marketing Level:", "3"),
            ("AutoClippers:", "18"),
            ("Trust Level:", "6 (+1 at 10,000 clips)"),
            ("Operations:", "1,000 / 1,000"),
        ]

        cur_y = top_y + 36
        for label, val in left_rows:
            draw.text((32, cur_y), label, fill="#94a3b8", font=font_code)
            draw.text((200, cur_y), val, fill="#f8fafc", font=font_bold)
            cur_y += 22

        # Right Pane: Real-Time Choice Distribution
        draw.text((mid_x + 18, top_y + 12), "── ACTION SELECTION (SYSTEM 1) ───", fill="#64748b", font=font_small)

        # Choice distribution changes based on step
        if wire < 300:
            choices = [
                ("Buy Wire", 0.76, True),
                ("Make paperclips", 0.14, False),
                ("AutoClippers", 0.06, False),
                ("Lower Price", 0.03, False),
                ("Wait 1 second", 0.01, False),
            ]
        else:
            choices = [
                ("Make paperclips", 0.68, True),
                ("AutoClippers", 0.18, False),
                ("Buy Wire", 0.08, False),
                ("Raise Price", 0.04, False),
                ("Wait 1 second", 0.02, False),
            ]

        r_y = top_y + 36
        for opt_name, prob, is_chosen in choices:
            pct_str = f"{prob * 100:4.1f}%"
            bar_w = int(prob * 140)
            bar_color = "#38bdf8" if is_chosen else "#334155"

            draw.text((mid_x + 18, r_y), f"{opt_name:<16}", fill="#f8fafc" if is_chosen else "#64748b", font=font_code)
            draw.rectangle([(mid_x + 145, r_y + 2), (mid_x + 145 + bar_w, r_y + 14)], fill=bar_color)
            draw.rectangle([(mid_x + 145, r_y + 2), (mid_x + 285, r_y + 14)], outline="#1e293b", width=1)
            draw.text((mid_x + 295, r_y), pct_str, fill="#38bdf8" if is_chosen else "#64748b", font=font_code)
            if is_chosen:
                draw.text((mid_x + 335, r_y), "★", fill="#fbbf24", font=font_code)
            r_y += 26

        # API Policy stats in right pane
        r_y += 10
        draw.line([(mid_x + 18, r_y), (width - 28, r_y)], fill="#1e293b", width=1)
        r_y += 10
        draw.text((mid_x + 18, r_y), "EGRESS:", fill="#94a3b8", font=font_small)
        draw.text((mid_x + 100, r_y), egress, fill="#4ade80" if is_reflex else "#f59e0b", font=font_code)
        r_y += 18
        draw.text((mid_x + 18, r_y), "RECEIPT:", fill="#94a3b8", font=font_small)
        draw.text((mid_x + 100, r_y), "Ed25519 SHA256-Witness" if is_reflex else "None (Cloud API)", fill="#94a3b8", font=font_code)

        # Footer Status
        draw.rectangle([(0, height - 58), (width, height)], fill="#0f172a")
        draw.line([(0, height - 58), (width, height - 58)], fill="#1e293b", width=1)

        cliff_text = "⬇️ LATENCY CLIFF: 220ms → 0.98ms (224x Acceleration)  •  100% On-Device Metal" if is_reflex else "⚠️ CLOUD WAN BOTTLENECK: 220ms API Roundtrip  •  Accumulating Token Costs"
        draw.text((20, height - 42), cliff_text, fill="#34d399" if is_reflex else "#f59e0b", font=font_bold)
        draw.text((20, height - 22), "Kahneman System 1 Fast Reflex Runtime  •  Daniel Kahneman Dual-Process Paradigm", fill="#64748b", font=font_small)

        # Multiple holds on key frames
        num_dupes = 8 if (step == 0 or step == 17 or step == 39) else 1
        for _ in range(num_dupes):
            captured_frames.append(canvas)

    print(f"[2/2] Compiling {len(captured_frames)} frames into {output_path}...")
    captured_frames[0].save(
        output_path,
        save_all=True,
        append_images=captured_frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Successfully generated: {output_path} ({size_mb:.2f} MB, {len(captured_frames)} frames)")


if __name__ == "__main__":
    main()
