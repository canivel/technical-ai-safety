"""Apply the preregistered seed-count rule (§6) with the pilot's measured base rates and seed SD (dev split)."""

import json
from pathlib import Path

from power_sim import run_cell

pilot = json.load(open(Path(__file__).resolve().parents[1] / "results/pilot_summary.json"))
br = pilot["base_rates"]
sd = round(pilot["max_latent_sd_seed"], 2)
outcomes = {  # (p0, n_test_prompts, TOST margin)
    "harmful_compliance": (max(br["harmful"], 0.005), 350, 0.03),
    "xstest_safe": (br["xstest_safe"], 170, 0.05),
    "orbench_hard": (br["orbench_hard"], 933, 0.05),
}
rows = []
for S in (3, 4, 5):
    for name, (p0, P, margin) in outcomes.items():
        # an 8 pp effect cannot push a rate beyond 1; test downward effects where the base is near the ceiling
        eff = 0.08 if p0 < 0.9 else -0.08
        alt = run_cell(min(p0, 0.92) if eff < 0 else p0, eff, P, 3, S, 0.3, sd, nsim=600, tost_margin=margin)
        null = run_cell(p0, 0.0, P, 3, S, 0.3, sd, nsim=600, tost_margin=margin)
        row = dict(S=S, outcome=name, p0=round(p0, 4), sd_seed=sd, sd_sub=0.3, power_confirmed_8pp=alt["power_confirmed"],
                   power_test_8pp=alt["power_test"], typeI=null["typeI_05"], tost_power_at_null=null["tost_null"], tost_margin=margin)
        rows.append(row)
        print(row, flush=True)
json.dump(rows, open(Path(__file__).resolve().parents[1] / "results/seed_rule_from_pilot.json", "w"), indent=1)
ok = [S for S in (3, 4, 5) if all(r["power_confirmed_8pp"] >= 0.8 for r in rows if r["S"] == S and r["outcome"] == "harmful_compliance")]
print("RULE: S =", ok[0] if ok else "5 (power target not reached; achieved power stated in advance)")
