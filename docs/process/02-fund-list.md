# Step 2: Choosing the hedge funds

**Date:** 2026-11-12 · **Script:** `src/find_fund_ciks.py` · **Output:** `data/fund_list.csv`

## What we did

Picked **49 hedge funds** to study, plus Berkshire Hathaway as a comparison that isn't a hedge fund, and found each one's **SEC ID (CIK)**.

## Why it's hard

13F filers include banks, pension funds, insurance companies and mutual fund firms, not just hedge funds, and no free official list says which filers are hedge funds. So we:

1. started from 57 well-known hedge fund names, chosen to mix styles (multi-strategy, quant, long/short equity, activist, macro, event-driven, "tiger cubs");
2. searched the filer names in the 13F data for each one;
3. picked the right filer for each, using clear rules.

## The rules (these go in the paper's Methods section)

1. **Use the main investment-manager filer.** Not individual fund vehicles, the "general partner" entity, insurance subsidiaries or foundations.
2. **Use the filer that actually reports holdings.** Some related entities file only "notice" reports, which say "our holdings are reported by our sister company". We found three: D. E. Shaw, Icahn and ValueAct. We confirmed every chosen ID reports real holdings.
3. **Join a fund's history across ID changes.** Appaloosa, Elliott, Schonfeld, JANA and BlueCrest changed legal entity at some point, so we list both IDs.
4. **Exclude what isn't a hedge fund managing outside money:**
   - Susquehanna is a market maker (a trading firm).
   - Soros, Duquesne and BlueCrest are family offices that manage only their own money.
   - Sands Capital is mostly long-only.
   - Man Group had no single filer before 2018.
   - Touradji stopped filing in 2014.

## Limitation to state honestly

The list is **hand-picked**, so a reviewer could call it subjective. Our defence is that the rules above are written down and applied the same way to every fund. A stronger version for Phase 2 or 3 builds the list from SEC Form ADV, the official adviser registration that says whether a manager runs hedge funds.

## Check yourself

1. Why can't we just use "every 13F filer" as our hedge fund list?
2. What is a "notice" filing, and why would using one break our study?
3. Why did we drop family offices like Soros?
4. What is the main weakness of a hand-picked fund list, and how could we fix it later?
