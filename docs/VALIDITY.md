# Are injected faults a valid stand-in for real bugs?

**Short answer: partly, the literature disagrees about how partly, and this project is built so that
the disagreement does not have to be resolved in order to report a result.**

This document exists because the honest version of that answer is not the one usually cited.

## The three positions

Secondary sources routinely cite only the first of these.

| Work | Design | Finding |
|---|---|---|
| **Just et al., FSE 2014** — *Are Mutants a Valid Substitute for Real Faults in Software Testing?* | 357 real faults, 5 Java projects, ~321 kLoC | **73%** of real faults couple to ≥1 mutant. Mutant detection correlates with real-fault detection and **predicts it better than statement coverage**. COR, ROR and SDL couple most often. **17%** couple to no mutant at all; a further ~10% would need operators nobody has written. |
| **Papadakis et al., ICSE 2018** — *Are Mutation Scores Correlated with Real Fault Detection?* | CoreBench (C) + Defects4J (Java) | **All correlations are weak once test-suite size is controlled.** Much of the apparent relationship in prior work is a size confound. |
| **Gay & Salahirad, ICST 2023** — *How Closely are Common Mutation Operators Coupled to Real Faults?* | 32,002 mutants, 31 operators, 144 real faults, graded coupling scale | Only **9.92%** of mutants are *strongly* coupled. **51.03%** of faults have at least one strongly coupled mutant — well below Just's 73%, because the bar is stricter. Also identifies operators that disproportionately produce mutants which make the **wrong tests fail**. |

These are not flatly contradictory — Just measured *coupling* and relative predictive power against
coverage; Papadakis measured *score correlation* controlling for suite size; Gay & Salahirad applied a
stricter graded criterion. But they support **opposite practical conclusions**, and citing only 2014
would misrepresent the state of the field.

## What we take from each

**From Gay & Salahirad — an implemented filter.** Their most actionable finding is that some operators
produce mutants whose failure signature does not look like any real fault's. `tools/inject.py` discards
candidates whose failure is a pytest *collection error*, since that means the mutation broke import
rather than behaviour.

⚠️ **This is a proxy, not their criterion.** Their full test asks whether the failing tests fail *for
the same reason* a real fault would, which needs coverage data to confirm the failing tests actually
exercise the mutated function. Ours catches only the blunt case. Stated plainly so nobody cites it as
more than it is.

**From Just et al. — a prediction we then measured.** They found 17% of real faults couple to no
mutant. `docs/TAXONOMY.md` reports the generator-side view: across 110 generated faults, **zero** fall
in any ODC `Missing` cell, and structurally none can. A mutation operator rewrites a token that is
already present; omitted logic has no token to rewrite. Their uncoupled residue and our empty column
are the same phenomenon seen from two directions.

**From Papadakis et al. — a reporting rule.** Never report a mutant-only number as though it
generalises. Test-suite characteristics confound the relationship, so any claim about agent behaviour
on mutants is a claim about *this corpus with these suites* until it is paired.

## How this project stays defensible

**The corpus is paired by construction.** Mutants supply scale and category control; **BugsInPy**
supplies 493 real Python bugs as the anchor. Every headline result is reported on both, with the
correlation between them stated.

That pairing is the design move that matters. If agent behaviour tracks across the two, the mutant
corpus is doing useful work and its scale is a genuine advantage. If it does not, **that is a finding
too** — and a more interesting one than any single number, because it would be direct evidence about
where injected faults stop resembling real ones.

## What this does not claim

- Not that mutants substitute for real faults. Nobody credible claims that any more.
- Not that this corpus is representative of real debugging. It is Logic-class only, on small repos,
  with no omission bugs at all.
- Not that the wrong-failure-signature filter implements Gay & Salahirad's criterion.
- Not that the field has settled. It has not, and this document will need revising when it does.

## A note on the LLM-generated alternative

The field is pivoting toward LLM-generated mutants (LLMorpheus, *TSE 2025*; SWE-smith, NeurIPS 2025
D&B; Meta's ACH deployed at production scale) precisely because of the realism gap above. We do not
use them, for two reasons: mutation operators supply the ODC category **for free**, whereas an LLM
mutant needs classifying after the fact; and nobody has yet run a Just-2014-style coupling study on
LLM-generated mutants, so adopting them would stack an unvalidated realism claim on top of an already
contested one.
