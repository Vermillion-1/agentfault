# Fault taxonomy, and the cells no operator can reach

Every generated fault carries three labels. Two are free, one is computed.

| Label | Source | Example |
|---|---|---|
| `provenance.operator` | cosmic-ray, natively | `core/ReplaceComparisonOperator_LtE_Lt` |
| `provenance.odc_type` / `odc_qualifier` | static map in `tools/inject.py` | `Checking` / `Incorrect` |
| `category` | derived from the operator | `off_by_one` |

We adopt **Orthogonal Defect Classification** (Chillarege et al., *IEEE TSE* 1992) rather than
inventing a scheme. Its Qualifier axis — **Missing / Incorrect / Extraneous** — turns out to be the
one that matters, for a reason that is a result rather than a formality.

## The finding: mutation cannot produce `Missing` faults

Measured over all **211 generated faults** (110 on `intervals`, 101 on `ledger`):

| ODC Type | Qualifier | Count |
|---|---|---|
| Algorithm | Incorrect | 114 |
| Checking | Incorrect | 86 |
| Assignment | Incorrect | 6 |
| Checking | **Extraneous** | 5 |
| *any* | **Missing** | **0** |

**No mutation operator produced a `Missing` fault, and structurally none can.** A mutation operator
rewrites a token that is already there. Omitted logic — a case never handled, a guard never written,
a branch that does not exist — has no token to rewrite.

This is the mechanism behind a number the mutation-testing literature has argued about for a decade.
Just et al. (*FSE 2014*) found **17% of real faults couple to no mutant at all**, with a further ~10%
needing operators nobody has written. Gay & Salahirad (*ICST 2023*), under a stricter criterion, found
only **51%** of real faults have a strongly coupled mutant. The ODC table above is what that looks like
from the generator's side: an entire column is unreachable.

**Consequence for this project.** A mutant-only corpus systematically under-samples omission bugs, so
results on it cannot be generalised to real debugging without the real-bug anchor. That is why
`corpus/external/` exists and why every headline number is reported paired. See `docs/VALIDITY.md`.

## The deliberate skew toward Logic

| Category | Count |
|---|---|
| `logic_error` | 198 |
| `off_by_one` | 13 |
| `syntax` | **0** |
| `reference` | **0** |

cosmic-ray only emits syntactically valid Python, so **100% of this corpus is Logic-class**. That is a
feature, and a deliberate divergence from prior work.

DebugBench (ACL Findings 2024) and RepoDebug (EMNLP Findings 2025) both use a four-class taxonomy
dominated by **Syntax** and **Reference** errors — categories a compiler or a linter catches without
any reasoning at all. A benchmark weighted that way can be scored respectably by a linter wrapper, and
reported aggregate numbers hide it. This corpus cannot be gamed that way because it contains no such
faults.

## Blast radius

How many tests a fault breaks, which is a proxy for how hard it is to localise. A fault breaking one
test is a far harder problem than one breaking twelve.

**53 of 211 break exactly one test**, which is the band where localisation actually discriminates,
and the distribution has a long tail out to 12.
`tools/inject.py` discards mutants above `--max-fail-frac` precisely to keep this distribution useful.

## Operator families that survived screening

| Family | Kept |
|---|---|
| ReplaceComparisonOperator | 58 |
| ReplaceBinaryOperator | 33 |
| ReplaceUnaryOperator | 6 |
| ReplaceOrWithAnd / ReplaceAndWithOr | 5 |
| ZeroIterationForLoop | 3 |
| NumberReplacer | 2 |
| AddNot | 2 |
| ReplaceTrueWithFalse | 1 |

Of 213 available operators, only these produced faults that survived the gate on this corpus —
the rest either had no applicable site, changed nothing observable, or broke too much.

## What screening discards, and why

| Repo | Screened | Kept | Broke nothing | Too broad | Collection error | Non-terminating |
|---|---|---|---|---|---|---|
| `intervals` | 146 | 110 | 23 | 13 | 0 | 0 |
| `ledger` | 104 | 101 | 3 | 0 | 0 | 0 |

The `intervals` discard rate is much higher because half-open interval arithmetic has many
boundary-equivalent rewrites — a mutant that swaps `<` for `<=` on a value that is never equal changes
nothing observable. That is the equivalent-mutant problem showing up as a measurable per-repo property
rather than as a footnote.

- **broke nothing** — equivalent mutants, or untested code. No signal either way.
- **too broad** — above `--max-fail-frac`; "did the agent find it" becomes trivially yes.
- **collection error** — a proxy for Gay & Salahirad's *wrong failure signature* finding. Their full
  criterion needs coverage data to confirm the failing tests exercise the mutated function; this
  catches only the blunt case where the mutation broke import. **Stated as a proxy, not their test.**
- **non-terminating** — mutants that hang. Turning a `break` into a `continue` inside a `while` is a
  one-token edit that makes the suite run forever. This is neither "equivalent" nor "killed"; it is a
  third outcome the usual framing does not name, and screening must bound it with a timeout or the
  corpus build never finishes.
