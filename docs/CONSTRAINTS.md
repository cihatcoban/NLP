# Project Constraints & Architectural Invariants (Keystone Framework)

This document establishes the non-negotiable boundaries, performance budgets, and architectural invariants for the **Turkish Word2Vec & Connotation Analysis** pipeline.

---

## 1. Hard Boundaries & Forbidden Patterns

1. **Zero Hardcoded Environment Paths**:
   - **Forbidden**: Absolute local paths like `C:\Users\username\...` or `/home/user/...`.
   - **Enforced**: All file resolution must use `pathlib.Path` relative to project root or configured via `configs/default_config.yaml` / CLI parameters.

2. **No Silent Failures**:
   - **Forbidden**: Empty `except:` blocks, silently swallowing `NaN` values without logging, or masking errors with default fallback strings.
   - **Enforced**: Custom domain exceptions derived from `NLPConnotationError` with structured context dictionaries.

3. **Memory Bounded Corpus Processing**:
   - **Forbidden**: Materializing full Wikipedia corpora or token streams into RAM (e.g., `list(LineSentence(...))`).
   - **Enforced**: Streaming generators and line-by-line iterators (`CorpusSentenceIterator`) maintaining $O(1)$ memory complexity regardless of dump size (even on 5GB+ dumps).

4. **Vectorized Lexicon Operations**:
   - **Forbidden**: `DataFrame.iterrows()` or iterative scalar assignments for dataset joining.
   - **Enforced**: Vectorized hash-joins via `pandas.merge()` executing in $<100\text{ ms}$ for 20,000+ records.

---

## 2. Domain & Scientific Invariants

1. **Turkish Casing Rules**:
   - Letter `I` must lowercase to `ı` (not standard ASCII `i`).
   - Letter `İ` must lowercase to `i`.
   - All text normalization must apply Unicode NFC standard form.

2. **Connotation Missing-Token Accounting**:
   - When evaluating top-$k$ semantic neighbors against a lexicon, tokens absent from the lexicon must not be assigned arbitrary zero or default values.
   - The metric mean must be computed strictly across tokens present in the vocabulary.
   - The record must report both `requested_k` and `found_k` (e.g., searching for top-3 where 1 word is missing yields `requested_k = 3`, `found_k = 2`).

---

## 3. Technology Stack & Quality Standards

- **Runtime**: Python $\ge 3.9$
- **Packaging**: PEP 517 / PEP 621 compliant `pyproject.toml` using `src/` layout.
- **Type Safety**: Type annotations throughout all public APIs.
- **Test Coverage**: Unit tests covering tokenization, casing, merging, lookups, and metric calculations.
