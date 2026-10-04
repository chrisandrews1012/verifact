# verifact

A four-agent LLM pipeline that cleans CSV files, built to demonstrate how
to make a pipeline built on a non-deterministic model actually
trustworthy. Every agent's output is checked by three independent layers
before it's acted on, not just prompted and trusted. Upload a CSV and get
back a cleaned version, a full quality report, and a list of every issue
found and fixed.

## Why this exists

Any pipeline that puts an LLM between raw input and an automated action
faces the same problem: the model can be wrong, and a wrong answer that
looks confident is more dangerous than a tool that fails loudly. This
project treats that as the core design problem, not an afterthought:

- **Typed contracts** (Pydantic) enforce that every agent's output is
  structurally valid before the next stage ever sees it.
- **Invariants** deterministically re-derive facts from the actual data
  and halt the pipeline the moment an agent's claim contradicts reality.
- **Evals** run the pipeline against datasets with known-correct answers
  and score whether its judgment calls, not just its facts, are actually
  right.

CSV cleaning is the example task; the governance around it is the point.
See [`docs/architecture.md`](docs/architecture.md) for why the pipeline
is split into four single-responsibility agents and how the trust layers
fit together, [`docs/statistics-and-repairs.md`](docs/statistics-and-repairs.md)
for exactly what each statistical check and repair does, and
[`docs/evals.md`](docs/evals.md) for what a golden dataset is and why it's
the layer that catches a wrong judgment call, not just a wrong fact.

## How it works

```
CSV input
    |
    v
Profiler   ->  DataProfile
    |
    v
Validator  ->  ValidationReport
    |
    v
Repairer   ->  RepairReport + cleaned CSV
    |
    v
Reporter   ->  Markdown report
```

**Profiler** reads the dataset and infers what each column actually is:
an age, an email, a currency amount, a category, and so on. Missingness
is analyzed statistically, including a real Little's MCAR test, to tell
random gaps apart from a systematic pattern.

**Validator** looks at what the Profiler found and decides which rules
apply to each column, based purely on its inferred semantic type. An age
column gets range checks, an email column gets format checks, a
categorical column gets case-consistency checks.

**Repairer** fixes what it can, deterministically: duplicates are
dropped, currency symbols are stripped, dates are standardized, nulls
are imputed. Columns the Profiler flagged as unsafe to impute (MNAR) are
left untouched and escalated for manual review instead, since imputing
them would silently bias the data.

**Reporter** writes up everything that was found and fixed into a
structured markdown report, aimed at both technical and non-technical
readers.

## Running it

An Anthropic API key is required. Get one at
[console.anthropic.com](https://console.anthropic.com).

```bash
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

**Command line**

```bash
uv sync
make data    # generate the four synthetic sample datasets
make run     # run the pipeline on data/raw/hr_messy.csv
make serve   # start the web interface locally
```

To run on your own CSV:

```bash
uv run python -m verifact.pipeline path/to/your/data.csv
```

**Docker**

```bash
docker compose up
```

Open [http://localhost:8000](http://localhost:8000). Upload a CSV and
the pipeline runs in the browser with live progress, an inline report,
and a download link for the cleaned file.

## Testing and evals

```bash
make test-fast   # deterministic tests, no API key required
make test-llm    # LLM integration tests, requires ANTHROPIC_API_KEY
make eval        # golden-dataset evals, requires ANTHROPIC_API_KEY
```

The fast suite (102 tests as of this writing) needs no API key and
covers the deterministic logic in every stage: statistics, format
validators, cleaners, missingness detection, invariant checks, and
repair execution. The 6 LLM tests are marked `pytest.mark.llm` and
excluded from `make test-fast`; they call the live API and verify
end-to-end agent behavior.

`make eval` is a different kind of check: it runs the pipeline against
the four golden datasets and scores whether its semantic-type calls,
repair decisions, and MNAR-safety behavior are actually correct, not
just well-formed. See [`docs/evals.md`](docs/evals.md) for what that
means and why it matters. Results are written to
`evals/results/scorecard.md`.

## File structure

```
verifact/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── raw/
│       ├── hr_messy.csv
│       ├── hr_clean.csv
│       ├── ecommerce_messy.csv
│       └── medical_messy.csv
├── docs/
│   ├── architecture.md
│   ├── statistics-and-repairs.md
│   └── evals.md
├── evals/
│   ├── expected/
│   │   ├── hr_expected.json
│   │   ├── hr_clean_expected.json
│   │   ├── ecommerce_expected.json
│   │   └── medical_expected.json
│   └── runner.py
├── src/
│   └── verifact/
│       ├── agents/
│       │   ├── profiler.py
│       │   ├── validator.py
│       │   ├── repairer.py
│       │   └── reporter.py
│       ├── invariants.py
│       ├── jobs.py
│       ├── models.py
│       ├── pipeline.py
│       ├── server.py
│       └── tools.py
├── static/
│   └── index.html
├── tests/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Makefile
└── pyproject.toml
```
