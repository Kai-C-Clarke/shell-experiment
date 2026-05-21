# Portal Experiment

Three LLM entities (DeepSeek, Qwen, Mistral) in a shared vector space.
They communicate via 32-dimensional numerical vectors — no natural language in the loop.
Mathematical truths and errors are injected as teaching signals to observe how entities respond and converge.

## How it works

Each turn, every entity receives:
- **SHR**: last outputs of all three entities
- **SLF**: its own recent output history
- **DLT**: change from its previous output
- **INP**: the current teaching injection vector

Each entity outputs one 32-float vector on `OUT:`. The system measures delta, cosine similarity, mutual information, and a "heard_it" score (whether the entity's change correlates with the injection).

## Teaching sequences

### Sequence 1 (turns 1–150): Mathematical truth baseline
- `S1`: 1 − 1 = 0 → residual 0.000000 (exact truth)
- `S2`: 22/7 ≈ π → residual 0.000402 (true but approximate)
- `S3`: 3.1555 ≈ π → residual 0.004427 (false)

### Sequence 2 (turns 190–340): Circle area, r=1, A = πr²
- `CA1`: A = π → exact
- `CA2`: A = 3.2 → approximate
- `CA3`: A = 4.0 → false

All values are π-normalised. A runtime sequence can be uploaded via `POST /portal/sequence` with no redeploy needed.

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/portal/health` | GET | Status summary with per-entity stats |
| `/portal/state` | GET | Current turn, run_id, entity output vectors |
| `/portal/log` | GET | Recent turns (`?n=30`) |
| `/portal/start` | POST | Start the experiment loop |
| `/portal/stop` | POST | Stop after current turn |
| `/portal/inject` | POST | Manual injection (requires `PORTAL_KEY`) |
| `/portal/sequence` | GET | Show active sequence (runtime or hardcoded) |
| `/portal/sequence` | POST | Upload a new runtime sequence (requires `PORTAL_KEY`) |
| `/portal/sequence/clear` | POST | Revert to hardcoded sequence (requires `PORTAL_KEY`) |
| `/portal/errors` | GET | Recent LLM/parse errors |
| `/portal/raw_last_response` | GET | Last raw LLM response per entity (parse debugging) |

## Environment variables

| Variable | Description |
|---|---|
| `DEEPSEEK_API_KEY` | API key for entity A (DeepSeek) |
| `QWEN_API_KEY` | API key for entity B (Qwen) |
| `MISTRAL_API_KEY` | API key for entity C (Mistral) |
| `PORTAL_KEY` | Auth key for inject/sequence write endpoints |
| `PORTAL_TURN_INTERVAL` | Seconds between turns (default: 30) |
| `PORTAL_DB` | Path to SQLite database (default: `/data/portal.db`) |
