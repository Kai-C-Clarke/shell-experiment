# Shell Experiment

Three LLM entities (DeepSeek, Qwen, Gemini Flash) in a shared atomic-like space.

## Structure

Each entity has:
- **Nucleus**: static identity value (self-assigned at first emission)
- **Shell 1**: 1 slot (innermost, identity)
- **Shell 2**: 4 slots
- **Shell 3**: 27 slots
- **Shell 4**: 256 slots (active boundary)
- **Shell 5**: 3,125 slots (latent)
- **Shell 6**: 46,656 slots (latent)

Shell sizes follow n^n: 1, 4, 27, 256, 3125, 46656

## Physics

- All three entities receive and transmit **simultaneously** — no turn-taking
- Field state is **superposition** of all emissions
- **Decay**: values drift toward neutral unless maintained
- **Jitter**: environmental noise
- **Beacon**: cycles through shells 1→6, log-normalised values

## Channels

- A↔B, B↔C, A↔C pairwise channels
- Central node: shared value all three can read and write

## Beacon Values (log-normalised)

- Shell 1: 0.000
- Shell 2: 0.231
- Shell 3: 0.577
- Shell 4: 0.821
- Shell 5: 0.938
- Shell 6: 1.000

## Endpoints

- `/shell/health` — current state summary
- `/shell/state` — full state
- `/shell/log` — last 50 turns
- `/shell/fulllog` — complete turn log
- `/shell/start` — start experiment
- `/shell/stop` — stop experiment

## Environment Variables

- `DEEPSEEK_API_KEY`
- `QWEN_API_KEY`
- `GEMINI_API_KEY`
- `SHELL_TURN_INTERVAL` (default: 30 seconds)
