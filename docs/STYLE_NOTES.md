# Style notes

How OllamaPower is written, learned from the owner's two existing projects:
[OllamaGate](https://github.com/111yoav111/OllamaGate) (the sister project,
which wins every *style* question) and
[DCP](https://github.com/111yoav111/Dynamic-Compute-Power) (the code we
inherit). On *behaviour or architecture* the OllamaPower spec wins. If a case
isn't covered here, copy what OllamaGate does.

## Project layout

- One installable package, `ollamapower/`, with a `pyproject.toml`
  (setuptools, `requires-python = ">=3.11"`, MIT) and a console script
  `ollamapower = "ollamapower.__main__:main"`. No `src/` folder and no
  `sys.path` tricks (DCP had both; they go away).
- Sub-packages by responsibility, short lower-case module names:
  `admin/`, `server/`, `tui/` in OllamaGate; here `core/`, `scheduling/`,
  `hub/`, `agent/`, `llamacpp/`, `ollama/`. Top-level single-purpose modules
  (`config.py`, `state.py`, `doctor.py`, `textsafe.py`) sit next to them.
- Non-Python package data (the TUI `.tcss`) lives next to the code that uses it
  and is listed in `[tool.setuptools.package-data]`; CI checks the built wheel
  really contains it.
- `config.example.toml` committed at the repo root; the real `config.toml`,
  `.env`, keys, state and logs are git-ignored.

## Naming

| Thing | Convention | Examples |
|---|---|---|
| Modules | short, lower case, one word when possible | `queue.py`, `pairing.py`, `textsafe.py` |
| Classes | nouns, CapWords; enums inherit `str, enum.Enum` when they are shown or serialised | `PriorityGate`, `DeviceStatus`, `ActiveStream` |
| Functions | verbs, snake_case; private helpers prefixed `_` | `load_config`, `check_ollama_binding`, `_private_dir` |
| Constants | UPPER_CASE at module top, with a comment saying *why* the value | `REFRESH_INTERVAL = 1.0  # redraws share the proxy's event loop; keep them rare` |
| Loggers | `logging.getLogger("ollamapower.<area>")` | `ollamagate.proxy`, `ollamagate.admin` |
| CLI | `ollamapower <noun> <verb>` sub-apps, `--kebab-case` flags | `devices list`, `bans add`, `--no-tui`, `--dry-run` |
| Config keys | `[section]` + snake_case key; env override `OLLAMAPOWER_<SECTION>__<KEY>` | `server.max_upstream_concurrency`, `OLLAMAGATE_SERVER__PORT` |
| Admin IPC commands | dotted `noun.verb` strings | `devices.list`, `pairing.open`, `lockdown.set` |
| Tests | `tests/test_<area>.py`; later-added review fixes in `test_council_p<N>.py` | `test_policy.py`, `test_council_p2.py` |

DCP's names (`master`, `worker`, `ctrl_*`, `PACKET_FLAGS`) are translated to the
spec's glossary: master → **hub**, worker → **node / helper / agent**. Wire-level
enum *member* names keep DCP's `ctrl_hello` style so the protocol stays
recognisable.

## Code style

- Every module starts with `from __future__ import annotations` and a module
  docstring that explains *what it is for and why it is built that way*
  (threat, invariant, trade-off), not a list of functions.
- Full type hints on public functions and methods, modern syntax
  (`str | None`, `list[str]`, `dict[str, Any]`). `@dataclass` for state and
  config objects.
- Docstrings: one-line summary, then a short paragraph only where the behaviour
  is subtle. Comments are sparse and say *why*
  (`# Bind now, so "address in use" is reported before any TUI takes over`).
  DCP's chatty comments are rewritten in this tone when code is ported.
- Line length ~120. ruff enforces formatting-level rules (OllamaGate has
  no linter in CI; the spec adds ruff).
- Errors: one small exception class per concern (`ConfigError`, `AdminError`,
  `NotRunning`), raised with a message that tells the owner how to fix it.
  Broad `except Exception` only at process boundaries, always with
  `# noqa: BLE001 - <reason>`.
- Never `print` from library code (DCP did); log instead. CLI output goes
  through `typer.echo` / `typer.secho`.
- Logging uses `%`-style arguments (`log.warning("x failed: %s", exc)`), never
  f-strings, and logs exception *types* rather than messages when the message
  could carry secrets. Noisy third-party loggers (`httpx`, `httpcore`) are set to
  WARNING.
- asyncio everywhere on one event loop. Plain (non-`async`) state mutations are
  treated as atomic; an `asyncio.Lock` is taken only to keep state consistent
  across an `await`. Background loops are `asyncio.ensure_future` tasks that are
  cancelled in `finally`. Cancellation is never swallowed
  (`if current.cancelling(): raise`): stop helper tasks with `aio.reap()`, not
  `suppress(CancelledError)`, and use `aio.wait_for()`, not
  `asyncio.wait_for()` (on Python 3.11 the latter can drop a cancellation
  that races the result, and a shut-down task keeps running).
- Constants and tunables: module-level constants for fixed values, `config.toml`
  for anything the owner might want to change. Config is plain dataclasses per
  section, loaded from TOML, overridden by env vars, then `validate()`d; unknown
  keys and sections are errors; every value has a safe default so running
  without a config file works.
- Attacker- or network-controlled text is passed through `textsafe.safe()`
  before it reaches a terminal (CLI and TUI), and rendered as Rich `Text`
  (never markup) in the TUI.

## TUI look and feel (Textual)

- One `App` subclass per program, `CSS_PATH = "<name>.tcss"`,
  `ENABLE_COMMAND_PALETTE = False`, `Header(show_clock=True)` + `Footer()`.
- Screens are tabs in a `TabbedContent`, switched with digit keys `1`–`N`
  (bindings with `show=False`). Detail views and dialogs are `ModalScreen`s
  (`ConfirmScreen`, `InputScreen`, `MessageScreen`) with a `.dialog` class.
- Single-letter bindings for actions, defined on the table they act on
  (`Binding("k", "app.device_kick", "Kick")`); `/` focuses a search box;
  `q` quits **with confirmation** and `ctrl+q` is rebound to the same action.
  Destructive actions always ask (`ConfirmScreen`, `y`/`n`/`escape`).
- Live data: the TUI polls a cheap `state.version` counter about once a second
  and redraws only what changed and only the visible tab.
- Status dots: `●` bold green = active, `●` yellow = recent/paused, `○` dim =
  idle, `●` bold red = suspended/offline, `✕` red = revoked/incompatible. The
  README uses the matching emoji (🟢 🟡 ⚪ 🔴).
- Warnings sit at the top of the dashboard in a red rounded border, one line
  each, prefixed `⚠ `, bold red. Critical modes get a full-width
  `bold white on red` banner (OllamaGate's LOCKDOWN). Toasts (`notify`) for
  events the owner should notice now.
- Panels: `border: round $primary-darken-2; padding: 0 1;` key/value grids
  (`Table.grid`, bold left column), `zebra_stripes=True` data tables with
  `cursor_type="row"`.
- If the TUI crashes, the service keeps running headless and says so on stderr.

## CLI conventions

- Typer, `add_completion=False, no_args_is_help=True`; sub-apps added with
  `cli.add_typer(..., name="devices")`. Every command has a one-line docstring
  (it is the `--help` text).
- `--config/-c` on every command; `--yes/-y` to skip confirmations.
- Commands other than the server itself talk to the running process over the
  local admin channel (Unix socket `0600` in a `0700` dir, or `127.0.0.1` on
  Windows, plus a secret in an owner-only `admin.json` and an HMAC proof on
  every reply). Never over the network.
- Output: aligned fixed-width columns with an upper-case header row for lists,
  `Label:` + value rows for details, timestamps as `YYYY-MM-DD HH:MM:SS`, short
  ids as the first 8 characters. Errors in red on stderr.
- Exit codes: `0` ok, `1` command failed / doctor found an error, `2` bad
  config or usage, `3` the service isn't running.
- `--json` and `--watch` are **spec additions** (OllamaGate has neither):
  `--json` prints the same dict the admin channel returned, `--watch` reprints
  the plain view in place.
- Refuse to run as root/Administrator unless `--allow-root`.

## Security posture

OllamaGate is strict and fails closed; OllamaPower matches that:

- Bind to loopback by default; anything else needs an explicit opt-in and a
  loud warning. Ollama must stay on loopback (validated in config).
- 256-bit secrets from `secrets`, compared with `hmac.compare_digest`; only
  hashes are stored where possible; secrets never in logs, audit entries or
  notifications.
- Owner-only files (`0600`, dirs `0700`), written atomically (temp file +
  `os.replace`), refusing symlinked or foreign-owned directories.
- Prompt/response content is never logged unless explicitly enabled.
- Bounded everything: message sizes, queue lengths, timeouts, ring buffers.
- `doctor` checks the real setup and prints `[OK]`/`[WARN]`/`[ERROR]` with a
  `fix:` line.
- `SECURITY.md` states the threat model, what is protected, and the limits,
  honestly; it is updated after every phase.

## Tests

- pytest + pytest-asyncio (`asyncio_mode = "auto"`), `testpaths = ["tests"]`.
- Fakes over mocks: a real fake Ollama (`tests/fake_ollama.py`, a Starlette app
  with a `Recorder`) served by a real uvicorn in a thread (`ThreadedServer`);
  factories as fixtures (`gate_factory`, `make_config`).
- Test names say the behaviour: `test_no_token_gets_generic_401`,
  `test_control_files_are_private`. Helpers like `wait_for(predicate)` instead
  of fixed sleeps.
- Every review finding that gets fixed gets a regression test.
- DCP's script-style tests (top-level asserts, `print("all good :)")`) are
  rewritten as proper pytest functions.

## Dependencies and CI

- Ranged dependencies in `pyproject.toml`; exact, hashed lock files generated by
  `uv pip compile --universal --generate-hashes --python-version 3.11`
  (`requirements.lock`, `requirements-dev.lock`).
- GitHub Actions kept cheap: Linux, Python 3.11 + 3.13, on pushes to
  `main`/`dev` and on PRs, docs-only changes skipped; Windows/macOS on demand
  via `workflow_dispatch`. A second short job builds the wheel, smoke-tests the
  installed CLI and runs `pip-audit --require-hashes`.

## README and docs

- Plain headings, **no emoji headers** (DCP had them, OllamaGate dropped them;
  OllamaGate wins). Emoji only as status legends.
- Order: one-paragraph pitch, Requirements, Install, Run, feature sections with
  tables of keys/commands/settings, Configuration, What's exposed, Tests.
- Honest about limits: say what doesn't work and why, right next to the
  feature.

## Commits

- Conventional Commits with a scope: `feat(tls): ...`, `fix(tui): ...`,
  `test: ...`, `docs: ...`, `ci: ...`. Imperative, lower case after the colon,
  ≤ 72 characters.
- The body explains *why*, often as a short bullet list; review-fix commits
  list what the review found.
- Phases are developed on a branch and merged with
  `Merge Phase N: <summary>`.
- Author and committer: `Yoav <197086229+111yoav111@users.noreply.github.com>`;
  Claude is credited with a `Co-Authored-By: Claude ...` trailer (the owner's
  rule, replacing the spec's original "no AI co-author trailers"). Still no
  session links in commits, PRs or code.
