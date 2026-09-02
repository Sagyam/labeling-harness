# CLAUDE.md

## Driving the app in a browser

A Playwright MCP server is configured in `.mcp.json` (server name: `playwright`), so
`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_console_messages`,
`browser_network_requests`, `browser_take_screenshot` etc. are available for debugging
UI issues against the running app.

Prefer `browser_snapshot` (accessibility tree, cheap) over `browser_take_screenshot`
for finding and clicking elements; take a screenshot when the question is visual —
layout, spacing, color, overflow.

### Bring the stack up first

The frontend calls the backend at `http://localhost:8000`; with the backend down the
UI renders but every panel shows "Failed to load ..." toasts.

Full stack:

    docker compose up -d          # postgres, minio, backend :8000, frontend :5173

Frontend only (fast, for pure UI/layout work — expect API errors in console):

    cd frontend && npm run dev    # http://localhost:5173

Then navigate to `http://localhost:5173/`.

### Notes

- Runs headless against system Chromium (`/usr/bin/chromium`). To watch the browser,
  drop `--headless` from `.mcp.json`.
- `@playwright/mcp` is pinned. Unpinning risks the bundled `playwright-core` wanting a
  browser build that is not installed.
- The server writes snapshots and console logs into `.playwright-mcp/` in the cwd; it
  is gitignored.
