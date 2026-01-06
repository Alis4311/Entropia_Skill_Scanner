# Entropia Skill Scanner

Command-line utilities for running the OCR pipeline and exporting skills/professions.

## CLI

```
entropia-skillscanner scan <inputs...> [--json]
entropia-skillscanner export --from-scan <scan.json> [--output skills.csv]
```

- `scan` accepts images or directories (recursively) and runs the OCR pipeline.
- `export` consumes the stable scan JSON and emits the CSV export or JSON summary.

### Examples

```bash
# Produce scan JSON (stable schema)
entropia-skillscanner scan screenshot.png --json > scan.json

# Export to CSV
entropia-skillscanner export --from-scan scan.json --output skills.csv

# Pipe between commands
entropia-skillscanner scan samples/ --json | entropia-skillscanner export --from-scan -
```

### Stable JSON schema (scan)

The scan subcommand emits a stable array of objects:

```json
[
  {
    "input": "<path to image>",
    "status": "<pipeline status string>",
    "ok": true,
    "rows": [
      {"name": "<skill name>", "value": "<stringified value>"}
    ],
    "logs": ["<pipeline log message>", "..."]
  }
]
```

This schema is preserved by `entropia_skillscanner.api.scan_paths()` and is accepted by the `export` subcommand.
