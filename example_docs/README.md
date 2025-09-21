# Example CLI Tool

This is an example CLI tool for testing EasyPrompt.

## Installation

```bash
pip install example-cli
```

## Commands

### list
List all items:
```bash
example-cli list
```

Options:
- `--format`: Output format (json, table)
- `--filter`: Filter by criteria

### add
Add a new item:
```bash
example-cli add "item name"
```

Options:
- `--description`: Item description
- `--tags`: Comma-separated tags

### delete
Delete an item:
```bash
example-cli delete <item-id>
```

### status
Check system status:
```bash
example-cli status
```
