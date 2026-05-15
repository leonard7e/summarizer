# summarizer

A command-line tool that iteratively summarizes multiple text files using large language models (LLMs).

When the combined content of your files exceeds a model's context window, `summarizer` automatically splits them into batches and feeds each batch to the model along with the result of the previous iteration — building up a coherent, rolling summary.

## Features

- **Iterative batching** — handles arbitrarily large file sets by chaining batches
- **Multiple providers** — Google Gemini, OpenRouter, Ollama (local), and any OpenAI-compatible API
- **Custom prompts** — pass an instruction via `--prompt` or a prompt file via `--prompt-file`
- **Model selection** — specify any model at runtime with `--model provider:model_id`
- **Interactive setup** — `summarizer init` walks you through configuration

## Quick Start

### 1. Build

```bash
cargo build --release
# Binary: ./target/release/summarizer
```

### 2. Configure

```bash
summarizer init
```

This walks you through entering your API keys and selecting a default model.

### 3. Summarize

```bash
# Summarize a single file
summarizer report.txt

# Summarize multiple files with a custom prompt
summarizer -p "List the key action items." meeting1.txt meeting2.txt meeting3.txt

# Use a model different from the default
summarizer --model ollama:llama3 notes.txt
```

## Documentation

* **[User Manual](doc/user_manual.adoc)** — Installation, Quick Start, and Configuration.
* **[Development Manual](doc/development_manual.adoc)** — Architecture and Development Guidelines.

To build the manuals as HTML:

```bash
# User Manual
asciidoctor doc/user_manual.adoc -D doc/out/

# Development Manual
asciidoctor doc/development_manual.adoc -D doc/out/
```

To build as PDF:

```bash
asciidoctor-pdf doc/user_manual.adoc -D doc/out/
asciidoctor-pdf doc/development_manual.adoc -D doc/out/
```

Install the tools with: `gem install asciidoctor asciidoctor-pdf`

## Supported Providers

| Provider | Model format | Requires |
|---|---|---|
| Google Gemini | `google:gemini-1.5-flash` | API key |
| OpenRouter | `openrouter:<model-id>` | API key |
| Ollama | `ollama:<model-name>` | Local Ollama instance |
| OpenAI-compatible | `<name>:<model>` | API key + base URL |

## Configuration

The configuration file is stored at `~/.config/summarizer/config.yaml` (Linux/macOS) and is created automatically on first run. See the [User Manual](doc/user_manual.adoc#configuration) for the full field reference.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

## Disclaimer & Regulatory Notice

**Usage at Your Own Risk:** This software is provided "as is," without warranty of any kind. The author is not liable for any damages, data loss, or costs arising from the use of this tool.

**Data Privacy (GDPR/DSGVO):**
*   This tool is a client-side utility. The author does not collect, store, or process any of your data.
*   By using cloud providers (e.g., Google Gemini, OpenRouter), you are transmitting your file contents to third-party servers. 
*   **You (the user) are the sole data controller.** It is your responsibility to ensure that you have the legal right to process and transmit the data (especially personal or sensitive information) according to local laws like the GDPR.

**AI Compliance:**
*   This tool uses Large Language Models. Outputs may be inaccurate, biased, or hallucinated. Always verify critical information.
*   The user is responsible for complying with the Terms of Service of the respective AI providers used through this tool.
