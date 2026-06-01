use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Selects the batching strategy used when processing files.
#[derive(Clone, Debug, PartialEq, ValueEnum)]
pub enum BatchingMode {
    /// Sequential: the summary of each batch is fed into the next (default).
    Linear,
    /// Tree: batches within a level are processed (potentially in parallel) and
    /// their summaries are re-batched level by level until one result remains.
    Tree,
}

#[derive(Parser)]
#[command(author, version, about = "Iterativly summarize multiple text files using LLMs.", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// The files to summarize (if no subcommand is given)
    #[arg(global = false)]
    pub files: Vec<PathBuf>,

    /// Instruction/Prompt for the summarization
    #[arg(short, long)]
    pub prompt: Option<String>,

    /// File containing the instruction/prompt
    #[arg(short = 'f', long)]
    pub prompt_file: Option<PathBuf>,

    /// Model to use (e.g. google:gemini-1.5-flash)
    #[arg(short, long)]
    pub model: Option<String>,

    /// Show debug information
    #[arg(long)]
    pub debug: bool,

    /// Batching mode: linear (rolling sequential) or tree (parallel fan-in)
    #[arg(long, default_value = "linear", value_name = "MODE")]
    pub batching_mode: BatchingMode,

    /// Maximum number of concurrent LLM requests (1 = fully sequential)
    #[arg(long, default_value = "1", value_name = "N")]
    pub max_concurrency: usize,
}

#[derive(Subcommand)]
pub enum Commands {
    /// List available models from configured providers
    ListModels,
    /// Initialize the configuration interactively
    Init,
    /// Select the default model interactively
    DefaultModel,
}
