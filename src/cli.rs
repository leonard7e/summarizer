use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
