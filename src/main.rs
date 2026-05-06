mod cli;
mod config;
mod engine;
mod file;
mod provider;
mod setup;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use config::Config;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle subcommands or execute the main summarization flow if no subcommand is provided.
    match cli.command {
        Some(Commands::Init) => setup::run_initialization().await,
        Some(cmd) => {
            let mut config = Config::load()?;
            match cmd {
                Commands::ListModels => {
                    println!("Available Models:");
                    let all_models = setup::get_all_models(&config).await;
                    for model in all_models {
                        println!("- {}", model);
                    }
                }
                Commands::DefaultModel => {
                    setup::select_default_model(&mut config).await?;
                }
                Commands::Init => unreachable!(), // Handled above
            }
            Ok(())
        }
        None => {
            // Determine which model to use: CLI argument overrides the default from config.
            let config = Config::load()?;
            if cli.files.is_empty() {
                println!("No files provided. Use `summarizer --help` for usage.");
                return Ok(());
            }

            let model_str = cli
                .model
                .or(config.default_model.clone())
                .unwrap_or_else(|| "ollama/llama3".to_string());

            let file_prompt = cli
                .prompt_file
                .map(|f| std::fs::read_to_string(&f))
                .transpose()?;

            // Combine prompt from file and CLI argument, falling back to a default instruction if neither is provided.
            let final_prompt = file_prompt
                .into_iter()
                .chain(cli.prompt)
                .reduce(|a, b| format!("{}\n\n{}", a, b))
                .unwrap_or_else(|| {
                    "Please summarize the following text comprehensively.".to_string()
                });

            engine::run_summarize_loop(cli.files, config, &model_str, cli.debug, &final_prompt)
                .await
        }
    }
}
