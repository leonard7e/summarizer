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

    if let Some(Commands::Init) = &cli.command {
        return setup::run_initialization().await;
    }

    let mut config = Config::load()?;

    match &cli.command {
        Some(Commands::ListModels) => {
            println!("Available Models:");
            let all_models = setup::get_all_models(&config).await;
            for model in all_models {
                println!("- {}", model);
            }
        }
        Some(Commands::Init) => unreachable!(),
        Some(Commands::DefaultModel) => {
            setup::select_default_model(&mut config).await?;
        }
        None => {
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

            let final_prompt = file_prompt
                .into_iter()
                .chain(cli.prompt)
                .reduce(|a, b| format!("{}\n\n{}", a, b))
                .unwrap_or_else(|| {
                    "Please summarize the following text comprehensively.".to_string()
                });

            engine::run_summarize_loop(cli.files, config, &model_str, cli.debug, &final_prompt)
                .await?;
        }
    }

    Ok(())
}
