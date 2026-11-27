use anyhow::Result;
use clap::Parser;
use infernix_gpu_admission::Cli;

fn main() -> Result<()> {
    Cli::parse().run()
}
