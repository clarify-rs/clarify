use std::fs::File;
use std::io::Read;
use std::thread::available_parallelism;

use clap::{Parser, Subcommand};
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use syn::spanned::Spanned;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
  #[command(subcommand)]
  command: Commands
}

#[derive(Subcommand)]
enum Commands {
  Init,
  Check,
  Suggest,
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Init => {
          println!("TODO: Initialize clarify (download a Llama model, perhaps asking which one they want)");
        },
        Commands::Check => {
          println!("TODO: Check with clarify (look for code lacking documentation and report it, also look for code that appears to have out-of-date documentation)");
        },
        Commands::Suggest => {
          println!("TODO: Do this more correctly (it's what the code below is starting to do, generate documentation comments for existing code)");
        }
    }
    let num_threads: i32 = available_parallelism().unwrap().get() as i32;
    let model_path = match std::env::var("GGUF") {
        Ok(v) => v.into(),
        Err(_) => "/oss/CodeLlama-13B-Instruct-GGUF/codellama-13b-instruct.Q4_K_M.gguf".into(), // TODO: Better error handling here
    };

    let model_options = ModelOptions {
        n_gpu_layers: 43,
        numa: true,
        ..Default::default()
    };

    let llama = LLama::new(model_path, &model_options).unwrap();

    // TODO: Don't hardwire this
    let mut file = File::open("./src/main.rs").expect("Unable to open file");
    let mut src = String::new();
    file.read_to_string(&mut src).expect("Unable to read file");
    let syntax = syn::parse_file(&src).unwrap();

    for item in syntax.items {
        match item {
            syn::Item::Fn(i) => {
                let predict_options = PredictOptions {
                    tokens: 0,
                    threads: num_threads,
                    token_callback: Some(Box::new(|token| {
                        print!("{}", token);

                        true
                    })),
                    ..Default::default()
                };
                let out = llama
                    .predict(
                        format!(
                            include_str!("./prompts/rs.txt"),
                            i.span().source_text().unwrap().replace("```", "\\`\\`\\`")
                        ),
                        predict_options,
                    )
                    .unwrap();
                println!("output: {}", out);
                //println!("fn: {}", i.span().source_text().unwrap());
            }
            _ => (),
        }
    }
}
