use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::thread::available_parallelism;

use clap::{Parser, Subcommand};
use dirs::data_dir;
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use reqwest::blocking::get;
use syn::spanned::Spanned;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init { url: Option<String> },
    Check,
    Suggest,
}

fn init(url: Option<String>, dest: OsString) {
    let url = url.clone().unwrap_or("https://huggingface.co/TheBloke/CodeLlama-13B-Instruct-GGUF/resolve/main/codellama-13b-instruct.Q4_K_M.gguf?download=true".to_string());
    match get(&url) {
        Ok(mut res) => {
            let mut out =
                File::create(&dest).expect(&*format!("Unable to open {:#?} for writing", dest));
            println!("Clarify is initializing. This will take a while."); // TODO: Add a spinner or
                                                                          // ideally a progress bar
            match res.copy_to(&mut out) {
                Ok(_) => {
                    println!(
                        "Clarify has been initialized!\n\n{}\nhas been downloaded into\n{:#?}",
                        url, dest
                    );
                }
                Err(error) => {
                    eprintln!("Failed to fetch {}:\n\n{:#?}", url, error);
                }
            };
        }
        Err(error) => {
            eprintln!("Failed to fetch {}:\n\n{:#?}", url, error);
        }
    };
}

fn check() {
    println!("TODO: Check with clarify (look for code lacking documentation and report it, also look for code that appears to have out-of-date documentation)");
}

fn suggest(model_path: OsString) {
    println!("TODO: Do this more correctly (it's what the code below is starting to do, generate documentation comments for existing code)");
    let num_threads: i32 = available_parallelism().unwrap().get() as i32;
    let model_path = match std::env::var("GGUF") {
        Ok(v) => v,
        Err(_) => model_path
            .into_string()
            .expect("What platform has non-Unicode-mappable paths?"),
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

fn main() {
    let model_path = match data_dir() {
        Some(mut data_path) => {
            data_path.set_file_name("clarify_model.gguf");
            data_path.into_os_string()
        }
        None => "~/.clarify_model.gguf".into(), // For the BSDs, I guess?
    };
    let cli = Cli::parse();
    match cli.command {
        Commands::Init { url } => init(url, model_path),
        Commands::Check => check(),
        Commands::Suggest => suggest(model_path),
    }
}
