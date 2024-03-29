use std::ffi::OsString;
use std::fs::{read_dir, File};
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
    Suggest { path: Option<String> },
    Clean,
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

fn clean(dest: OsString) {
    let _ = std::fs::remove_file(&dest).expect(&*format!("Unable to delete {:#?}", dest));
}

fn check() {
    println!("TODO: Check with clarify (look for code lacking documentation and report it, also look for code that appears to have out-of-date documentation)");
}

fn get_files_rec(path: OsString) -> Vec<OsString> {
    let mut out = vec![];
    match read_dir(path) {
        Err(_) => return out, // TODO: Is there any better solution for this?
        Ok(entries) => {
            for entry in entries {
                match entry {
                    Err(_) => return out, // TODO: How can a single entry be an error in the first
                                          // place?
                    Ok(entry) => {
                        if entry.path().is_dir() {
                            if !entry.path().into_os_string().into_string().expect("the unexpected").ends_with("target") {
                                out.extend(get_files_rec(entry.path().into_os_string()));
                            }
                        } else if entry.path().into_os_string().into_string().expect("the unexpected").ends_with(".rs") {
                            out.push(entry.path().into_os_string());
                        }
                    }
                }
            }
        }
    }
    return out;
}

fn suggest(path: Option<String>, model_path: OsString) {
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
        context_size: 1024,
        ..Default::default()
    };

    let llama = LLama::new(model_path, &model_options).unwrap();

    let files = get_files_rec(path.unwrap_or(".".to_string()).into());
    for filename in files {
        let mut file = File::open(filename).expect("Unable to open file");
        let mut src = String::new();
        file.read_to_string(&mut src).expect("Unable to read file");
        let syntax = syn::parse_file(&src).unwrap();

        for item in syntax.items {
            match item {
                syn::Item::Fn(i) => {
                    let predict_options = PredictOptions {
                        tokens: 0,
                        threads: num_threads,
                        ..Default::default()
                    };
                    let docs = if i.attrs.len() > 0 {
                      let docs: Vec<&syn::Attribute> = i.attrs.iter().filter(|attr| {
                        if let syn::Meta::NameValue(nv) = &attr.meta {
                          if let Some(ident) = nv.path.get_ident() {
                            if ident.to_string() == "doc" {
                              return true;
                            }
                          }
                        }
                        return false;
                      }).collect();
                      if docs.len() > 0 {
                        Some(docs)
                      } else {
                        None
                      }
                    } else {
                      None
                    };
                    let out = llama
                        .predict(
                            match docs {
                              Some(docs) => {
                                format!(
                                    include_str!("./prompts/rs_with_comments.txt"),
                                    i.span().source_text().unwrap().replace("```", "\\`\\`\\`").replace("\n", "\\n"),
                                    docs.iter().map(|attr| {
                                      if let syn::Meta::NameValue(nv) = &attr.meta {
                                        if let syn::Expr::Lit(lit) = &nv.value {
                                          if let syn::Lit::Str(s) = &lit.lit {
                                            s.value()
                                          } else {
                                            "".to_string()
                                          }
                                        } else {
                                          "".to_string()
                                        }
                                      } else {
                                        "".to_string()
                                      }
                                    }).collect::<Vec<String>>().join("\n")
                                )
                              }
                              None => {
                                format!(
                                    include_str!("./prompts/rs.txt"),
                                    i.span().source_text().unwrap().replace("```", "\\`\\`\\`").replace("\n", "\\n")
                                )
                              }
                            },
                            predict_options,
                        )
                        .unwrap();
                    println!("{}", out);
                }
                _ => {}
            }
        }
    }
}

/// This is a rustdoc comment!
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
        Commands::Suggest { path } => suggest(path, model_path),
        Commands::Clean => clean(model_path),
    }
}
