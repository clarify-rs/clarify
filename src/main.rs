use std::fs::File;
use std::io::Read;
use std::thread::available_parallelism;

use syn::spanned::Spanned;
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

fn main() {
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
                  format!(r#"
Consider the following rust function:

```rs
{}
```

There is no rustdoc comment above the function describing its purpose.

This is not good for future developers who encounter this function in the future. So, after reading the code to understand what it is doing, we add the rustdoc comment, and the full function with annotations is as follows:

```rs

      "#, i.span().source_text().unwrap().replace("```", "\\`\\`\\`")),
                  predict_options,
              )
              .unwrap();
          println!("output: {}", out);
          //println!("fn: {}", i.span().source_text().unwrap());
        },
        _ => (),
      }
    }
}
