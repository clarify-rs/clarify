use std::thread::available_parallelism;

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

    let predict_options = PredictOptions {
        tokens: 0,
        threads: num_threads,
        top_k: 90,
        top_p: 0.86,
        token_callback: Some(Box::new(|token| {
            print!("{}", token);

            true
        })),
        ..Default::default()
    };

    let out = llama
        .predict(
            r#"
Consider the following rust function:

```rs
pub async fn run(bytecode: Vec<i64>, http_config: Option<HttpType>) -> VMResult<()> {
  let program = Program::load(bytecode, http_config);
  PROGRAM
    .set(program)
    .map_err(|_| VMError::Other("A program is already loaded".to_string()))?;
  let mut vm = VM::new()?;
  const START: EventEmit = EventEmit {
    id: BuiltInEvents::START as i64,
    payload: None,
  };
  vm.add(START)?;
  vm.run().await
}
```

There is no rustdoc comment above the function describing its purpose.

This is not good for future developers who encounter this function in the future. So, after reading the code to understand what it is doing, we add the rustdoc comment, and the full function with annotations is as follows:

```rs

"#.into(),
            predict_options,
        )
        .unwrap();
    println!("output: {}", out);
}
