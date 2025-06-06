# Rust / Vulkan / Egui Example

This is an example of using rust, vulkan, and egui together without using eframe.

This uses Vulkan 1.3 with dynamic rendering, including the features needed for gpu-driven rendering

## Quickstart

Run using the precompiled shaders with:

```rust
cargo run -r
```

If you'd like to edit the shaders, install the [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) to get `glslangValidator` for compiling glsl to spir-v

Then install [just](https://github.com/casey/just) to run the commands in the justfile. (or you can run them manually)
 
```rust
just run
```


![image](https://github.com/user-attachments/assets/4beef2f5-5895-4402-9977-51e04f006d0a)
