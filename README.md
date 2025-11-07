# Rust / Vulkan / Winit / Egui Example

This is an example of using rust, winit, vulkan, and egui together without using eframe.

This uses Vulkan 1.3 with dynamic rendering, including the features needed for gpu-driven rendering

<img width="837" height="632" alt="vulkan" src="https://github.com/user-attachments/assets/001493e8-fa77-4eab-b621-baf455c1890b" />

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
