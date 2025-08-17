set windows-shell := ["powershell.exe"]

export RUST_LOG := "info"
export RUST_BACKTRACE := "1"

alias cs := compile-shaders

[private]
default:
    @just --list

# Check the workspace
check:
    cargo check --all --tests
    cargo fmt --all -- --check

# Compile glsl shaders to spir-v
[windows]
compile-shaders:
    powershell -Command "Get-ChildItem -Path ./src/shaders/ -File -Recurse -Exclude *.spv | ForEach-Object { `$sourcePath = `$_.FullName; `$targetPath = `$_.FullName + '.spv'; if (`$_.Name -match 'mesh\.(task|mesh)') { glslangValidator -V --target-env vulkan1.3 --spirv-val -o `$targetPath `$sourcePath } else { glslangValidator -V -o `$targetPath `$sourcePath } }"

# Compile glsl shaders to spir-v
[unix]
compile-shaders:
    find ./src/shaders/ -type f ! -name "*.spv" -exec sh -c 'if echo "$1" | grep -q "mesh\\.\\(task\\|mesh\\)"; then glslangValidator -V --target-env vulkan1.3 --spirv-val -o "${1}.spv" "$1"; else glslangValidator -V -o "${1}.spv" "$1"; fi' _ {} \;

# Autoformat the workspace
format:
    cargo fmt --all

# Lint the workspace
lint:
    cargo clippy --all --tests -- -D warnings

# Run in release mode
run: compile-shaders
    cargo run -r 