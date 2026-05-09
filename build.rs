use std::fs;
use std::path::Path;

fn main() {
    let shader_dir = Path::new("shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_shader_dir = Path::new(&out_dir).join("shaders");
    fs::create_dir_all(&out_shader_dir).unwrap();

    println!("cargo::rerun-if-changed=shaders/");
    println!("cargo::rerun-if-changed=shaders/compiled/");

    // CARGO_FEATURE_COMPILE_SHADERS is set by Cargo when --features compile-shaders is passed
    if std::env::var("CARGO_FEATURE_COMPILE_SHADERS").is_ok() {
        compile_shaders(shader_dir, &out_shader_dir);
    } else {
        copy_precompiled_shaders(shader_dir, &out_shader_dir);
    }
}

/// Compile GLSL shaders to SPIR-V using shaderc, and also write them to shaders/compiled/
/// so they can be committed as pre-built artifacts.
fn compile_shaders(shader_dir: &Path, out_shader_dir: &Path) {
    let compiler = shaderc::Compiler::new().expect("Failed to create shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("Failed to create compile options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );

    for entry in fs::read_dir(shader_dir).expect("Failed to read shaders directory") {
        let entry = entry.unwrap();
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str());

        let shader_kind = match ext {
            Some("comp") => shaderc::ShaderKind::Compute,
            Some("vert") => shaderc::ShaderKind::Vertex,
            Some("frag") => shaderc::ShaderKind::Fragment,
            _ => continue,
        };

        let source = fs::read_to_string(&path).unwrap();
        let file_name = path.file_name().unwrap().to_str().unwrap();

        let artifact = compiler
            .compile_into_spirv(&source, shader_kind, file_name, "main", Some(&options))
            .unwrap_or_else(|e| panic!("Failed to compile {file_name}: {e}"));

        let spv_name = format!("{file_name}.spv");
        let spv_bytes = artifact.as_binary_u8();

        // Write to OUT_DIR for include_bytes!
        let spv_path = out_shader_dir.join(&spv_name);
        fs::write(&spv_path, spv_bytes).unwrap();

        // Also copy to shaders/compiled/ so they can be committed as pre-compiled artifacts
        let compiled_dir = shader_dir.join("compiled");
        fs::create_dir_all(&compiled_dir).unwrap();
        fs::write(compiled_dir.join(&spv_name), spv_bytes).unwrap();
    }
}

/// Copy pre-compiled SPIR-V files from shaders/compiled/ to OUT_DIR/shaders/.
/// Falls back to writing minimal SPIR-V stubs if shaders/compiled/ does not exist.
///
/// Fails the build if any shader source is newer than its committed SPV. The
/// previous behaviour silently shipped stale SPIR-V whenever a developer
/// edited a shader without remembering `--features compile-shaders`, which
/// burned a multi-day debugging session chasing visual artifacts that came
/// from running an old vertex shader. Failing fast with a clear message is
/// the cheapest fix that keeps the precompiled-by-default story intact.
fn copy_precompiled_shaders(shader_dir: &Path, out_shader_dir: &Path) {
    let compiled_dir = shader_dir.join("compiled");

    if compiled_dir.exists() {
        check_spv_freshness(shader_dir, &compiled_dir);
        for entry in fs::read_dir(&compiled_dir)
            .expect("Failed to read shaders/compiled/ directory")
            .flatten()
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("spv") {
                let file_name = path.file_name().unwrap();
                let dest = out_shader_dir.join(file_name);
                fs::copy(&path, &dest).unwrap_or_else(|e| panic!("Failed to copy {:?}: {e}", path));
            }
        }
    } else {
        // shaders/compiled/ not yet populated — write minimal SPIR-V stubs so that
        // include_bytes! in the renderer can resolve during initial development.
        // Run `cargo build --features compile-shaders` to generate real shaders.
        println!(
            "cargo::warning=shaders/compiled/ not found — writing SPIR-V stubs. \
             Run `cargo build --features compile-shaders` to generate real shaders."
        );
        stub_shaders(shader_dir, out_shader_dir);
    }
}

/// Panic if any GLSL source in `shader_dir` is newer than its corresponding
/// `.spv` in `compiled_dir`. The error message tells the developer exactly
/// how to refresh the SPV cache. See `copy_precompiled_shaders` for why this
/// matters.
fn check_spv_freshness(shader_dir: &Path, compiled_dir: &Path) {
    // Self-disable when consumed as a cargo git/registry dep: the source
    // tree lives under cargo's checkout cache, the consumer has no way to
    // edit the shader sources, and the mtimes there are arbitrary (cargo
    // doesn't preserve git-commit ordering on checkout). The freshness
    // check only catches real edits in voxel_engine's own working copy.
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        if manifest_dir.contains("/.cargo/git/checkouts/")
            || manifest_dir.contains("/.cargo/registry/src/")
        {
            return;
        }
    }
    // Explicit escape hatch — set VOXEL_ENGINE_SKIP_FRESHNESS_CHECK=1 to
    // bypass the check entirely (e.g. CI building a vendored snapshot).
    if std::env::var("VOXEL_ENGINE_SKIP_FRESHNESS_CHECK").is_ok() {
        return;
    }

    let mut stale: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();

    let entries = match fs::read_dir(shader_dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str());
        if !matches!(ext, Some("vert") | Some("frag") | Some("comp")) {
            continue;
        }
        let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if file_name.is_empty() {
            continue;
        }
        let spv_path = compiled_dir.join(format!("{file_name}.spv"));

        let src_mtime = match fs::metadata(&path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let spv_mtime = match fs::metadata(&spv_path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(_) => {
                missing.push(file_name.to_string());
                continue;
            }
        };
        if src_mtime > spv_mtime {
            stale.push(file_name.to_string());
        }
    }

    if stale.is_empty() && missing.is_empty() {
        return;
    }

    let mut msg = String::from(
        "\n\
         Shader SPIR-V cache is out of date — running this build would ship\n\
         stale precompiled shaders and silently mask your source-level edits.\n\
         \n",
    );
    if !stale.is_empty() {
        msg.push_str("  Source newer than shaders/compiled/*.spv:\n");
        for f in &stale {
            msg.push_str(&format!("    - {f}\n"));
        }
    }
    if !missing.is_empty() {
        msg.push_str("  Source files with no committed SPV:\n");
        for f in &missing {
            msg.push_str(&format!("    - {f}\n"));
        }
    }
    msg.push_str(
        "\n\
         Refresh the SPV cache and commit the result:\n\
           cargo build --features compile-shaders -p voxel_engine\n\
           git add shaders/compiled/*.spv\n",
    );
    panic!("{msg}");
}

/// Write minimal valid SPIR-V headers as placeholder .spv files.
fn stub_shaders(shader_dir: &Path, out_shader_dir: &Path) {
    let magic: [u8; 20] = [
        0x03, 0x02, 0x23, 0x07, // SPIR-V magic
        0x00, 0x00, 0x01, 0x00, // version 1.0
        0x00, 0x00, 0x00, 0x00, // generator
        0x01, 0x00, 0x00, 0x00, // bound = 1
        0x00, 0x00, 0x00, 0x00, // schema
    ];

    if let Ok(entries) = fs::read_dir(shader_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());
            match ext {
                Some("comp") | Some("vert") | Some("frag") => {
                    let file_name = path.file_name().unwrap().to_str().unwrap();
                    let spv_path = out_shader_dir.join(format!("{file_name}.spv"));
                    if !spv_path.exists() {
                        fs::write(&spv_path, &magic).unwrap();
                    }
                }
                _ => {}
            }
        }
    }
}
