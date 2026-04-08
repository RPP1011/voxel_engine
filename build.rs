use std::fs;
use std::path::Path;

fn main() {
    let shader_dir = Path::new("shaders");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_shader_dir = Path::new(&out_dir).join("shaders");
    fs::create_dir_all(&out_shader_dir).unwrap();

    println!("cargo::rerun-if-changed=shaders/");

    #[cfg(feature = "compile-shaders")]
    compile_shaders(shader_dir, &out_shader_dir);

    #[cfg(not(feature = "compile-shaders"))]
    stub_shaders(shader_dir, &out_shader_dir);
}

#[cfg(feature = "compile-shaders")]
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

        let spv_path = out_shader_dir.join(format!("{file_name}.spv"));
        fs::write(&spv_path, artifact.as_binary_u8()).unwrap();
    }
}

#[cfg(not(feature = "compile-shaders"))]
fn stub_shaders(shader_dir: &Path, out_shader_dir: &Path) {
    // Write minimal valid SPIR-V stubs (magic number only) so include_bytes! can resolve.
    // Real SPIR-V will be compiled when compile-shaders feature is enabled.
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
