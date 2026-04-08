#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugRenderMode {
    Normals,
    Depth,
    Albedo,
    Material,
}

pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub vsync: bool,
    pub shadow_resolution: u32,
    pub debug_mode: Option<DebugRenderMode>,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            vsync: true,
            shadow_resolution: 2048,
            debug_mode: None,
        }
    }
}
