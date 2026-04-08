use crate::render::RendererConfig;
use crate::scene::SceneConfig;

pub struct AppConfig {
    pub window_title: String,
    pub width: u32,
    pub height: u32,
    pub renderer: RendererConfig,
    pub scene: SceneConfig,
    pub fixed_tick_rate: f32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            window_title: "Voxel Engine".to_string(),
            width: 1280,
            height: 720,
            renderer: RendererConfig::default(),
            scene: SceneConfig::default(),
            fixed_tick_rate: 10.0,
        }
    }
}

pub trait App {
    fn setup(&mut self, scene: &mut crate::scene::Scene) -> anyhow::Result<()>;
    fn tick(&mut self, scene: &mut crate::scene::Scene, dt: f32);
    fn on_input(&mut self, scene: &mut crate::scene::Scene, event: &winit::event::WindowEvent);
}
