use crate::scene::cleanup::CleanupPolicy;

/// Top-level configuration for a Scene.
pub struct SceneConfig {
    pub cleanup: CleanupPolicy,
    /// Fixed-step physics tick rate in Hz.
    pub physics_tick_rate: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            cleanup: CleanupPolicy::default(),
            physics_tick_rate: 10.0,
        }
    }
}
