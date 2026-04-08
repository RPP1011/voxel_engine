/// Policy controlling how dead/sleeping fragments are cleaned up.
pub struct CleanupPolicy {
    /// Fragments with fewer live voxels than this are immediately removed.
    pub min_fragment_voxels: u32,
    /// Seconds a body must be at rest before being removed.
    pub rest_timeout_secs: f32,
    /// Entities below this Y value are removed.
    pub kill_plane_y: f32,
    /// Maximum number of live fragment bodies allowed at once.
    pub max_fragments: u32,
    /// Duration of the fade-out effect before final removal.
    pub fade_out_secs: f32,
    /// If set, entities farther than this from any loaded chunk are removed.
    pub max_distance_from_loaded_chunks: Option<f32>,
    /// If set, entities that have been fully offscreen for this long are removed.
    pub offscreen_timeout_secs: Option<f32>,
}

impl Default for CleanupPolicy {
    fn default() -> Self {
        Self {
            min_fragment_voxels: 4,
            rest_timeout_secs: 10.0,
            kill_plane_y: -100.0,
            max_fragments: 1024,
            fade_out_secs: 0.5,
            max_distance_from_loaded_chunks: None,
            offscreen_timeout_secs: None,
        }
    }
}
