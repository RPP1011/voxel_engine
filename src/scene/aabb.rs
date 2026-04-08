use glam::Vec3;
use crate::voxel::grid::VoxelGrid;
use super::transform::Transform;

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn from_grid_and_transform(grid: &VoxelGrid, transform: &Transform) -> Self {
        let (w, h, d) = grid.dimensions();
        let s = transform.scale;
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(w as f32, 0.0, 0.0),
            Vec3::new(0.0, h as f32, 0.0),
            Vec3::new(0.0, 0.0, d as f32),
            Vec3::new(w as f32, h as f32, 0.0),
            Vec3::new(w as f32, 0.0, d as f32),
            Vec3::new(0.0, h as f32, d as f32),
            Vec3::new(w as f32, h as f32, d as f32),
        ];
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for corner in &corners {
            let world_pos = transform.position + transform.rotation * (*corner * s);

            min = min.min(world_pos);
            max = max.max(world_pos);
        }
        Self { min, max }
    }
}
