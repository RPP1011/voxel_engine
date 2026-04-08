use glam::{IVec3, Vec3};
use crate::scene::handle::EntityHandle;

#[derive(Debug, Clone)]
pub struct RayHit {
    pub entity: EntityHandle,
    pub point: Vec3,
    pub normal: Vec3,
    pub voxel_pos: IVec3,
    pub distance: f32,
}
