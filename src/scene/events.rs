use glam::Vec3;
use crate::physics::PhysicsBody;
use super::handle::EntityHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupReason {
    TooSmall,
    RestTimeout,
    BelowKillPlane,
    FragmentLimitExceeded,
    OutOfRange,
    OffscreenTimeout,
}

#[derive(Debug, Clone)]
pub enum SceneEvent {
    FragmentCreated { parent: EntityHandle, fragment: EntityHandle, body: PhysicsBody },
    EntityCollision { a: EntityHandle, b: EntityHandle, point: Vec3, normal: Vec3 },
    EntityAtRest { handle: EntityHandle },
    FragmentCleaned { handle: EntityHandle, reason: CleanupReason },
}
