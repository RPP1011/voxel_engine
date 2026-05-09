#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    unused_assignments
)]

pub mod ai;
pub mod camera;
pub mod compute;
pub mod fluid;
pub mod physics;
pub mod render;
pub mod scene;
pub mod terrain_compute;
pub mod voxel;
pub mod vulkan;
pub mod world;

#[cfg(feature = "app-harness")]
pub mod app;

#[cfg(feature = "app-harness")]
pub mod ui;
