use crate::scene::config::SceneConfig;
use crate::scene::events::SceneEvent;
use crate::scene::handle::EntityHandle;
use crate::scene::transform::Transform;
use crate::voxel::grid::VoxelGrid;
use crate::voxel::material::MaterialPalette;
use glam::{IVec3, Vec3};

enum VoxelMutation {
    DamageSphere { handle: EntityHandle, center: Vec3, radius: f32 },
    DamageBox { handle: EntityHandle, min: Vec3, max: Vec3 },
    SetVoxel { handle: EntityHandle, pos: IVec3, material: u8 },
}

pub(crate) struct EntitySlot {
    pub(crate) generation: u32,
    pub(crate) alive: bool,
    pub(crate) transform: Transform,
    pub(crate) previous_transform: Transform,
    pub(crate) grid: VoxelGrid,
    pub(crate) palette: MaterialPalette,
}

/// Slot-based generational entity storage.
pub struct Scene {
    entities: Vec<EntitySlot>,
    free_indices: Vec<u32>,
    config: SceneConfig,
    event_queue: Vec<SceneEvent>,
    pending_mutations: Vec<VoxelMutation>,
}

impl Scene {
    /// Create a scene without any GPU/windowing context (suitable for tests and
    /// headless simulation).
    pub fn new_headless(config: SceneConfig) -> Self {
        Self {
            entities: Vec::new(),
            free_indices: Vec::new(),
            config,
            event_queue: Vec::new(),
            pending_mutations: Vec::new(),
        }
    }

    /// Spawn a new entity with the given voxel data, transform, and palette.
    /// Returns an opaque handle that can be used to refer to the entity later.
    pub fn spawn(&mut self, grid: VoxelGrid, transform: Transform, palette: MaterialPalette) -> EntityHandle {
        if let Some(index) = self.free_indices.pop() {
            let slot = &mut self.entities[index as usize];
            slot.generation += 1;
            slot.alive = true;
            slot.previous_transform = transform;
            slot.transform = transform;
            slot.grid = grid;
            slot.palette = palette;
            EntityHandle { index, generation: slot.generation }
        } else {
            let index = self.entities.len() as u32;
            self.entities.push(EntitySlot {
                generation: 0,
                alive: true,
                previous_transform: transform,
                transform,
                grid,
                palette,
            });
            EntityHandle { index, generation: 0 }
        }
    }

    /// Remove an entity.  Silently ignores invalid or already-dead handles.
    pub fn despawn(&mut self, handle: EntityHandle) {
        if let Some(slot) = self.entities.get_mut(handle.index as usize) {
            if slot.alive && slot.generation == handle.generation {
                slot.alive = false;
                self.free_indices.push(handle.index);
            }
        }
    }

    /// Returns `true` if the handle refers to a live entity.
    pub fn is_alive(&self, handle: EntityHandle) -> bool {
        self.entities
            .get(handle.index as usize)
            .map_or(false, |s| s.alive && s.generation == handle.generation)
    }

    /// Update the world-space transform of an entity.
    /// Saves the current transform as `previous_transform` before updating,
    /// so that `interpolated_transform` can lerp between them.
    /// Silently ignores invalid or dead handles.
    pub fn set_transform(&mut self, handle: EntityHandle, transform: Transform) {
        if let Some(slot) = self.get_alive_slot_mut(handle) {
            slot.previous_transform = slot.transform;
            slot.transform = transform;
        }
    }

    /// Return a reference to the transform of a live entity, or `None`.
    pub fn get_transform(&self, handle: EntityHandle) -> Option<&Transform> {
        self.entities
            .get(handle.index as usize)
            .filter(|s| s.alive && s.generation == handle.generation)
            .map(|s| &s.transform)
    }

    /// Number of currently live entities.
    pub fn entity_count(&self) -> usize {
        self.entities.iter().filter(|s| s.alive).count()
    }

    /// Advance one simulation tick.
    /// Flushes pending voxel mutations, then runs any simulation logic.
    /// Transform snapshots for interpolation are recorded by `set_transform`.
    pub fn tick_sim(&mut self) {
        let mutations = std::mem::take(&mut self.pending_mutations);
        for mutation in mutations {
            self.apply_mutation(mutation);
        }
    }

    fn apply_mutation(&mut self, mutation: VoxelMutation) {
        match mutation {
            VoxelMutation::DamageSphere { handle, center, radius } => {
                if let Some(slot) = self.entities.get_mut(handle.index as usize) {
                    if slot.generation == handle.generation && slot.alive {
                        crate::voxel::destruction::remove_sphere(
                            &mut slot.grid,
                            (center.x as u32, center.y as u32, center.z as u32),
                            radius,
                        );
                    }
                }
            }
            VoxelMutation::DamageBox { handle, min, max } => {
                if let Some(slot) = self.entities.get_mut(handle.index as usize) {
                    if slot.generation == handle.generation && slot.alive {
                        crate::voxel::destruction::remove_box(
                            &mut slot.grid,
                            (min.x as u32, min.y as u32, min.z as u32),
                            (max.x as u32, max.y as u32, max.z as u32),
                        );
                    }
                }
            }
            VoxelMutation::SetVoxel { handle, pos, material } => {
                if let Some(slot) = self.entities.get_mut(handle.index as usize) {
                    if slot.generation == handle.generation && slot.alive {
                        slot.grid.set(pos.x as u32, pos.y as u32, pos.z as u32, material);
                    }
                }
            }
        }
    }

    /// Queue a sphere damage mutation to be applied on the next `tick_sim`.
    /// Silently ignores dead or invalid handles.
    pub fn damage_sphere(&mut self, handle: EntityHandle, center: Vec3, radius: f32) {
        if self.is_alive(handle) {
            self.pending_mutations.push(VoxelMutation::DamageSphere { handle, center, radius });
        }
    }

    /// Queue a box damage mutation to be applied on the next `tick_sim`.
    /// Silently ignores dead or invalid handles.
    pub fn damage_box(&mut self, handle: EntityHandle, min: Vec3, max: Vec3) {
        if self.is_alive(handle) {
            self.pending_mutations.push(VoxelMutation::DamageBox { handle, min, max });
        }
    }

    /// Queue a single-voxel set mutation to be applied on the next `tick_sim`.
    /// Silently ignores dead or invalid handles.
    pub fn set_voxel(&mut self, handle: EntityHandle, pos: IVec3, material: u8) {
        if self.is_alive(handle) {
            self.pending_mutations.push(VoxelMutation::SetVoxel { handle, pos, material });
        }
    }

    /// Return the number of non-empty voxels in the entity's grid, or `None` if dead.
    pub fn voxel_count(&self, handle: EntityHandle) -> Option<usize> {
        self.entities
            .get(handle.index as usize)
            .filter(|s| s.generation == handle.generation && s.alive)
            .map(|s| s.grid.count_nonempty())
    }

    /// Push an event onto the internal event queue.
    pub fn push_event(&mut self, event: SceneEvent) {
        self.event_queue.push(event);
    }

    /// Drain all pending events, clearing the queue.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, SceneEvent> {
        self.event_queue.drain(..)
    }

    /// Return an interpolated transform between the previous and current snapshots.
    /// `t = 0.0` returns the previous snapshot; `t = 1.0` returns the current transform.
    /// Returns `None` if the handle is invalid or the entity is dead.
    pub fn interpolated_transform(&self, handle: EntityHandle, t: f32) -> Option<Transform> {
        self.entities
            .get(handle.index as usize)
            .filter(|s| s.generation == handle.generation && s.alive)
            .map(|s| s.previous_transform.lerp(&s.transform, t))
    }

    /// Mutable access to a slot for a live entity.  Intended for internal use
    /// by sibling modules (physics integration, mutation API, etc.).
    pub(crate) fn get_alive_slot_mut(&mut self, handle: EntityHandle) -> Option<&mut EntitySlot> {
        self.entities
            .get_mut(handle.index as usize)
            .filter(|s| s.alive && s.generation == handle.generation)
    }
}
