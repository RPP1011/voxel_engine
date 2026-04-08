use crate::scene::config::SceneConfig;
use crate::scene::handle::EntityHandle;
use crate::scene::transform::Transform;
use crate::voxel::grid::VoxelGrid;
use crate::voxel::material::MaterialPalette;

pub(crate) struct EntitySlot {
    pub(crate) generation: u32,
    pub(crate) alive: bool,
    pub(crate) transform: Transform,
    pub(crate) grid: VoxelGrid,
    pub(crate) palette: MaterialPalette,
}

/// Slot-based generational entity storage.
pub struct Scene {
    entities: Vec<EntitySlot>,
    free_indices: Vec<u32>,
    config: SceneConfig,
}

impl Scene {
    /// Create a scene without any GPU/windowing context (suitable for tests and
    /// headless simulation).
    pub fn new_headless(config: SceneConfig) -> Self {
        Self {
            entities: Vec::new(),
            free_indices: Vec::new(),
            config,
        }
    }

    /// Spawn a new entity with the given voxel data, transform, and palette.
    /// Returns an opaque handle that can be used to refer to the entity later.
    pub fn spawn(&mut self, grid: VoxelGrid, transform: Transform, palette: MaterialPalette) -> EntityHandle {
        if let Some(index) = self.free_indices.pop() {
            let slot = &mut self.entities[index as usize];
            slot.generation += 1;
            slot.alive = true;
            slot.transform = transform;
            slot.grid = grid;
            slot.palette = palette;
            EntityHandle { index, generation: slot.generation }
        } else {
            let index = self.entities.len() as u32;
            self.entities.push(EntitySlot {
                generation: 0,
                alive: true,
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
    /// Silently ignores invalid or dead handles.
    pub fn set_transform(&mut self, handle: EntityHandle, transform: Transform) {
        if let Some(slot) = self.get_alive_slot_mut(handle) {
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

    /// Mutable access to a slot for a live entity.  Intended for internal use
    /// by sibling modules (physics integration, mutation API, etc.).
    pub(crate) fn get_alive_slot_mut(&mut self, handle: EntityHandle) -> Option<&mut EntitySlot> {
        self.entities
            .get_mut(handle.index as usize)
            .filter(|s| s.alive && s.generation == handle.generation)
    }
}
