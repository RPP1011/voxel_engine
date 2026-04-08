use crate::physics::PhysicsBody;
use crate::scene::cleanup::CleanupPolicy;
use crate::scene::config::SceneConfig;
use crate::scene::events::{CleanupReason, SceneEvent};
use crate::scene::handle::{ChunkHandle, EntityHandle};
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
    pub(crate) is_fragment: bool,
    pub(crate) is_at_rest: bool,
    pub(crate) spawn_order: u64,
    pub(crate) physics: Option<PhysicsBody>,
}

struct ChunkSlot {
    generation: u32,
    loaded: bool,
    position: IVec3,
    grid: VoxelGrid,
}

/// Slot-based generational entity storage.
pub struct Scene {
    entities: Vec<EntitySlot>,
    free_indices: Vec<u32>,
    config: SceneConfig,
    event_queue: Vec<SceneEvent>,
    pending_mutations: Vec<VoxelMutation>,
    pub(crate) next_spawn_order: u64,
    chunks: Vec<ChunkSlot>,
    free_chunk_indices: Vec<u32>,
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
            next_spawn_order: 0,
            chunks: Vec::new(),
            free_chunk_indices: Vec::new(),
        }
    }

    /// Spawn a new entity with the given voxel data, transform, and palette.
    /// Returns an opaque handle that can be used to refer to the entity later.
    pub fn spawn(&mut self, grid: &VoxelGrid, transform: Transform, palette: &MaterialPalette) -> EntityHandle {
        if let Some(index) = self.free_indices.pop() {
            let slot = &mut self.entities[index as usize];
            slot.generation += 1;
            slot.alive = true;
            slot.previous_transform = transform;
            slot.transform = transform;
            slot.grid = grid.clone();
            slot.palette = palette.clone();
            slot.is_fragment = false;
            slot.is_at_rest = false;
            slot.spawn_order = 0;
            slot.physics = None;
            EntityHandle { index, generation: slot.generation }
        } else {
            let index = self.entities.len() as u32;
            self.entities.push(EntitySlot {
                generation: 0,
                alive: true,
                previous_transform: transform,
                transform,
                grid: grid.clone(),
                palette: palette.clone(),
                is_fragment: false,
                is_at_rest: false,
                spawn_order: 0,
                physics: None,
            });
            EntityHandle { index, generation: 0 }
        }
    }

    /// Spawn a fragment entity (subject to automatic cleanup).
    pub fn spawn_fragment(&mut self, grid: &VoxelGrid, transform: Transform, palette: &MaterialPalette) -> EntityHandle {
        let handle = self.spawn(grid, transform, palette);
        let order = self.next_spawn_order;
        if let Some(slot) = self.get_alive_slot_mut(handle) {
            slot.is_fragment = true;
            slot.spawn_order = order;
        }
        self.next_spawn_order += 1;
        handle
    }

    /// Mark an entity as at-rest (eligible for limit-based eviction).
    pub fn mark_at_rest(&mut self, handle: EntityHandle) {
        if let Some(slot) = self.get_alive_slot_mut(handle) {
            slot.is_at_rest = true;
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
    /// Flushes pending voxel mutations, then runs cleanup, then any simulation logic.
    pub fn tick_sim(&mut self) {
        let mutations = std::mem::take(&mut self.pending_mutations);
        for mutation in mutations {
            self.apply_mutation(mutation);
        }
        self.run_cleanup();
    }

    fn run_cleanup(&mut self) {
        let mut to_remove: Vec<(EntityHandle, CleanupReason)> = Vec::new();

        // Check each fragment for cleanup criteria
        for (i, slot) in self.entities.iter().enumerate() {
            if !slot.alive || !slot.is_fragment { continue; }
            let handle = EntityHandle { index: i as u32, generation: slot.generation };

            if (slot.grid.count_nonempty() as u32) < self.config.cleanup.min_fragment_voxels {
                to_remove.push((handle, CleanupReason::TooSmall));
                continue;
            }
            if slot.transform.position.y < self.config.cleanup.kill_plane_y {
                to_remove.push((handle, CleanupReason::BelowKillPlane));
                continue;
            }
        }

        // Fragment limit: evict oldest at-rest
        let alive_fragments = self.entities.iter().filter(|s| s.alive && s.is_fragment).count();
        let already_removing = to_remove.len();
        let remaining = alive_fragments.saturating_sub(already_removing);

        if remaining > self.config.cleanup.max_fragments as usize {
            let excess = remaining - self.config.cleanup.max_fragments as usize;
            let mut candidates: Vec<(u32, u32, u64)> = self.entities.iter().enumerate()
                .filter(|(i, s)| s.alive && s.is_fragment && s.is_at_rest
                    && !to_remove.iter().any(|(h, _)| h.index == *i as u32))
                .map(|(i, s)| (i as u32, s.generation, s.spawn_order))
                .collect();
            candidates.sort_by_key(|&(_, _, order)| order);
            for &(index, generation, _) in candidates.iter().take(excess) {
                to_remove.push((EntityHandle { index, generation }, CleanupReason::FragmentLimitExceeded));
            }
        }

        // Apply removals and emit events
        for (handle, reason) in to_remove {
            self.despawn(handle);
            self.event_queue.push(SceneEvent::FragmentCleaned { handle, reason });
        }
    }

    /// Update the cleanup policy.
    pub fn set_cleanup_policy(&mut self, policy: CleanupPolicy) {
        self.config.cleanup = policy;
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

    // -------------------------------------------------------------------------
    // Physics body API (Task 13)
    // -------------------------------------------------------------------------

    /// Attach a physics body to an entity.
    pub fn set_physics(&mut self, handle: EntityHandle, body: PhysicsBody) {
        if let Some(slot) = self.get_alive_slot_mut(handle) {
            slot.physics = Some(body);
        }
    }

    /// Remove the physics body from an entity.
    pub fn remove_physics(&mut self, handle: EntityHandle) {
        if let Some(slot) = self.get_alive_slot_mut(handle) {
            slot.physics = None;
        }
    }

    /// Get a reference to the physics body of an entity.
    pub fn get_physics(&self, handle: EntityHandle) -> Option<&PhysicsBody> {
        self.entities.get(handle.index as usize)
            .filter(|s| s.generation == handle.generation && s.alive)
            .and_then(|s| s.physics.as_ref())
    }

    /// Apply a force (scaled by 1/mass) to an entity's linear velocity.
    pub fn apply_force(&mut self, handle: EntityHandle, force: Vec3) {
        if let Some(slot) = self.get_alive_slot_mut(handle) {
            if let Some(body) = &mut slot.physics {
                if !body.is_static && body.mass > 0.0 {
                    body.linear_velocity += force / body.mass;
                }
            }
        }
    }

    /// Apply an impulse (scaled by 1/mass) to an entity's linear velocity.
    pub fn apply_impulse(&mut self, handle: EntityHandle, impulse: Vec3) {
        if let Some(slot) = self.get_alive_slot_mut(handle) {
            if let Some(body) = &mut slot.physics {
                if !body.is_static && body.mass > 0.0 {
                    body.linear_velocity += impulse / body.mass;
                }
            }
        }
    }

    /// Apply a torque directly to angular velocity.
    pub fn apply_torque(&mut self, handle: EntityHandle, torque: Vec3) {
        if let Some(slot) = self.get_alive_slot_mut(handle) {
            if let Some(body) = &mut slot.physics {
                if !body.is_static {
                    body.angular_velocity += torque;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Chunk streaming API (Task 14)
    // -------------------------------------------------------------------------

    /// Load a chunk at a given world-space grid position.
    pub fn load_chunk(&mut self, pos: IVec3, grid: &VoxelGrid) -> ChunkHandle {
        if let Some(index) = self.free_chunk_indices.pop() {
            let slot = &mut self.chunks[index as usize];
            slot.generation += 1;
            slot.loaded = true;
            slot.position = pos;
            slot.grid = grid.clone();
            ChunkHandle { index, generation: slot.generation }
        } else {
            let index = self.chunks.len() as u32;
            self.chunks.push(ChunkSlot {
                generation: 0,
                loaded: true,
                position: pos,
                grid: grid.clone(),
            });
            ChunkHandle { index, generation: 0 }
        }
    }

    /// Unload a chunk. Silently ignores stale or already-unloaded handles.
    pub fn unload_chunk(&mut self, handle: ChunkHandle) {
        if let Some(slot) = self.chunks.get_mut(handle.index as usize) {
            if slot.generation == handle.generation && slot.loaded {
                slot.loaded = false;
                self.free_chunk_indices.push(handle.index);
            }
        }
    }

    /// Returns `true` if the chunk handle refers to a currently loaded chunk.
    pub fn is_chunk_loaded(&self, handle: ChunkHandle) -> bool {
        self.chunks.get(handle.index as usize)
            .is_some_and(|s| s.generation == handle.generation && s.loaded)
    }

    /// Number of currently loaded chunks.
    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.iter().filter(|s| s.loaded).count()
    }
}
