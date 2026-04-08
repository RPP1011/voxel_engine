/// Opaque generational handle for a scene entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityHandle {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

impl EntityHandle {
    /// The slot index.  Exposed for testing handle-reuse behaviour.
    pub fn slot_index(self) -> u32 {
        self.index
    }

    /// The generation counter.  Exposed for testing handle-reuse behaviour.
    pub fn generation(self) -> u32 {
        self.generation
    }
}

/// Opaque generational handle for a voxel chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkHandle {
    pub(crate) index: u32,
    pub(crate) generation: u32,
}
