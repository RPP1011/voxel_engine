# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`voxel_engine` is a Rust **library crate** (no `[[bin]]` target, no `main()`) implementing a Vulkan (via `ash`) voxel/destruction game engine: GPU ray-marched voxel rendering, voxel-fragment physics, fluids (SPH/SWE), navmesh AI, and an optional windowed app harness. It has no `fn main()` anywhere in `src/` — a separate consumer project owns the window/event loop and drives the engine through the `App` trait (see `src/app/harness.rs`).

## Commands

- **Build (headless, default features):** `cargo build`
- **Build with the windowed app harness** (winit + egui, for consumers that render into a window): `cargo build --features app-harness`
- **Run all tests:** `cargo test`
- **Run one test file:** `cargo test --test <file_stem>` e.g. `cargo test --test voxel_grid`
- **Run one test by name:** `cargo test --test <file_stem> <test_fn_name>` e.g. `cargo test --test navmesh path_avoids_obstacles`
- **Run app-harness-gated tests:** `cargo test --features app-harness --test app_harness`

### Shader changes — required extra step

Shaders live in `shaders/*.{vert,frag,comp}` (GLSL) and are compiled to SPIR-V. The compiled `.spv` files are **committed** in `shaders/compiled/` and copied into `OUT_DIR` on every normal build — `shaderc` is *not* invoked by default. `build.rs` checks the mtime of every GLSL source against its committed `.spv` and **panics the build** if the source is newer (this replaced a bug where stale precompiled shaders were silently shipped, costing a multi-day debugging session).

After editing any file in `shaders/`:
```
cargo build --features compile-shaders   # recompiles GLSL -> SPIR-V via shaderc, rewrites shaders/compiled/*.spv
git add shaders/compiled/*.spv           # the .spv artifacts must be committed alongside the source edit
```
To bypass the freshness check (e.g. a snapshot/CI build that intentionally can't touch shader sources): set `VOXEL_ENGINE_SKIP_FRESHNESS_CHECK=1`. The check also self-disables automatically when the crate is consumed as a `cargo` git/registry dependency (detected via `CARGO_MANIFEST_DIR`).

### GPU requirements for tests

Most integration tests in `tests/` are pure-CPU logic (voxel grids, scene graph, physics math, navmesh, cleanup policy, etc.) and need no GPU. A handful create a real `VulkanContext` and therefore require an actual **discrete GPU** supporting Vulkan 1.3 with `VK_KHR_buffer_device_address`, `VK_KHR_synchronization2`, `VK_KHR_ray_tracing_pipeline`, and `VK_KHR_acceleration_structure` — these will fail on a machine without one:
`vulkan_instance`, `terrain_compute_smoke`, `gpu_harness`, `gpu_voxel_upload`, `memory_allocator`, `compute_pipeline`.

Crate edition is `2024` (Cargo.toml) — needs a Rust toolchain new enough to support it. No `rust-toolchain.toml`, `rustfmt.toml`, `clippy.toml`, CI config, `.cursor` rules, or Copilot instructions exist in this repo.

## Cargo features

- `default = []` — headless simulation core only (no window/UI deps).
- `app-harness` — pulls in `winit`, `egui`, `egui-winit`, `egui-ash-renderer`. Non-windowed consumers (e.g. a dedicated server or a sim harness) skip this and pay no cost.
- `compile-shaders` — makes `build.rs` invoke `shaderc` instead of copying precompiled SPIR-V (see above).
- `tracing` — enables the optional `tracing` crate dependency.

## Architecture

### Module map (`src/`)

- **`vulkan/`** — low-level Vulkan plumbing. `instance.rs` (`VulkanContext`: instance creation, discrete-GPU selection with the required extensions listed above, graphics/compute/transfer queues), `allocator.rs` (wraps `gpu-allocator`; `AllocatedBuffer`), `swapchain.rs`, `gbuffer.rs`, `shadow_map.rs`, `render_target.rs`, `graphics_pipeline.rs`, `compute.rs`, `occlusion.rs`, `sync.rs`, `debug.rs` (validation layers), `voxel_gpu.rs`, and `gpu_harness.rs` (generic taichi-style CPU↔GPU field/kernel dispatch abstraction — `create_field` / `upload` / `load_kernel` / `dispatch` / `download`). Note: `src/compute/harness.rs` is a near-duplicate of `src/vulkan/gpu_harness.rs` (same size); `compute::harness::GpuHarness` is the one re-exported as the public `compute` module API — check which one call sites actually use before editing either.
- **`terrain_compute.rs`** — the GPU terrain materialization pipeline. Owns a fixed pool of `NUM_SLOTS = 1024` GPU-resident chunk texture slots, each a 64³ `R8_UINT` 3D image plus three OR-downsampled mips (32³/16³/8³). Each slot carries its own descriptor set, command pool, and fence, and cycles through `Free → InFlight(request_id) → Loaded(ChunkKey)`, evicted LRU by `last_touched_frame` — eviction is refused if the LRU slot was touched this frame or last, to avoid evicting chunks still in the current render set. `submit_chunk_with_frame` dispatches `shaders/terrain_materialize.comp` against uploaded region-plan (`upload_region_plan`) and river polyline (`upload_rivers`) storage buffers, then `shaders/chunk_mip_downsample.comp` three times. The renderer samples slot images **directly** — no CPU readback in the hot path (the per-slot `readback_buffer` exists only for test/sync fallback paths). `NUM_SLOTS` must stay in sync with `POOL_SLOT_COUNT` in `render/renderer.rs`.
- **`voxel/`** — CPU-side voxel data. `grid.rs` (`VoxelGrid`: dense flat `Vec<u8>`, `0` = air), `svdag.rs` (Sparse Voxel DAG — deduplicated compressed octree, used for distant/streaming chunks), `mip.rs`, `destruction.rs` (`remove_sphere`/`remove_box` damage ops), `material.rs` (`MaterialType` enum with per-material density/strength, drives physics and destruction thresholds), `raycast.rs`, `vox_import.rs` (`.vox` file import via `dot_vox`). `splitting.rs`, `connectivity.rs`, `articulation.rs`, `cluster_graph.rs`, `coarse_narrow.rs`, `structural.rs` together form the structural-fragmentation pipeline: after damage, decide which parts of a voxel body are still connected and split disconnected pieces into new physics fragments (`physics::body_from_fragment`).
- **`scene/`** — `Scene` (`scene.rs`): a slot-based, generational-index store of entities (`EntityHandle{index, generation}` → `EntitySlot{transform, VoxelGrid, MaterialPalette, physics: Option<PhysicsBody>, is_fragment, is_at_rest, ...}`) plus a parallel slot array of loaded chunks (`ChunkHandle`). `Scene::new_headless(config)` builds a GPU/window-free scene and is what nearly every test in `tests/` constructs. `cleanup.rs` (`CleanupPolicy`: min fragment voxel count, rest timeout, kill-plane Y, max live fragments, fade-out, distance/offscreen limits — governs automatic fragment eviction), `events.rs` (`SceneEvent`/`CleanupReason` queue), `aabb.rs`, `transform.rs` (current + previous transform for interpolation), `config.rs`.
- **`world/`** — `chunks.rs`: `ChunkManager` tiers chunks by camera distance into `Active` (full-res, physics-enabled), `Streaming` (SVDAG, read-only), `Unloaded`. `spatial.rs`: world-level spatial queries/raycasts.
- **`physics/`** — `body.rs` (`PhysicsBody`: mass/velocity/restitution/friction), `avbd2d.rs` (Augmented Vertex Block Descent 2D solver), `body_from_fragment.rs` (build a rigid body from a disconnected voxel fragment), `cloth.rs`/`cloth_gpu.rs`, `culling.rs`, `graph_coloring.rs` (constraint graph coloring so the solver can process independent constraints in parallel), `shape_matching.rs`, `sleeping.rs` (rest-state tracking, feeds `Scene::mark_at_rest`/`CleanupPolicy`).
- **`fluid/`** — `sph.rs`/`sph_gpu.rs` (Smoothed Particle Hydrodynamics), `swe.rs`/`swe_gpu.rs` (Shallow Water Equations), `erosion.rs`.
- **`ai/`** — `navmesh.rs` (tile-based `NavMesh`: 16×16 XZ tiles built from `VoxelGrid` walkability, incremental rebuild, A* pathfinding), `navigation.rs`, `spatial.rs`, `structural_query.rs`, `tactical.rs`.
- **`camera/`** — `Camera`, `CameraController`/`InputState` traits, and `OrbitCamera`/`FreeCamera`/`FollowCamera` implementations. Anything the renderer draws from implements `render::RenderCamera` (a thin adapter trait implemented for the concrete camera types in `render/renderer.rs`).
- **`render/`** — `renderer.rs`: `VoxelRenderer`, the full deferred-pipeline orchestrator (GBuffer → ShadowMap → deferred lighting → tonemap/SSAO), owning descriptor pool/set caches keyed on image-view hashes to avoid per-frame Vulkan object churn, and double-buffered command buffers/fences/semaphores for 2-frame CPU/GPU pipelining. `config.rs`: `RendererConfig`, `DebugRenderMode`.
- **`app/`** (feature `app-harness`) — `App` trait (`setup`/`tick`/`on_input`) + `AppConfig`. This crate does **not** contain a winit event loop; a consumer implements `App` and drives it.
- **`ui/`** (feature `app-harness`) — `EguiState`: bridges `egui` + `egui-winit` + `egui-ash-renderer`. Per-frame lifecycle is documented at the top of `src/ui/mod.rs` (`handle_window_event` → `run` → `cmd_paint`). Paints in its own render pass with `LOAD_OP_LOAD` so it composites over whatever the voxel renderer already drew to the swapchain image.

### Rendering technique: GPU ray-marched voxels, not meshing

Voxel chunks are **not** converted to triangle meshes. For each visible chunk, a unit-cube mesh (`shaders/bbox.vert`) is rasterized to bound the chunk in screen space, and `shaders/gbuffer.frag` performs a per-fragment 3D DDA (Amanatides–Woo) ray march through the chunk's `R8_UINT` voxel texture (sampling coarser mips from `terrain_compute.rs`'s mip chain to skip empty space) to find the actual hit voxel and writes real `gl_FragDepth` for it. Early fragment tests are deliberately disabled in `gbuffer.frag` — see the comment block at its top explaining a depth-write-before-discard occlusion bug this avoids. `shaders/dda.vert`/`dda.frag`/`dda_heatmap.frag` are a related/debug ray-march path (the heatmap variant visualizes DDA step counts).

### Data flow for a chunk, end to end

1. `world::chunks::ChunkManager` tiers a chunk (`Active`/`Streaming`/`Unloaded`) by camera distance.
2. For an active chunk not yet resident, `terrain_compute::TerrainComputePipeline::submit_chunk_with_frame` claims a pool slot and dispatches the materialize + mip-downsample compute shaders.
3. Once the slot's fence signals, the slot is `Loaded`; `render::renderer::VoxelRenderer` samples its images directly during the GBuffer pass via the DDA ray march described above — there is no CPU-side voxel copy in this path.
4. Destructible/dynamic pieces instead go through `voxel::grid::VoxelGrid` (CPU dense data) inside a `scene::Scene` entity; damage (`voxel::destruction`) can disconnect parts of the grid, which `voxel::splitting`/`connectivity`/`articulation`/`cluster_graph` detect and split into new fragment entities with their own `physics::PhysicsBody` (`physics::body_from_fragment`), subject to `scene::cleanup::CleanupPolicy` eviction once at rest.
