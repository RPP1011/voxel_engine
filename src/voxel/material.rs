#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MaterialType {
    Air = 0,
    Wood = 1,
    Metal = 2,
    Glass = 3,
    Concrete = 4,
    Brick = 5,
    Stone = 6,
    Plite = 7,
    Dirt = 8,
    Sand = 9,
    Plastic = 10,
    Foliage = 11,
    Rubber = 12,
    Ice = 13,
}

impl MaterialType {
    pub fn density_kg_m3(self) -> f32 {
        match self {
            Self::Air => 0.0,
            Self::Wood => 500.0,
            Self::Metal => 7800.0,
            Self::Glass => 2500.0,
            Self::Concrete => 2400.0,
            Self::Brick => 1900.0,
            Self::Stone => 2600.0,
            Self::Plite => 2200.0,
            Self::Dirt => 1500.0,
            Self::Sand => 1600.0,
            Self::Plastic => 1200.0,
            Self::Foliage => 300.0,
            Self::Rubber => 1100.0,
            Self::Ice => 917.0,
        }
    }

    /// Max supportable stress in Pascals.
    pub fn strength_pa(self) -> f32 {
        match self {
            Self::Air => 0.0,
            Self::Wood => 40_000_000.0,       // 40 MPa
            Self::Metal => 250_000_000.0,     // 250 MPa
            Self::Glass => 10_000_000.0,      // 10 MPa (brittle)
            Self::Concrete => 30_000_000.0,   // 30 MPa
            Self::Brick => 20_000_000.0,      // 20 MPa
            Self::Stone => 60_000_000.0,      // 60 MPa
            Self::Plite => 25_000_000.0,      // 25 MPa
            Self::Dirt => 500_000.0,          // 0.5 MPa
            Self::Sand => 200_000.0,          // 0.2 MPa
            Self::Plastic => 50_000_000.0,    // 50 MPa
            Self::Foliage => 1_000_000.0,     // 1 MPa
            Self::Rubber => 15_000_000.0,     // 15 MPa
            Self::Ice => 5_000_000.0,         // 5 MPa
        }
    }

    /// Compressive strength in Pascals.
    pub fn compressive_strength_pa(self) -> f32 {
        match self {
            Self::Air => 0.0,
            Self::Wood => 40_000_000.0,
            Self::Metal => 250_000_000.0,
            Self::Glass => 100_000_000.0,     // Glass is strong in compression
            Self::Concrete => 30_000_000.0,
            Self::Brick => 20_000_000.0,
            Self::Stone => 60_000_000.0,
            Self::Plite => 25_000_000.0,
            Self::Dirt => 500_000.0,
            Self::Sand => 200_000.0,
            Self::Plastic => 50_000_000.0,
            Self::Foliage => 1_000_000.0,
            Self::Rubber => 15_000_000.0,
            Self::Ice => 5_000_000.0,
        }
    }

    /// Tensile strength in Pascals.
    pub fn tensile_strength_pa(self) -> f32 {
        match self {
            Self::Air => 0.0,
            Self::Wood => 30_000_000.0,       // Wood decent in tension
            Self::Metal => 800_000_000.0,     // High-strength structural steel
            Self::Glass => 7_000_000.0,       // Glass is brittle — very low tensile
            Self::Concrete => 3_000_000.0,    // Concrete weak in tension
            Self::Brick => 5_000_000.0,
            Self::Stone => 10_000_000.0,
            Self::Plite => 15_000_000.0,
            Self::Dirt => 100_000.0,
            Self::Sand => 0.0,                // Sand has no tensile strength
            Self::Plastic => 40_000_000.0,
            Self::Foliage => 500_000.0,
            Self::Rubber => 10_000_000.0,
            Self::Ice => 1_000_000.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PaletteEntry {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub roughness: u8,
    pub emissive: u8,
    pub material_type: MaterialType,
}

impl Default for PaletteEntry {
    fn default() -> Self {
        Self {
            r: 0,
            g: 0,
            b: 0,
            roughness: 0,
            emissive: 0,
            material_type: MaterialType::Air,
        }
    }
}

#[derive(Clone)]
pub struct MaterialPalette {
    entries: [PaletteEntry; 256],
}

impl MaterialPalette {
    pub fn new() -> Self {
        Self {
            entries: [PaletteEntry::default(); 256],
        }
    }

    pub fn get(&self, index: u8) -> &PaletteEntry {
        &self.entries[index as usize]
    }

    pub fn set(&mut self, index: u8, entry: PaletteEntry) {
        if index == 0 {
            return; // entry 0 is always Air
        }
        self.entries[index as usize] = entry;
    }

    /// Convert palette entries to a 256-element RGBA array for GPU upload.
    pub fn to_rgba(&self) -> [[u8; 4]; 256] {
        let mut rgba = [[0u8; 4]; 256];
        for i in 1..256 {
            let e = &self.entries[i];
            rgba[i] = [e.r, e.g, e.b, 255];
        }
        rgba
    }
}

/// Generate a default rainbow palette for scenes that don't have real palette data.
/// Index 0 is air (transparent black), indices 1-255 get HSV-rainbow colors.
pub fn default_palette() -> [[u8; 4]; 256] {
    let mut rgba = [[0u8; 4]; 256];
    for i in 1..256 {
        let hue = (i as f32 - 1.0) / 255.0 * 360.0;
        let (r, g, b) = hsv_to_rgb(hue, 0.7, 0.9);
        rgba[i] = [r, g, b, 255];
    }
    rgba
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    (
        ((r1 + m) * 255.0) as u8,
        ((g1 + m) * 255.0) as u8,
        ((b1 + m) * 255.0) as u8,
    )
}
