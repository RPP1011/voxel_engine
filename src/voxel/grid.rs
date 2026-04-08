/// Dense voxel grid stored as a flat Vec<u8>. Index 0 = air (empty).
#[derive(Clone)]
pub struct VoxelGrid {
    data: Vec<u8>,
    width: u32,
    height: u32,
    depth: u32,
}

impl VoxelGrid {
    pub fn new(width: u32, height: u32, depth: u32) -> Self {
        Self {
            data: vec![0u8; (width * height * depth) as usize],
            width,
            height,
            depth,
        }
    }

    pub fn dimensions(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.depth)
    }

    fn index(&self, x: u32, y: u32, z: u32) -> Option<usize> {
        if x < self.width && y < self.height && z < self.depth {
            Some((z * self.height * self.width + y * self.width + x) as usize)
        } else {
            None
        }
    }

    pub fn get(&self, x: u32, y: u32, z: u32) -> Option<u8> {
        self.index(x, y, z).map(|i| self.data[i])
    }

    pub fn set(&mut self, x: u32, y: u32, z: u32, value: u8) {
        if let Some(i) = self.index(x, y, z) {
            self.data[i] = value;
        }
    }

    pub fn count_nonempty(&self) -> usize {
        self.data.iter().filter(|&&v| v != 0).count()
    }

    pub fn bounding_box_of_nonempty(&self) -> Option<((u32, u32, u32), (u32, u32, u32))> {
        let mut min = (self.width, self.height, self.depth);
        let mut max = (0u32, 0u32, 0u32);
        let mut found = false;

        for z in 0..self.depth {
            for y in 0..self.height {
                for x in 0..self.width {
                    if self.get(x, y, z) != Some(0) {
                        found = true;
                        min.0 = min.0.min(x);
                        min.1 = min.1.min(y);
                        min.2 = min.2.min(z);
                        max.0 = max.0.max(x);
                        max.1 = max.1.max(y);
                        max.2 = max.2.max(z);
                    }
                }
            }
        }

        if found { Some((min, max)) } else { None }
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }
}
