
pub mod heap_rng {

    use rand::SeedableRng;
    use rand_core::{impls, Error as RngError, RngCore};

    // Set up a PRNG - we want a PRNG because we want the same sequence of 'random' numbers in our tests.
    const SEED_SIZE: usize = 4;
    pub struct HeapRngSeed {
        pub data: [u32; SEED_SIZE],
    }

    pub struct HeapRng(HeapRngSeed);

    impl Default for HeapRngSeed {
        fn default() -> HeapRngSeed {
            HeapRngSeed {
                data: [0x193a6754, 0xa8a7d469, 0x97830e05, 0x113ba7bb],
            }
        }
    }

    impl AsMut<[u8]> for HeapRngSeed {
        fn as_mut(&mut self) -> &mut [u8] {
            unsafe { std::mem::transmute::<&mut [u32], &mut [u8]>(&mut self.data) }
        }
    }

    impl SeedableRng for HeapRng {
        type Seed = HeapRngSeed;

        fn from_seed(seed: HeapRngSeed) -> HeapRng {
            HeapRng(seed)
        }
    }

    impl RngCore for HeapRng {
        #[inline]
        fn next_u32(&mut self) -> u32 {
            let x = self.0.data[SEED_SIZE - 1];
            let t = x ^ (x << 11);
            self.0.data[SEED_SIZE - 1] = self.0.data[SEED_SIZE - 2]; // x = ...
            self.0.data[SEED_SIZE - 2] = self.0.data[SEED_SIZE - 3]; // y = ...
            let w = self.0.data[SEED_SIZE - 4];
            self.0.data[SEED_SIZE - 3] = w; // z = ...
            self.0.data[SEED_SIZE - 4] = w ^ (w >> 19) ^ (t ^ (t >> 8));
            self.0.data[SEED_SIZE - 4]
        }

        #[inline]
        fn next_u64(&mut self) -> u64 {
            impls::next_u64_via_u32(self)
        }

        #[inline]
        fn fill_bytes(&mut self, dest: &mut [u8]) {
            impls::fill_bytes_via_next(self, dest)
        }

        #[inline]
        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), RngError> {
            Ok(self.fill_bytes(dest))
        }
    }

    impl HeapRng {
        pub fn new() -> HeapRng {
            HeapRng(HeapRngSeed::default())
        }
    }
}
