use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rand::{Rng};
use rand::distributions::{Distribution, Standard};
use rand::SeedableRng;
use rand_core::{RngCore, Error, impls};

use binary_heap_plus::BinaryHeap;

use gheap::*;

// Set up a PRNG - we want a PRNG because we want the same sequence of 'random' numbers in our tests.
const SEED_SIZE: usize = 4;
pub struct HeapRngSeed {
    pub data:   [u32; SEED_SIZE],
}

pub struct HeapRng(HeapRngSeed);

impl Default for HeapRngSeed {
    fn default() -> HeapRngSeed {
        HeapRngSeed{ data: [0x193a6754, 0xa8a7d469, 0x97830e05, 0x113ba7bb]}
    }
}

impl AsMut<[u8]> for HeapRngSeed {
    fn as_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::mem::transmute::<&mut [u32], &mut [u8]>(&mut self.data)
        }
        
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
        let x = self.0.data[SEED_SIZE-1];
        let t = x ^ (x << 11);
        self.0.data[SEED_SIZE-1] = self.0.data[SEED_SIZE-2]; // x = ...
        self.0.data[SEED_SIZE-2] = self.0.data[SEED_SIZE-3]; // y = ...
        let w = self.0.data[SEED_SIZE-4]; 
        self.0.data[SEED_SIZE-3] =  w; // z = ...
        self.0.data[SEED_SIZE-4] = w ^ (w >> 19) ^ (t ^ (t >> 8));
        self.0.data[SEED_SIZE-4]
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
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        Ok(self.fill_bytes(dest))
    }
}

impl HeapRng {
    fn new() -> HeapRng {
        HeapRng(HeapRngSeed::default())
    }
}
#[derive(Debug)]
struct Obj {
    w: usize,
    x: usize,
    y: usize,
    z: usize,
}

impl Distribution<Obj> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Obj {
        let (w, x, y, z) = rng.gen();
        Obj {w, x, y, z}
    }
}

fn init_array<T>(n: usize) -> Vec<T> 
where
Standard: Distribution<T>
{
    let mut rng: HeapRng = HeapRng::new();

    Standard.sample_iter(&mut rng).take(n).collect()
}
fn criterion_benchmark(c: &mut Criterion) {

    let usize_vec: Vec<usize> = init_array(250000000);
    //let usize_vec1: Vec<usize> = init_array(1000000);
    //if usize_vec != usize_vec1 {
    //    println!("ftw!!!")
    //}

    //c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));

    c.bench_function("GHeap<usize, MaxComparator, DefaultIndexer>::from_vec usize 1000000", 
        |b| {
            
            b.iter(|| {
                let working = usize_vec.clone();
                let _heap: GHeap<usize, MaxComparator, DefaultIndexer> = GHeap::from_vec(working);
            });
    });

    c.bench_function("BinaryHeap<usize>::from_vec usize 1000000", 
        |b| {
            
            b.iter(|| {
                let working = usize_vec.clone();
                let _heap: BinaryHeap<usize, MaxComparator> = binary_heap_plus::BinaryHeap::from_vec(working);
            });
    });

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);