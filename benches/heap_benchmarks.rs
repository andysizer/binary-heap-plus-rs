use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use rand::distributions::{Distribution, Standard};
use rand::Rng;

use binary_heap_plus::BinaryHeap;

use gheap::*;

mod utils;
use utils::memory_pressure::MemoryPressure;
use utils::prng::HeapRng;

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
        Obj { w, x, y, z }
    }
}

fn init_vec<T>(n: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let mut rng: HeapRng = HeapRng::new();

    Standard.sample_iter(&mut rng).take(n).collect()
}

fn bench_from_vec<H, T>(funk: &dyn Fn(Vec<T>) -> H, v: Vec<T>) {
    funk(v);
}

fn criterion_benchmark(c: &mut Criterion) {
    let num_items: u64 = 3200000;
    let usize_vec: Vec<usize> = init_vec(num_items as usize);

    let setup_from_vec_usize = || usize_vec.clone();

    {
        let memory_pressure = MemoryPressure::new(num_items);
        match memory_pressure {
            Some(_) => {
                c.bench_function("BinaryHeap<usize>::from_vec usize 1000000", |b| {
                    b.iter_batched(
                        setup_from_vec_usize,
                        |v| {
                            bench_from_vec::<BinaryHeap<usize, MaxComparator>, usize>(
                                &binary_heap_plus::BinaryHeap::from_vec,
                                v,
                            );
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
            None => {
                eprintln!("Couldn't set up memeory pressure.");
            }
        };
    }

    {
        let memory_pressure = MemoryPressure::new(num_items);
        match memory_pressure {
            Some(_) => {
                c.bench_function(
                    "GHeap<usize, MaxComparator, DefaultIndexer>::from_vec usize 1000000",
                    |b| {
                        b.iter_batched(
                        setup_from_vec_usize,
                        |v| {
                            bench_from_vec::<GHeap<usize, MaxComparator, DefaultIndexer>, usize>(
                                &GHeap::from_vec,
                                v,
                            );
                        },
                        BatchSize::LargeInput,
                    );
                    },
                );
            }
            None => {
                eprintln!("Couldn't set up memeory pressure.");
            }
        };
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
