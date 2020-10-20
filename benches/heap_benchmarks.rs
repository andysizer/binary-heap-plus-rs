#![allow(unused_parens)]
#![feature(trace_macros)]

trace_macros!(true);

use std::time::Duration;
use std::cmp::Ordering;
use std::ops::Sub;

use compare::Compare;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, BenchmarkId};

use rand::distributions::{Distribution, Standard, Uniform};

use rand::Rng;

use binary_heap_plus::BinaryHeap;

use gheap::*;

mod utils;
//use utils::memory_pressure::MemoryPressure;
use utils::prng::HeapRng;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Obj {
    w: i64,
    x: i64,
    y: i64,
    z: i64,
}

#[inline(always)]
fn abs_diff<T: Sub<Output = T> + Ord>(l: T, r: T) -> T {
    if l < r {
        r - l
    } else {
        l - r
    }
}
impl Ord for Obj {
    fn cmp(&self, other: &Self) -> Ordering {
        let s = abs_diff(abs_diff(self.w, self.x), abs_diff(self.y, self.z));
        let o = abs_diff(abs_diff(other.w, other.x), abs_diff(other.y, other.z));
        s.cmp(&o)
    }
}

impl PartialOrd for Obj {
    fn partial_cmp(&self, other: &Obj) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Distribution<Obj> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Obj {
        let (w, x, y, z) = rng.gen();
        Obj { w, x, y, z }
    }
}

trait Heap<T: Ord> {
    fn push(&mut self, t: T);
    fn pop(&mut self) -> Option<T>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

impl<T: Ord, C: Compare<T>> Heap<T> for BinaryHeap<T, C> {
    #[inline(always)]
    fn push(&mut self, t: T) {
        BinaryHeap::push(self, t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        BinaryHeap::pop(self)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        BinaryHeap::len(self)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        BinaryHeap::is_empty(self)
    }
}

impl<T: Ord, C: Compare<T>, I: Indexer> Heap<T> for GHeap<T, C, I> {
    #[inline(always)]
    fn push(&mut self, t: T) {
        GHeap::push(self, t);
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<T> {
        GHeap::pop(self)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        GHeap::len(self)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        GHeap::is_empty(self)
    }
}

fn init_vec<T>(n: usize) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let mut rng: HeapRng = HeapRng::new();

    Standard.sample_iter(&mut rng).take(n).collect()
}

#[inline(always)]
fn binary_heap_from_vec<T: Ord>(v: Vec<T>) -> BinaryHeap<T> {
    BinaryHeap::from_vec(v)
}

#[inline(always)]
fn gheap_from_vec<T, I>(v: Vec<T>, i: I) -> GHeap<T, MaxComparator, I>
where
    T: Ord,
    I: Indexer,
{
    GHeap::from_vec_indexer(v, i)
}

fn heap_push<T: Ord, H: Heap<T>>(v: Vec<T>, mut heap: H) {
    for e in v {
        heap.push(e);
    }
}

fn heap_pop<T: Ord, H: Heap<T>>(mut heap: H) {
    while !heap.is_empty() {
        heap.pop();
    }
}

fn heap_push_pop<T: Ord, H: Heap<T>>(mut v: Vec<T>, mut heap: H) {
    //let mut heap = GHeap::with_capacity_indexer(v.len(), i);

    let vlen = v.len();

    let push_rng: HeapRng = HeapRng::new();
    let push_range = Uniform::new_inclusive(1, vlen / 2);
    let mut push_iter = push_range.sample_iter(push_rng);

    let pop_rng: HeapRng = HeapRng::new();
    let pop_range = Uniform::new_inclusive(1, (vlen / 2) - (vlen / 10));
    let mut pop_iter = pop_range.sample_iter(pop_rng);

    loop {
        for _i in 0..push_iter.next().unwrap() {
            if !v.is_empty() {
                heap.push(v.pop().unwrap());
            } else {
                break;
            }
        }

        if v.is_empty() {
            while !heap.is_empty() {
                heap.pop();
            }
            break;
        } else {
            for _i in 0..pop_iter.next().unwrap() {
                if !heap.is_empty() {
                    heap.pop();
                } else {
                    break;
                }
            }
        }
    }
}

fn compute_size() -> usize {
    50_000_000
}

macro_rules! gheap_bench_init {
    ( $ID:ident, $FUN_NAME:ident, $TYPE:ty, $INDEXER:ident, $PARAM:ident, $FANOUT:literal, $PAGECHUNKS:literal )  => 
    {
        paste::item! {
            //let $ID = format!("{}_{}_{}_{}", stringify!([< gheap _ $FUN_NAME:snake _ $TYPE:snake _ >]), $PARAM, $FANOUT, $PAGECHUNKS );
            let $ID = stringify!([< gheap _ $FUN_NAME:snake _ $TYPE:snake _ $FANOUT _ $PAGECHUNKS>]);
            def_indexer!([< Indexer $FUN_NAME:camel $FANOUT _ $PAGECHUNKS >], $FANOUT, $PAGECHUNKS);
            let $INDEXER = [< Indexer $FUN_NAME:camel $FANOUT _ $PAGECHUNKS >] {};
        }
    }
}

const MEM_PRESSURE_SIZE: usize = 100_000_000;
//const HEAP_SIZES: [usize; 7] = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, MEM_PRESSURE_SIZE];
const HEAP_SIZES: [usize; 6] = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000];

macro_rules! def_group {
    ( $GROUP:ident, $FUN_NAME:ident, $TYPE:ty, $DATA:ident, $INDEXER:ident, $B_SETUP:tt , $B_LAMBDA:tt, $G_SETUP: tt, $G_LAMBDA:tt ) => {

        paste::item! {

            fn [<  bench _ $FUN_NAME:snake _ $TYPE:snake >] (c: &mut Criterion) {

                let mut group = c.benchmark_group(stringify!([< $GROUP >]));

                for i in HEAP_SIZES.iter() {

                    let mut size = *i;
                    let param : String;
                    if size == MEM_PRESSURE_SIZE {
                        size = compute_size();
                        param = String::from("pressure");
                    } else {
                        param = format!("{}", size);
                    }

                    let $DATA: Vec<$TYPE> = init_vec(size);

                    //let id = format!("{}_{}", stringify!([< binary_heap _ $FUN_NAME:snake _ $TYPE:snake>]), param);
                    let id = stringify!([< binary_heap _ $FUN_NAME:snake _ $TYPE:snake>]);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $B_SETUP, $B_LAMBDA, BatchSize::LargeInput));


                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 2, 1);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));


                    // Uncomment to check these typically unpromising examples
                    // gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 2, 2);
                    // group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                    //     |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));
                        
                    // gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 2, 4);
                    // group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                    //     |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));
                        

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 1);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 2);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    // Uncomment to check
                    // gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 3);
                    // group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                    //     |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));
                            

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 4);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 8);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 16);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 32);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));
                            

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 64);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 4, 128);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));    
    
                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 1);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 2);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    // Uncomment to check
                    // gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 3);
                    // group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                    //     |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));
                            

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 4);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 8);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 16);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 32);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));
                            

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 64);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));

                    gheap_bench_init!(id, $FUN_NAME, $TYPE, $INDEXER, param, 8, 128);
                    group.bench_with_input(BenchmarkId::new(id, &param), &$DATA,
                        |b, $DATA|  b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput));
                }
                group.finish();
            }

            criterion_group!{
                name = $GROUP; 
                config = Criterion::default().significance_level(0.1).sample_size(100).measurement_time(Duration::from_secs(300));
                targets = [<  bench _ $FUN_NAME:snake _ $TYPE:snake >]
            }
        }
    }
}

def_group!( from_vec_usize, from_vec, usize, data, indexer,
    (|| data.clone()),
    (|v| binary_heap_from_vec(v)),
    (|| data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( from_vec_obj, from_vec, Obj, data, indexer,
    (|| data.clone()),
    (|v| binary_heap_from_vec(v)),
    (|| data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( push_usize, push, usize, data, indexer,
    (|| {
        let v = data.clone();
        let h = BinaryHeap::<usize>::new();
        (v, h)
    }),
    (|t| {
        let (v, h) = t;
        heap_push(v, h);
    }),
    (|| {
        let v = data.clone();
        let tv: Vec<usize> = vec![];
        let h = gheap_from_vec(tv, indexer);
        (v, h)
    }),
    (|t| {
        let (v, h) = t;
        heap_push(v, h);
    })
);

def_group!( push_obj, push, Obj, data, indexer,
    (|| {
        let v = data.clone();
        let h = BinaryHeap::<Obj>::new();
        (v, h)
    }),
    (|t| {
        let (v, h) = t;
        heap_push(v, h);
    }),
    (|| {
        let v = data.clone();
        let tv: Vec<Obj> = vec![];
        let h = gheap_from_vec(tv, indexer);
        (v, h)
    }),
    (|t| {
        let (v, h) = t;
        heap_push(v, h);
    })
);


// fn criterion_benchmark(c: &mut Criterion) {
//     let num_items: u64 = 3200000;
//     let usize_vec: Vec<usize> = init_vec(num_items as usize);
//     let obj_vec: Vec<Obj> = init_vec(num_items as usize);

//     let setup_from_vec_usize = || usize_vec.clone();
//     let setup_from_vec_obj = || obj_vec.clone();

//     c.bench_function(
//         "GHeap<usize, MaxComparator, DefaultIndexer>::from_vec Obj 1000000",
//         |b| {
//             b.iter_batched(
//                 setup_from_vec_obj,
//                 |v| {
//                     gheap_from_vec(v, DefaultIndexer{})
//                 },
//                 BatchSize::LargeInput,
//             );
//         },
//     );

//     c.bench_function("BinaryHeap<usize>::from_vec Obj 1000000", |b| {
//         b.iter_batched(
//             setup_from_vec_obj,
//             |v| {
//                 //bench_from_vec_1::<BinaryHeap<Obj, MaxComparator>, Obj>(v);
//                 //let _h: BinaryHeap<Obj, MaxComparator> =  BinaryHeap::from_vec(v);
//                 binary_heap_from_vec(v)
//             },
//             BatchSize::LargeInput,
//         );
//     });

//     c.bench_function(
//         "GHeap<usize, MaxComparator, DefaultIndexer>::from_vec usize 1000000",
//         |b| {
//             b.iter_batched(
//                 setup_from_vec_usize,
//                 |v| {
//                     bench_from_vec::<GHeap<usize, MaxComparator, DefaultIndexer>, usize>(
//                         &GHeap::from_vec,
//                         v,
//                     );
//                 },
//                 BatchSize::LargeInput,
//             );
//         },
//     );

//     c.bench_function("BinaryHeap<usize>::from_vec usize 1000000",
//     |b| {
//         b.iter_batched(
//             setup_from_vec_usize,
//             |v| {
//                 bench_from_vec::<BinaryHeap<usize, MaxComparator>, usize>(
//                     &binary_heap_plus::BinaryHeap::from_vec,
//                     v,
//                 );
//             },
//             BatchSize::LargeInput,
//         );
//     });

//     let memory_pressure = MemoryPressure::new(num_items);

//     match memory_pressure {
//         Some(_) => {

//             c.bench_function(
//                 "MP GHeap<usize, MaxComparator, DefaultIndexer>::from_vec Obj 1000000",
//                 |b| {
//                     b.iter_batched(
//                         setup_from_vec_obj,
//                         |v| {
//                             bench_from_vec::<GHeap<Obj, MaxComparator, DefaultIndexer>, Obj>(
//                                 &GHeap::from_vec,
//                                 v,
//                             );
//                         },
//                         BatchSize::LargeInput,
//                     );
//                 },
//             );

//             c.bench_function("MP BinaryHeap<usize>::from_vec Obj 1000000",
//             |b| {
//                 b.iter_batched(
//                     setup_from_vec_obj,
//                     |v| {
//                         bench_from_vec::<BinaryHeap<Obj, MaxComparator>, Obj>(
//                             &binary_heap_plus::BinaryHeap::from_vec,
//                             v,
//                         );
//                     },
//                     BatchSize::LargeInput,
//                 );
//             });

//             c.bench_function(
//                 "MP GHeap<usize, MaxComparator, DefaultIndexer>::from_vec usize 1000000",
//                 |b| {
//                     b.iter_batched(
//                         setup_from_vec_usize,
//                         |v| {
//                             bench_from_vec::<GHeap<usize, MaxComparator, DefaultIndexer>, usize>(
//                                 &GHeap::from_vec,
//                                 v,
//                             );
//                         },
//                         BatchSize::LargeInput,
//                     );
//                 },
//             );

//             c.bench_function("MP BinaryHeap<usize>::from_vec usize 1000000",
//             |b| {
//                 b.iter_batched(
//                     setup_from_vec_usize,
//                     |v| {
//                         bench_from_vec::<BinaryHeap<usize, MaxComparator>, usize>(
//                             &binary_heap_plus::BinaryHeap::from_vec,
//                             v,
//                         );
//                     },
//                     BatchSize::LargeInput,
//                 );
//             });

//         }
//         None => {
//             eprintln!("Couldn't set up memeory pressure.");
//         }
//     };
// }


criterion_main!(from_vec_usize, from_vec_obj, push_usize, push_obj );
