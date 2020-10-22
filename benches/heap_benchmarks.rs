#![allow(unused_parens)]
// #![feature(trace_macros)]
// trace_macros!(true);

use std::time::Duration;
use std::cmp::Ordering;

use compare::Compare;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, BenchmarkId};

use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rand_distr::{ChiSquared};

use binary_heap_plus::BinaryHeap;

use gheap::*;

mod utils;
//use utils::memory_pressure::MemoryPressure;
use utils::prng::HeapRng;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct Obj {
    w: i64,
    x: i64,
    y: i64,
    z: i64,
}

impl Ord for Obj {
    fn cmp(&self, other: &Self) -> Ordering {
        self.w.cmp(&other.w)
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

impl<T: Ord, C: Compare<T>, I: HeapIndexer> Heap<T> for GHeap<T, C, I> {
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
    I: HeapIndexer,
{
    GHeap::from_vec_indexer(v, i)
}

const MEM_PRESSURE_SIZE: usize = 100_000_000;
const HEAP_SIZES: [usize; 7] = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, MEM_PRESSURE_SIZE];


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
                config = Criterion::default().significance_level(0.1).sample_size(100).measurement_time(Duration::from_secs(10));
                targets = [<  bench _ $FUN_NAME:snake _ $TYPE:snake >]
            }
        }
    }
}

// push the contents of the Vec v onto/into(?) heap
fn heap_push<T: Ord, H: Heap<T>>(v: Vec<T>, mut heap: H) {
    for e in v {
        heap.push(e);
    }
}

// pop n items from heap
fn heap_pop<T: Ord, H: Heap<T>>(mut n: usize, mut heap: H) {
    while n > 0 {
        heap.pop();
        n = n -1;
    }
}

// While Vec is not empty, push a (pseudo) random number (sampled from a ChiSquared distribution) 
// of items into/onto heap then pop a similarly selceted (psuedo) random number of items from heap.
// This is intended to approximate a plausible pattern of use. 
fn heap_push_pop<T: Ord, H: Heap<T>>(mut v: Vec<T>, mut heap: H) {

    // 'Empirically' k=1.80 seems to give a reasonable 'shape'
    let chi = ChiSquared::new(1.80).unwrap();
    let rng = HeapRng::new();
    
    let mut sample_iter = rng.sample_iter(chi);

    loop {
        let end = sample_iter.next().unwrap() as usize + 1;
        for _i in 0..end {
            if !v.is_empty() {
                heap.push(v.pop().unwrap());
            } else {
                break;
            }
        }

        if v.is_empty() {
            break;
        } else {
            let end = sample_iter.next().unwrap() as usize + 1;
            for _i in 0..end {
                if !heap.is_empty() {
                    heap.pop();
                } else {
                    break;
                }
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
        let size = data.len();
        let ten_pc = size / 10;
        let heap_vec = data[.. 9 * ten_pc].to_vec();
        let push_vec = data[9 * ten_pc ..].to_vec();
        let heap = binary_heap_from_vec(heap_vec);
        (push_vec, heap)
    }),
    (|t| {
        let (v, h) = t;
        heap_push(v, h);
    }),
    (|| {
        let size = data.len();
        let ten_pc = size / 10;
        let heap_vec = data[.. 9 * ten_pc].to_vec();
        let push_vec = data[9 * ten_pc ..].to_vec();
        let heap = gheap_from_vec(heap_vec, indexer);
        (push_vec, heap)
    }),
    (|t| {
        let (v, h) = t;
        heap_push(v, h);
    })
);

def_group!( push_obj, push, Obj, data, indexer,
    (|| {
        let size = data.len();
        let ten_pc = size / 10;
        let heap_vec = data[.. 9 * ten_pc].to_vec();
        let push_vec = data[9 * ten_pc ..].to_vec();
        let heap = binary_heap_from_vec(heap_vec);
        (push_vec, heap)
    }),
    (|t| {
        let (v, h) = t;
        heap_push(v, h);
    }),
    (|| {
        let size = data.len();
        let ten_pc = size / 10;
        let heap_vec = data[.. 9 * ten_pc].to_vec();
        let push_vec = data[9 * ten_pc ..].to_vec();
        let heap = gheap_from_vec(heap_vec, indexer);
        (push_vec, heap)
    }),
    (|t| {
        let (v, h) = t;
        heap_push(v, h);
    })
);

def_group!( pop_usize, pop, usize, data, indexer,
    (|| { 
        ( data.len() / 10 , binary_heap_from_vec(data.clone()))
    }),
    (|t| {
        let (n, h) = t;

        heap_pop(n, h);
    }),
    (|| {
        ( data.len() / 10 , gheap_from_vec(data.clone(), indexer) )
    }),
    (|t| {
        let (n, h) = t;
        heap_pop(n, h);
    })
);

def_group!( pop_obj, pop, Obj, data, indexer,
    (|| { 
        ( data.len() / 10 , binary_heap_from_vec(data.clone()))
    }),
    (|t| {
        let (n, h) = t;

        heap_pop(n, h);
    }),
    (|| {
        ( data.len() / 10 , gheap_from_vec(data.clone(), indexer) )
    }),
    (|t| {
        let (n, h) = t;
        heap_pop(n, h);
    })
);

def_group!( push_pop_usize, push_pop, usize, data, indexer,
    (|| {
        let size = data.len();
        let ten_pc = size / 10;
        let heap_vec = data[.. 9 * ten_pc].to_vec();
        let push_vec = data[9 * ten_pc ..].to_vec();
        let heap = binary_heap_from_vec(heap_vec);
        (push_vec, heap)
    }),
    (|t| {
        let (v, h) = t;
        heap_push_pop(v, h);
    }),
    (|| {
        let size = data.len();
        let ten_pc = size / 10;
        let heap_vec = data[.. 9 * ten_pc].to_vec();
        let push_vec = data[9 * ten_pc ..].to_vec();
        let heap = gheap_from_vec(heap_vec, indexer);
        (push_vec, heap)
    }),
    (|t| {
        let (v, h) = t;
        heap_push_pop(v, h);
    })
);

def_group!( push_pop_obj, push_pop, Obj, data, indexer,
    (|| {
        let size = data.len();
        let ten_pc = size / 10;
        let heap_vec = data[.. 9 * ten_pc].to_vec();
        let push_vec = data[9 * ten_pc ..].to_vec();
        let heap = binary_heap_from_vec(heap_vec);
        (push_vec, heap)
    }),
    (|t| {
        let (v, h) = t;
        heap_push_pop(v, h);
    }),
    (|| {
        let size = data.len();
        let ten_pc = size / 10;
        let heap_vec = data[.. 9 * ten_pc].to_vec();
        let push_vec = data[9 * ten_pc ..].to_vec();
        let heap = gheap_from_vec(heap_vec, indexer);
        (push_vec, heap)
    }),
    (|t| {
        let (v, h) = t;
        heap_push_pop(v, h);
    })
);

criterion_main!(from_vec_usize, from_vec_obj, push_usize, push_obj, pop_usize, pop_obj, push_pop_usize, push_pop_obj );
