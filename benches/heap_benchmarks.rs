// #![feature(trace_macros)]

// trace_macros!(true);
#![allow(unused_parens)]

use std::ops::Sub;
use std::cmp::Ordering;

use compare::Compare;


use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use rand::distributions::{Distribution, Standard, Uniform};

use rand::Rng;

use binary_heap_plus::BinaryHeap;

use gheap::*;

mod utils;
use utils::memory_pressure::MemoryPressure;
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
    fn push(&mut self,t: T);
    fn pop(&mut self) -> Option<T>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

impl <T: Ord, C: Compare<T>> Heap<T> for BinaryHeap<T, C> {
    #[inline(always)]
    fn push(&mut self,t: T) {
        BinaryHeap::push(self,t);
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

impl <T: Ord, C: Compare<T>, I: Indexer> Heap<T> for GHeap<T, C, I> {
    #[inline(always)]
    fn push(&mut self,t: T) {
        GHeap::push(self,t);
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
    I: Indexer 
{
    GHeap::from_vec_indexer(v, i)
}


fn heap_push<T: Ord, H: Heap<T>>(v: Vec<T>, mut heap: H) {
    for e in v {
        heap.push(e);
    }
}

fn heap_pop<T: Ord, H: Heap<T>>( mut heap: H) {
    while ! heap.is_empty() {
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
            if ! v.is_empty() {
                heap.push(v.pop().unwrap());
            } else {
                break;
            }
        }

        if v.is_empty() {
            while ! heap.is_empty() {
                heap.pop();
            }
            break;
        } else {
            for _i in 0..pop_iter.next().unwrap() {
                if ! heap.is_empty() {
                    heap.pop();
                } else {
                    break;
                }
            }
        }
    }
}

macro_rules! def_group {
    ( $GROUP:ident, $FUN_NAME:ident, $TYPE:ty, $SIZE:literal, $DATA:ident, $INDEXER:ident, $B_SETUP:tt , $B_LAMBDA:tt, $G_SETUP: tt, $G_LAMBDA:tt ) => {

        paste::item! {

            fn [< binary_heap _ $FUN_NAME _ $TYPE:snake _ $SIZE >] (c: &mut Criterion) {
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< binary_heap _ $FUN_NAME _ $TYPE:snake _ $SIZE >]), 
                    |b| {
                        b.iter_batched( $B_SETUP, $B_LAMBDA, BatchSize::LargeInput );
                    }
                );
            }

            fn [< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "2_1" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "2_1" >], 2, 1);

                let $INDEXER = [< Indexer $FUN_NAME:camel "2_1" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "2_1" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }
       
            fn [< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "2_2" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "2_2" >], 2, 2);

                let $INDEXER = [< Indexer $FUN_NAME:camel "2_2" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "2_2" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }
            
            fn [< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "2_4" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "2_4" >], 2, 4);

                let $INDEXER = [< Indexer $FUN_NAME:camel "2_4" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "2_4" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

            fn [< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_1" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_1" >], 4, 1);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_1" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_1" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

            fn [< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_2" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_2" >], 4, 2);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_2" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_2" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

            fn [< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_3" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_3" >], 4, 3);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_3" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_3" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

            fn [< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_4" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_4" >], 4, 4);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_4" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_4" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

        
            fn [< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_8" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_8" >], 4, 8);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_8" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _$TYPE:snake _ $SIZE _ "4_8" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

            fn [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_16" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_16" >], 4, 16);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_16" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_16" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }
 
            fn [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_32" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_32" >], 4, 32);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_32" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_32" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

            fn [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_64" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_64" >], 4, 64);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_64" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_64" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

            fn [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_128" >](c: &mut Criterion) {

                def_indexer!([< Indexer $FUN_NAME:camel "4_128" >], 4, 128);

                let $INDEXER = [< Indexer $FUN_NAME:camel "4_128" >] {};
                let $DATA: Vec<$TYPE> = init_vec($SIZE);
                c.bench_function(
                    stringify!([< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_128" >]),
                    |b| {
                        b.iter_batched( $G_SETUP, $G_LAMBDA, BatchSize::LargeInput);
                    }
                );
            }

            criterion_group!($GROUP, 
                [< binary_heap _ $FUN_NAME _ $TYPE:snake _ $SIZE >],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "2_1">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "2_2">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "2_4">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_1">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_2">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_3">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_4">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_8">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_16">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_32">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_64">],
                [< gheap _ $FUN_NAME _ $TYPE:snake _ $SIZE _ "4_128">],
            );
        }
    }
}



def_group!( group1, 
    from_vec, 
    usize, 
    100, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group2, 
    from_vec, 
    usize, 
    1000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group3, 
    from_vec, 
    usize, 
    10000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group4, 
    from_vec, 
    usize, 
    100000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group5, 
    from_vec, 
    usize, 
    1000000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group6, 
    from_vec, 
    usize, 
    10000000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group7, 
    from_vec, 
    usize, 
    100000000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group8, 
    from_vec, 
    Obj, 
    100, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group9, 
    from_vec, 
    Obj, 
    1000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group10, 
    from_vec, 
    Obj, 
    10000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group11, 
    from_vec, 
    Obj, 
    100000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group12, 
    from_vec, 
    Obj, 
    1000000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group13, 
    from_vec, 
    Obj, 
    10000000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

def_group!( group14, 
    from_vec, 
    Obj, 
    100000000, 
    data, 
    indexer, 
    ( || data.clone()),
    ( |v| binary_heap_from_vec(v)),
    ( || data.clone()),
    (|v| gheap_from_vec(v, indexer))
);

 
def_group!( group15, 
    push, 
    usize, 
    100, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<usize>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<usize> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group16, 
    push, 
    usize, 
    1000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<usize>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<usize> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group17, 
    push, 
    usize, 
    10000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<usize>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<usize> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group18, 
    push, 
    usize, 
    100000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<usize>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<usize> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group19, 
    push, 
    usize, 
    1000000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<usize>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<usize> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group20, 
    push, 
    usize, 
    10000000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<usize>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<usize> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group21, 
    push, 
    usize, 
    100000000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<usize>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<usize> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group22, 
    push, 
    Obj, 
    100, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<Obj>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<Obj> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group23, 
    push, 
    Obj, 
    1000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<Obj>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<Obj> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group24, 
    push, 
    Obj, 
    10000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<Obj>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<Obj> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group25, 
    push, 
    Obj, 
    100000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<Obj>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<Obj> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group26, 
    push, 
    Obj, 
    1000000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<Obj>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<Obj> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group27, 
    push, 
    Obj, 
    10000000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<Obj>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<Obj> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
);

def_group!( group28, 
    push, 
    Obj, 
    100000000, 
    data, 
    indexer, 
    (|| {
            let v = data.clone();
            let h = BinaryHeap::<Obj>::new();
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    ),
    (|| {
            let v = data.clone();
            let tv: Vec<Obj> = vec![];
            let h = gheap_from_vec(tv, indexer);
            (v, h)
        }
    ),
    ( |t| {
            let (v, h) = t;
            heap_push(v, h);
        }
    )
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

//criterion_group!(benches, criterion_benchmark);
criterion_main!(group1, group2, group3, group4, group5, group6, group7,
                group8, group9, group10, group11, group12, group13, group14,
                group15, group16, group17, group18, group19, group20, group21,
                group22, group23, group24, group25, group26, group27, group28) ;
