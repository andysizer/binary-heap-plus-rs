use std::cmp::Ordering;
use std::panic;

use rand::distributions::{Distribution, Uniform};
use rand::thread_rng;

use stdext::function_name;

use gheap::*;

#[cfg(test)]

fn passed() {
    println!("ok");
}

fn test_parent_child<I: HeapIndexer>(idxer: &I, idx_type: &str, start_index: usize, n: usize) {
    assert!(start_index > 0);
    assert!(start_index <= std::usize::MAX - n);

    print!(
        "    {}<{}>(start_index={}, n={}) ... ",
        function_name!(),
        idx_type,
        start_index,
        n
    );

    for i in 0..n {
        let u = start_index + i;
        let v = idxer.get_child_index(u);
        if v < std::usize::MAX {
            assert!(v > u);
            let v = idxer.get_parent_index(v);
            assert!(v == u);
        }

        let v = idxer.get_parent_index(u);
        assert!(v < u);
        let v = idxer.get_child_index(v);
        assert!(v <= u && u - v < idxer.get_fanout());
    }

    passed();
}

fn test_is_heap<I: HeapIndexer + Copy>(idxer: &I, idx_type: &str, n: usize) {
    assert!(n > 0);

    print!("    {}<{}>(n={}) ... ", function_name!(), idx_type, n);

    let mut heap: GHeap<usize, MaxComparator, I> = GHeap::new_indexer(*idxer);

    heap.clear();
    for i in 0..n {
        heap.push(i);
    }
    assert!(heap.is_heap());

    heap.clear();
    for i in 0..n {
        heap.push(n - i);
    }
    assert!(heap.is_heap());

    heap.clear();
    for _i in 0..n {
        heap.push(n);
    }
    assert!(heap.is_heap());

    passed();
}

fn init_array(n: usize) -> Vec<usize> {
    let mut rng = thread_rng();
    let dist = Uniform::from(0..std::usize::MAX);

    dist.sample_iter(&mut rng).take(n).collect()
}

fn test_make_heap<I: HeapIndexer + Copy + Default>(idxer: &I, idx_type: &str, n: usize) {
    print!("    {}<{}>(n={}) ... ", function_name!(), idx_type, n);

    let v = init_array(n);
    let mut heap: GHeap<usize, MaxComparator, I> = GHeap::from_vec_indexer(v, *idxer);
    assert!(heap.is_heap());
    passed();
}

#[inline]
fn assert_sorted_asc(v: Vec<usize>) {
    let end = v.len();
    for i in 1..end {
        assert!(v[i] >= v[i - 1]);
    }
}

#[inline]
fn assert_sorted_desc(v: Vec<usize>) {
    let end = v.len();
    for i in 1..end {
        assert!(v[i] <= v[i - 1]);
    }
}

fn test_sort_heap<I: HeapIndexer + Copy + Default>(idxer: &I, idx_type: &str, n: usize) {
    print!("    {}<{}>(n={}) ... ", function_name!(), idx_type, n);

    let v = init_array(n);
    let heap: GHeap<usize, MaxComparator, I> = GHeap::from_vec_indexer(v, *idxer);
    let v = heap.into_sorted_vec();
    assert_sorted_asc(v);

    let v = init_array(n);
    let heap: GHeap<usize, MinComparator, I> =
        GHeap::from_vec_cmp_indexer(v, MinComparator {}, *idxer);
    let v = heap.into_sorted_vec();
    assert_sorted_desc(v);

    passed();
}

fn test_push_heap<I: HeapIndexer + Copy + Default>(idxer: &I, idx_type: &str, n: usize) {
    print!("    {}<{}>(n={}) ... ", function_name!(), idx_type, n);

    let v = init_array(n);

    let mut heap: GHeap<usize, MaxComparator, I> = GHeap::with_capacity_indexer(n, *idxer);
    for i in v {
        heap.push(i);
        assert!(heap.is_heap())
    }

    passed();
}
fn test_pop_heap<I: HeapIndexer + Copy + Default>(idxer: &I, idx_type: &str, n: usize) {
    print!("    {}<{}>(n={}) ... ", function_name!(), idx_type, n);

    let v = init_array(n);

    let mut heap: GHeap<usize, MaxComparator, I> = GHeap::from_vec_indexer(v, *idxer);
    assert!(heap.is_heap());

    let mut last = heap.pop();
    assert!(heap.is_heap());

    for _i in 0..n - 1 {
        let current = heap.pop();
        assert!(heap.is_heap());
        assert_ne!(last.cmp(&current), Ordering::Less);
        last = current;
    }

    passed();
}

fn test_func<I: HeapIndexer>(idxer: &I, idx_type: &str, func: fn(&I, idx_type: &str, usize)) {
    for i in 1..12 {
        func(idxer, idx_type, i);
    }
    func(idxer, idx_type, 1001);
}

macro_rules! test_all {
    ( $fanout:literal, $page_chunks:literal) => {
        paste::item! {
         #[test]
         fn [< test_indexer_ $fanout _ $page_chunks >] () {

            def_indexer!([< Indexer $fanout _ $page_chunks>], $fanout, $page_chunks);
            let idx = [< Indexer $fanout _ $page_chunks>] {};
            test_all(&idx, stringify!([< Indexer $fanout _ $page_chunks>]));

         }

        }
    };
}
fn test_all<I: HeapIndexer + Copy + Default + std::panic::RefUnwindSafe>(idx: &I, idx_type: &str) {
    let result = panic::catch_unwind(|| {
        let n = 1000000;
        test_parent_child(idx, idx_type, 1, n);
        test_parent_child(idx, idx_type, std::usize::MAX - n, n);

        test_func(idx, idx_type, test_is_heap::<I>);
        test_func(idx, idx_type, test_make_heap::<I>);
        test_func(idx, idx_type, test_sort_heap::<I>);
        test_func(idx, idx_type, test_push_heap::<I>);
        test_func(idx, idx_type, test_pop_heap::<I>);
    });
    if let Err(err) = result {
        println!("FAILED");
        panic::resume_unwind(err);
    }
}

test_all!(1, 1);
test_all!(2, 1);
test_all!(3, 1);
test_all!(4, 1);
test_all!(101, 1);

test_all!(1, 2);
test_all!(2, 2);
test_all!(3, 2);
test_all!(4, 2);
test_all!(101, 2);

test_all!(1, 3);
test_all!(2, 3);
test_all!(3, 3);
test_all!(4, 3);
test_all!(101, 3);

test_all!(1, 4);
test_all!(2, 4);
test_all!(4, 4);
test_all!(101, 4);

test_all!(1, 101);
test_all!(2, 101);
test_all!(3, 101);
test_all!(4, 101);
test_all!(101, 101);
