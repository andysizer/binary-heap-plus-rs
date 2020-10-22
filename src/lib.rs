//! This crate provides `GHeap` which implements a n-ary heap in which subtrees containing k items are laid out contiguously 
//! in a 'chunk'. `GHeap` can be used to implement a [Priority Queue](https://www.wikiwand.com/en/Priority_queue#:~:text=In%20computer%20science%2C%20a%20priority,an%20element%20with%20low%20priority.)
//! in a similar fashion to `std::collections::BinaryHeap`.
//! 
//! This implementation was motivated by the article ['You're Doing It Wrong'](https://queue.acm.org/detail.cfm?id=1814327) which
//! introduces the [B-heap](https://www.wikiwand.com/en/B-heap). The main idea is that laying out subtrees contiguously will tend to increase
//! locality when the heap is traversed using the parent-child relation and that he benefits of increased locality will tend to outweigh
//! the increased cost of computing the parent-child relation when memory is under pressure due to the cost of paging.
//! 
//! The `benches` directory contains benchmarks to test this idea. These appear to show that most of the time any improvement shown by `GHeap` over 
//! `BinaryHeap` is due to 'fanout' (n-ariness) rather than chunking. However there appear to be 'sweet spots' where a judicious choice of fanout and
//! pagechunk size for items of a given size can show significant improvement in performance, implying that for certain use cases GHeap might be a 
//! reasonable replacement for `std::collections::BinaryHeap` (or a simpler implementation of an n-ary heap).
//! 
//! Insertion and popping the largest element from a GHeap with a fanout (arity) of 'd' have `O(log n / log d)` 
//! time complexity. Checking the largest element is `O(1)`. Converting 
//! a vector to a GHeap can be done in-place, and has `O(n)` complexity. A GHeap can also be
//! converted to a sorted vector in-place, allowing it to be used for an `O(n
//! log n)` in-place heapsort.
//!
//! 
//! This implementation is derived from:
//! 1. [binary-heap-plus](https://github.com/sekineh/binary-heap-plus-rs)
//!    which is itself forked from the version of BinaryHeap in the Rust standard library.
//! 2. The C++ version of [gheap](https://github.com/valyala/gheap.git)
//!
//! The main difference between a binary heap and a B-heap are the equations used
//! to compute the indices of parent and child nodes in the underlying representatoin of the heap. `GHeap` abstracts these computations out into
//! a trait: `HeapIndexer`. The primary reason for this is Rust's lack of support for `const generics` in stable builds (i.e. rustc will reject 'GHeap<i32, 4, 2>'). 
//! This may change at some point in the not too distant future (see [Shipping Const Generics in 2020](https://without.boats/blog/shipping-const-generics/)) as
//! usable implementation of `const generics` is now available in nightly builds.
//! 
//! The additional features that `GHeap` has over and above those of `std::collections::BinaryHeap` are due to [binary-heap-plus](https://github.com/sekineh/binary-heap-plus-rs)
//! and include:
//! 
//! * Heaps other than max heap e.g. min heap and heaps constructed using custom comparators.
//! * Support for serialization and deserialization if `serde` is enabled.
//!
//! A utility macro 'def_indexer' is provided to make defining indexers with different 'fanouts' and 'pagechunks' trivial.
//! 
//! # Quick start
//!
//! ## Max/Min Heap
//!
//! `GHeap::from_vec()` is the easiest way to create a max heap from a vector.
//!
//! ```rust
//!     use gheap::*;
//!
//!     // max heap
//!     let mut h: GHeap<i32> = GHeap::from_vec(vec![]);
//!     // max heap with initial capacity
//!     let mut h: GHeap<i32> = GHeap::from_vec(Vec::with_capacity(16));
//!     // max heap from iterator
//!     let mut h: GHeap<i32> = GHeap::from_vec((0..42).collect());
//!     assert_eq!(h.pop(), Some(41));
//! ```
//!
//! A min heap can be created in a similar fashion through type annotation.
//!
//! ```rust
//!     use gheap::*;
//!
//!     // min heap
//!     let mut h: GHeap<i32, MinComparator> = GHeap::from_vec(vec![]);
//!     // min heap with initial capacity
//!     let mut h: GHeap<i32, MinComparator> = GHeap::from_vec(Vec::with_capacity(16));
//!     // min heap from iterator
//!     let mut h: GHeap<i32, MinComparator> = GHeap::from_vec((0..42).collect());
//!     assert_eq!(h.pop(), Some(0));
//! ```
//!
//! ## Custom Heap
//!
//! `GHeap::from_vec_cmp()` takes a comparator closure to produce a custom heap.
//!
//! ```rust
//!     use gheap::*;
//!
//!     // custom heap: ordered by second value (_.1) of the tuples; min first
//!     let mut h = GHeap::from_vec_cmp( 
//!         vec![(1, 5), (3, 2), (2, 3)],
//!         |a: &(i32, i32), b: &(i32, i32)| b.1.cmp(&a.1), // comparator closure here
//!     );
//!     assert_eq!(h.pop(), Some((3, 2)));
//! ```
//! ## Custom Indexer
//!
//! `GHeap::from_vec_indexer()` constructs a heap with a custom indexer.
//!
//! ```rust
//!     use gheap::*;
//!
//!     // define ThreeTwoIndexer to be a HeapIndexer that uses a fanout of 3 and pagechunk size of 2.
//!     def_indexer!(ThreeTwoIndexer, 3,2);
//!     let mut h = GHeap::from_vec_indexer(
//!         vec![1,5,3], 
//!         ThreeTwoIndexer{}, // supply the indexer
//!     );
//!     assert_eq!(h.pop(), Some(5));
//! ```
//!
//! # Construction
//!
//! ## Generic methods to create different kind of heaps from initial `vec` data.
//!
//! * `GHeap::from_vec(vec)`
//! * `GHeap::from_vec_cmp(vec, cmp)`
//! * `GHeap::from_vec_indexer(vec, indexer)`
//! * `GHeap::from_vec_cmp_indexer(vec, cmp, indexer)`
//!
//! ```
//! use gheap::*;
//!
//! // max heap (default)
//! let mut heap: GHeap<i32> = GHeap::from_vec(vec![1,5,3]);
//! assert_eq!(heap.pop(), Some(5));
//!
//! // min heap
//! let mut heap: GHeap<i32, MinComparator> = GHeap::from_vec(vec![1,5,3]);
//! assert_eq!(heap.pop(), Some(1));
//!
//! // custom-sort heap
//! let mut heap = GHeap::from_vec_cmp(vec![1,5,3], |a: &i32, b: &i32| b.cmp(a));
//! assert_eq!(heap.pop(), Some(1));
//!
//! // custom-key heap
//! let mut heap = GHeap::from_vec_cmp(vec![6,3,1], KeyComparator(|k: &i32| k % 4));
//! assert_eq!(heap.pop(), Some(3));
//!
//! // TIP: How to reuse a comparator
//! let mod4_comparator = KeyComparator(|k: &_| k % 4);
//! let mut heap1 = GHeap::from_vec_cmp(vec![6,3,1], mod4_comparator);
//! assert_eq!(heap1.pop(), Some(3));
//! let mut heap2 = GHeap::from_vec_cmp(vec![2,4,1], mod4_comparator);
//! assert_eq!(heap2.pop(), Some(2));
//! 
//! // max heap with a customized indexer
//! def_indexer!(ThreeTwoIndexer, 3,2);
//! let idxer = ThreeTwoIndexer{}; 
//! let mut heap = GHeap::from_vec_indexer(
//!     vec![1,5,3], 
//!     idxer
//! );
//! assert_eq!(heap.pop(), Some(5));
//! 
//! // custom-key heap with custom indexer
//! let mut heap = GHeap::from_vec_cmp_indexer(
//!     vec![6,3,1],
//!     mod4_comparator, 
//!     idxer
//! );
//! assert_eq!(heap.pop(), Some(3));
//! ```
//!
//! ## Dedicated methods to create different kind of heaps
//!
//! * `GHeap::new()` creates a max heap.
//! * `GHeap::with_capacity()` creates a max heap with the requested capacity.
//! * `GHeap::new_indexer()` creates a max heap with the supplied indexer.
//! * `GHeap::with_capacity_indexer()` creates a max heap with the requested capacity and supplied indexer.
//! * `GHeap::new_min()` creates a min heap.
//! * `GHeap::new_min_indexer()` creates a min heap with the supplied indexer.
//! * `GHeap::with_capacity_min()` creates a min heap with the requested capacity.
//! * `GHeap::with_capacity_min_indexer()` creates a min heap with the requested capacity and supplied indexer.
//! * `GHeap::new_by()` creates a heap ordered by the given closure.
//! * `GHeap::new_by_indexer()` creates a heap with the supplied indexer and ordered by the given closure.
//! * `GHeap::with_capacity_by()` creates a heap with the requested capacity ordered by the given closure.
//! * `GHeap::with_capacity_by_indexer()` creates a heap with the requested capacity and supplied indexer, ordered by the given closure.
//! * `GHeap::new_by_key()` creates a heap ordered by the key generated by the given closure.
//! * `GHeap::new_by_key_indexer()` creates a heap with the supplied indexer and ordered by the key generated by the given closure.
//! * `GHeap::with_capacity_by_key()` creates a heap with the requested capacity ordered by the key generated by the given closure.
//! * `GHeap::with_capacity_by_key_indexer()` creates a heap with the requested capacity and supplied indexer, ordered by the key generated by the given closure.
//!
//! 
//! # Example
//! 
//! A priority queue implemented with a GHeap.
//!
//! This example implements [Dijkstra's algorithm][dijkstra]
//! to solve the [shortest path problem][sssp] on a [directed graph][dir_graph].
//! It shows how to use [`GHeap`] with custom types.
//!
//! [dijkstra]: http://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
//! [sssp]: http://en.wikipedia.org/wiki/Shortest_path_problem
//! [dir_graph]: http://en.wikipedia.org/wiki/Directed_graph
//! [`GHeap`]: struct.GHeap.html
//!
//! ```
//! use std::cmp::Ordering;
//! use gheap::*;
//! use std::usize;
//!
//! #[derive(Copy, Clone, Eq, PartialEq)]
//! struct State {
//!     cost: usize,
//!     position: usize,
//! }
//!
//! // The priority queue depends on `Ord`.
//! // Explicitly implement the trait so the queue becomes a min-heap
//! // instead of a max-heap.
//! impl Ord for State {
//!     fn cmp(&self, other: &State) -> Ordering {
//!         // Notice that the we flip the ordering on costs.
//!         // In case of a tie we compare positions - this step is necessary
//!         // to make implementations of `PartialEq` and `Ord` consistent.
//!         other.cost.cmp(&self.cost)
//!             .then_with(|| self.position.cmp(&other.position))
//!     }
//! }
//!
//! // `PartialOrd` needs to be implemented as well.
//! impl PartialOrd for State {
//!     fn partial_cmp(&self, other: &State) -> Option<Ordering> {
//!         Some(self.cmp(other))
//!     }
//! }
//!
//! // Each node is represented as an `usize`, for a shorter implementation.
//! struct Edge {
//!     node: usize,
//!     cost: usize,
//! }
//!
//! // Dijkstra's shortest path algorithm.
//!
//! // Start at `start` and use `dist` to track the current shortest distance
//! // to each node. This implementation isn't memory-efficient as it may leave duplicate
//! // nodes in the queue. It also uses `usize::MAX` as a sentinel value,
//! // for a simpler implementation.
//! fn shortest_path(adj_list: &Vec<Vec<Edge>>, start: usize, goal: usize) -> Option<usize> {
//!     // dist[node] = current shortest distance from `start` to `node`
//!     let mut dist: Vec<_> = (0..adj_list.len()).map(|_| usize::MAX).collect();
//!
//!     let mut heap = GHeap::new();
//!
//!     // We're at `start`, with a zero cost
//!     dist[start] = 0;
//!     heap.push(State { cost: 0, position: start });
//!
//!     // Examine the frontier with lower cost nodes first (min-heap)
//!     while let Some(State { cost, position }) = heap.pop() {
//!         // Alternatively we could have continued to find all shortest paths
//!         if position == goal { return Some(cost); }
//!
//!         // Important as we may have already found a better way
//!         if cost > dist[position] { continue; }
//!
//!         // For each node we can reach, see if we can find a way with
//!         // a lower cost going through this node
//!         for edge in &adj_list[position] {
//!             let next = State { cost: cost + edge.cost, position: edge.node };
//!
//!             // If so, add it to the frontier and continue
//!             if next.cost < dist[next.position] {
//!                 heap.push(next);
//!                 // Relaxation, we have now found a better way
//!                 dist[next.position] = next.cost;
//!             }
//!         }
//!     }
//!
//!     // Goal not reachable
//!     None
//! }
//!
//! fn main() {
//!     // This is the directed graph we're going to use.
//!     // The node numbers correspond to the different states,
//!     // and the edge weights symbolize the cost of moving
//!     // from one node to another.
//!     // Note that the edges are one-way.
//!     //
//!     //                  7
//!     //          +-----------------+
//!     //          |                 |
//!     //          v   1        2    |  2
//!     //          0 -----> 1 -----> 3 ---> 4
//!     //          |        ^        ^      ^
//!     //          |        | 1      |      |
//!     //          |        |        | 3    | 1
//!     //          +------> 2 -------+      |
//!     //           10      |               |
//!     //                   +---------------+
//!     //
//!     // The graph is represented as an adjacency list where each index,
//!     // corresponding to a node value, has a list of outgoing edges.
//!     // Chosen for its efficiency.
//!     let graph = vec![
//!         // Node 0
//!         vec![Edge { node: 2, cost: 10 },
//!              Edge { node: 1, cost: 1 }],
//!         // Node 1
//!         vec![Edge { node: 3, cost: 2 }],
//!         // Node 2
//!         vec![Edge { node: 1, cost: 1 },
//!              Edge { node: 3, cost: 3 },
//!              Edge { node: 4, cost: 1 }],
//!         // Node 3
//!         vec![Edge { node: 0, cost: 7 },
//!              Edge { node: 4, cost: 2 }],
//!         // Node 4
//!         vec![]];
//!
//!     assert_eq!(shortest_path(&graph, 0, 1), Some(1));
//!     assert_eq!(shortest_path(&graph, 0, 3), Some(3));
//!     assert_eq!(shortest_path(&graph, 3, 0), Some(7));
//!     assert_eq!(shortest_path(&graph, 0, 4), Some(5));
//!     assert_eq!(shortest_path(&graph, 4, 0), None);
//! }
//! ```


#[macro_use] mod gheap;
pub use crate::gheap::*;

/// An intermediate trait for specialization of `Extend`.
// #[doc(hidden)]
// trait SpecExtend<I: IntoIterator> {
//     /// Extends `self` with the contents of the given iterator.
//     fn spec_extend(&mut self, iter: I);
// }

#[cfg(test)]
mod from_liballoc {
    // The following tests copyed from liballoc/tests/binary_heap.rs

    use super::gheap::*;
    // use std::panic;
    // use std::collections::GHeap;
    // use std::collections::gheap::{Drain, PeekMut};

    #[test]
    fn test_iterator() {
        let data = vec![5, 9, 3];
        let iterout = [9, 5, 3];
        let heap = GHeap::from(data);
        let mut i = 0;
        for el in &heap {
            assert_eq!(*el, iterout[i]);
            i += 1;
        }
    }

    #[test]
    fn test_iterator_reverse() {
        let data = vec![5, 9, 3];
        let iterout = vec![3, 5, 9];
        let pq = GHeap::from(data);

        let v: Vec<_> = pq.iter().rev().cloned().collect();
        assert_eq!(v, iterout);
    }

    #[test]
    fn test_move_iter() {
        let data = vec![5, 9, 3];
        let iterout = vec![9, 5, 3];
        let pq = GHeap::from(data);

        let v: Vec<_> = pq.into_iter().collect();
        assert_eq!(v, iterout);
    }

    #[test]
    fn test_move_iter_size_hint() {
        let data = vec![5, 9];
        let pq = GHeap::from(data);

        let mut it = pq.into_iter();

        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(it.next(), Some(9));

        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next(), Some(5));

        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_move_iter_reverse() {
        let data = vec![5, 9, 3];
        let iterout = vec![3, 5, 9];
        let pq = GHeap::from(data);

        let v: Vec<_> = pq.into_iter().rev().collect();
        assert_eq!(v, iterout);
    }

    #[test]
    fn test_into_iter_sorted_collect() {
        let heap = GHeap::from(vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
        let it = heap.into_iter_sorted();
        let sorted = it.collect::<Vec<_>>();
        assert_eq!(sorted, vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1, 1, 0]);
    }

    #[test]
    fn test_peek_and_pop() {
        let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
        let mut sorted = data.clone();
        sorted.sort();
        let mut heap = GHeap::from(data);
        while !heap.is_empty() {
            assert_eq!(heap.peek().unwrap(), sorted.last().unwrap());
            assert_eq!(heap.pop().unwrap(), sorted.pop().unwrap());
        }
    }

    #[test]
    fn test_peek_mut() {
        let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
        let mut heap = GHeap::from(data);
        assert_eq!(heap.peek(), Some(&10));
        {
            let mut top = heap.peek_mut().unwrap();
            *top -= 2;
        }
        assert_eq!(heap.peek(), Some(&9));
    }

    #[test]
    fn test_peek_mut_pop() {
        let data = vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1];
        let mut heap = GHeap::from(data);
        assert_eq!(heap.peek(), Some(&10));
        {
            let mut top = heap.peek_mut().unwrap();
            *top -= 2;
            assert_eq!(PeekMut::pop(top), 8);
        }
        assert_eq!(heap.peek(), Some(&9));
    }

    #[test]
    fn test_push() {
        let mut heap = GHeap::from(vec![2, 4, 9]);
        assert_eq!(heap.len(), 3);
        assert!(*heap.peek().unwrap() == 9);
        heap.push(11);
        assert_eq!(heap.len(), 4);
        assert!(*heap.peek().unwrap() == 11);
        heap.push(5);
        assert_eq!(heap.len(), 5);
        assert!(*heap.peek().unwrap() == 11);
        heap.push(27);
        assert_eq!(heap.len(), 6);
        assert!(*heap.peek().unwrap() == 27);
        heap.push(3);
        assert_eq!(heap.len(), 7);
        assert!(*heap.peek().unwrap() == 27);
        heap.push(103);
        assert_eq!(heap.len(), 8);
        assert!(*heap.peek().unwrap() == 103);
    }

    // #[test]
    // fn test_push_unique() {
    //     let mut heap = GHeap::<Box<_>>::from(vec![box 2, box 4, box 9]);
    //     assert_eq!(heap.len(), 3);
    //     assert!(**heap.peek().unwrap() == 9);
    //     heap.push(box 11);
    //     assert_eq!(heap.len(), 4);
    //     assert!(**heap.peek().unwrap() == 11);
    //     heap.push(box 5);
    //     assert_eq!(heap.len(), 5);
    //     assert!(**heap.peek().unwrap() == 11);
    //     heap.push(box 27);
    //     assert_eq!(heap.len(), 6);
    //     assert!(**heap.peek().unwrap() == 27);
    //     heap.push(box 3);
    //     assert_eq!(heap.len(), 7);
    //     assert!(**heap.peek().unwrap() == 27);
    //     heap.push(box 103);
    //     assert_eq!(heap.len(), 8);
    //     assert!(**heap.peek().unwrap() == 103);
    // }

    fn check_to_vec(mut data: Vec<i32>) {
        let heap = GHeap::from(data.clone());
        let mut v = heap.clone().into_vec();
        v.sort();
        data.sort();

        assert_eq!(v, data);
        assert_eq!(heap.into_sorted_vec(), data);
    }

    #[test]
    fn test_to_vec() {
        check_to_vec(vec![]);
        check_to_vec(vec![5]);
        check_to_vec(vec![3, 2]);
        check_to_vec(vec![2, 3]);
        check_to_vec(vec![5, 1, 2]);
        check_to_vec(vec![1, 100, 2, 3]);
        check_to_vec(vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 0]);
        check_to_vec(vec![2, 4, 6, 2, 1, 8, 10, 3, 5, 7, 0, 9, 1]);
        check_to_vec(vec![9, 11, 9, 9, 9, 9, 11, 2, 3, 4, 11, 9, 0, 0, 0, 0]);
        check_to_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        check_to_vec(vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
        check_to_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 1, 2]);
        check_to_vec(vec![5, 4, 3, 2, 1, 5, 4, 3, 2, 1, 5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_empty_pop() {
        let mut heap = GHeap::<i32>::new();
        assert!(heap.pop().is_none());
    }

    #[test]
    fn test_empty_peek() {
        let empty = GHeap::<i32>::new();
        assert!(empty.peek().is_none());
    }

    #[test]
    fn test_empty_peek_mut() {
        let mut empty = GHeap::<i32>::new();
        assert!(empty.peek_mut().is_none());
    }

    #[test]
    fn test_from_iter() {
        let xs = vec![9, 8, 7, 6, 5, 4, 3, 2, 1];

        let mut q: GHeap<_> = xs.iter().rev().cloned().collect();

        for &x in &xs {
            assert_eq!(q.pop().unwrap(), x);
        }
    }

    #[test]
    fn test_drain() {
        let mut q: GHeap<_> = [9, 8, 7, 6, 5, 4, 3, 2, 1].iter().cloned().collect();

        assert_eq!(q.drain().take(5).count(), 5);

        assert!(q.is_empty());
    }

    #[test]
    fn test_extend_ref() {
        let mut a = GHeap::new();
        a.push(1);
        a.push(2);

        a.extend(&[3, 4, 5]);

        assert_eq!(a.len(), 5);
        assert_eq!(a.into_sorted_vec(), [1, 2, 3, 4, 5]);

        let mut a = GHeap::new();
        a.push(1);
        a.push(2);
        let mut b = GHeap::new();
        b.push(3);
        b.push(4);
        b.push(5);

        a.extend(&b);

        assert_eq!(a.len(), 5);
        assert_eq!(a.into_sorted_vec(), [1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_append() {
        let mut a = GHeap::from(vec![-10, 1, 2, 3, 3]);
        let mut b = GHeap::from(vec![-20, 5, 43]);

        a.append(&mut b);

        assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
        assert!(b.is_empty());
    }

    #[test]
    fn test_append_to_empty() {
        let mut a = GHeap::new();
        let mut b = GHeap::from(vec![-20, 5, 43]);

        a.append(&mut b);

        assert_eq!(a.into_sorted_vec(), [-20, 5, 43]);
        assert!(b.is_empty());
    }

    #[test]
    fn test_extend_specialization() {
        let mut a = GHeap::from(vec![-10, 1, 2, 3, 3]);
        let b = GHeap::from(vec![-20, 5, 43]);

        a.extend(b);

        assert_eq!(a.into_sorted_vec(), [-20, -10, 1, 2, 3, 3, 5, 43]);
    }

    // #[test]
    // fn test_placement() {
    //     let mut a = GHeap::new();
    //     &mut a <- 2;
    //     &mut a <- 4;
    //     &mut a <- 3;
    //     assert_eq!(a.peek(), Some(&4));
    //     assert_eq!(a.len(), 3);
    //     &mut a <- 1;
    //     assert_eq!(a.into_sorted_vec(), vec![1, 2, 3, 4]);
    // }

    // #[test]
    // fn test_placement_panic() {
    //     let mut heap = GHeap::from(vec![1, 2, 3]);
    //     fn mkpanic() -> usize {
    //         panic!()
    //     }
    //     let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
    //         &mut heap <- mkpanic();
    //     }));
    //     assert_eq!(heap.len(), 3);
    // }

    #[allow(dead_code)]
    fn assert_covariance() {
        fn drain<'new>(d: Drain<'static, &'static str>) -> Drain<'new, &'new str> {
            d
        }
    }
}

#[cfg(feature = "serde")]
#[cfg(test)]
mod tests_serde {
    use super::gheap::*;
    use serde_json;

    #[test]
    fn deserialized_same_small_vec() {
        let heap = GHeap::from(vec![1, 2, 3]);
        let serialized = serde_json::to_string(&heap).unwrap();
        let deserialized: GHeap<i32> = serde_json::from_str(&serialized).unwrap();

        let v0: Vec<_> = heap.into_iter().collect();
        let v1: Vec<_> = deserialized.into_iter().collect();
        assert_eq!(v0, v1);
    }
    #[test]
    fn deserialized_same() {
        let vec: Vec<i32> = (0..1000).collect();
        let heap = GHeap::from(vec);
        let serialized = serde_json::to_string(&heap).unwrap();
        let deserialized: GHeap<i32> = serde_json::from_str(&serialized).unwrap();

        let v0: Vec<_> = heap.into_iter().collect();
        let v1: Vec<_> = deserialized.into_iter().collect();
        assert_eq!(v0, v1);
    }
}
