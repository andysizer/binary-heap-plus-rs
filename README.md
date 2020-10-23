# gheap

`GHeap` implements a d-ary heap in which subtrees containing k items are laid out contiguously 
in a 'pagechunk'. `GHeap` can be used to implement a [Priority Queue](https://www.wikiwand.com/en/Priority_queue#:~:text=In%20computer%20science%2C%20a%20priority,an%20element%20with%20low%20priority.)
in a similar fashion to `std::collections::BinaryHeap`.

This implementation was motivated by the article ['You're Doing It Wrong'](https://queue.acm.org/detail.cfm?id=1814327) 
which introduces the [B-heap](https://www.wikiwand.com/en/B-heap). The main idea is that laying out subtrees
contiguously will tend to increase locality when the heap is traversed using the parent-child relation and that, 
due to the cost of paging, the benefits of increased locality will tend to outweigh the increased cost of computing 
the parent-child relation when memory is under pressure.

Insertion and popping the largest element from a GHeap with a fanout (arity) of 'd' have `O(log n / log d)` 
time complexity. Checking the largest element is `O(1)`. Converting a vector to a GHeap can be done in-place, 
and has `O(n)` complexity. A GHeap can also be converted to a sorted vector in-place, allowing it to be used 
for an `O(nlog n)` in-place heapsort.

The `benches` directory contains benchmarks to test this idea. These appear to show that most of the time any 
improvement shown by `GHeap` over `BinaryHeap` is due to 'fanout' (n-ariness) (i.e. ) rather than chunking. 
However there appear to be 'sweet spots' where a judicious choice of fanout and pagechunk size for items of a 
given size can show significant improvement in performance, implying that for certain use cases GHeap might be a 
reasonable replacement for `std::collections::BinaryHeap` (or a simpler implementation of an d-ary heap).


This implementation is derived from:
1. [binary-heap-plus](https://github.com/sekineh/binary-heap-plus-rs)
   which is itself forked from the version of BinaryHeap in the Rust standard library.
2. The C++ version of [gheap](https://github.com/valyala/gheap.git)


The main difference between a binary heap and a GHeap are the equations used to compute the indices of 
parent and child nodes in the underlying representatoin of the heap. `GHeap` abstracts these computations out into
a trait: `HeapIndexer`. The primary reason for this is Rust's lack of support for `const generics` in stable builds 
(i.e. rustc will reject 'GHeap<i32, 4, 2>').  This may change at some point in the not too distant future 
(see [Shipping Const Generics in 2020](https://without.boats/blog/shipping-const-generics/)) as usable implementation 
of `const generics` is now available in nightly builds.

The additional features that `GHeap` has over and above those of `std::collections::BinaryHeap` are due to 
[binary-heap-plus](https://github.com/sekineh/binary-heap-plus-rs) and include:

* Heaps other than max heap e.g. min heap and heaps constructed using custom comparators.
* Support for serialization and deserialization if `serde` is enabled.

## Benchmarks
TBD

## MSRV (Minimum Supported Rust Version)

This crate requires Rust 1.31.1 or later.

