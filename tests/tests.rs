


#[macro_use] use gheap::*;

fn passed() {
    println!(" OK");
}

fn test_parent_child<I: Indexer>(idxer: &I, start_index: usize, n: usize) {

    assert!(start_index > 0);
    assert!(start_index <= std::usize::MAX - n);

    print!("    test_parent_child(start_index={}, n={})", start_index, n);

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


fn test_is_heap<I: Indexer+Default>(n: usize) {
    assert!(n > 0);

    print!("    test_is_heap(n={})", n);

    let mut heap: GHeap<usize, MaxComparator, I> = GHeap::new_indexer();

    heap.clear();
    for i in 0 .. n {
        heap.push(i);
    }
    assert!(heap.is_heap());

    heap.clear();
    for i in 0 .. n {
        heap.push(n - i);
    }
    assert!(heap.is_heap());

    heap.clear();
    for _i in 0 .. n {
        heap.push(n);
    }
    assert!(heap.is_heap());

    passed();
}

macro_rules! test_all {
    ($( $s:ident, $f:literal, $p:literal)+) => {
        $(
            def_indexer!($s, $f, $p);
            let idx = $s {};
            test_all(&idx);
        
        )+
        
    };
}

fn test_all<I: Indexer+Default>(idx: &I) {
    let n = 1000000;
    test_parent_child(idx, 1, n);
    test_parent_child(idx, std::usize::MAX - n, n);
}

#[test]
fn main_test() {
    println!("    main_test() start");
    
    test_all!(TI1_1, 1,1);
    test_all!(TI2_1, 2,1);
    test_all!(TI3_1, 3,1);
    test_all!(TI4_1, 4,1);
    test_all!(TI101_1, 101,1);

    test_all!(TI1_2, 1,2);
    test_all!(TI2_2, 2,2);
    test_all!(TI3_2, 3,2);
    test_all!(TI4_2, 4,2);
    test_all!(TI101_2, 101,2);

    test_all!(TI1_3, 1,3);
    test_all!(TI2_3, 2,3);
    test_all!(TI3_3, 3,3);
    test_all!(TI4_3, 4,3);
    test_all!(TI101_3, 101,3);

    test_all!(TI1_4, 1,4);
    test_all!(TI2_4, 2,4);
    test_all!(TI3_4, 3,4);
    test_all!(TI4_4, 4,4);
    test_all!(TI101_4, 101,4);

    test_all!(TI1_101, 1,101);
    test_all!(TI2_101, 2,101);
    test_all!(TI3_101, 3,101);
    test_all!(TI4_101, 4,101);
    test_all!(TI101_101, 101,101);

    println!("    main_test() OK");

}