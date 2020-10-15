use nix::sys::signal::Signal::*;
use nix::sys::wait::WaitStatus::*;
use nix::sys::wait::{waitpid, WaitPidFlag};
use nix::unistd::Pid;
use rand::{distributions::Uniform, prelude::Distribution};
// use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::{sleep, spawn, JoinHandle};
use std::time::Duration;

use nix::sys::signal::*;
use nix::unistd::fork;
use nix::unistd::getpid;
use nix::unistd::ForkResult::*;
use sysinfo::{ProcessExt, System, SystemExt};

use crate::utils::prng::HeapRng;

/// Message used to shut down memory pressure
#[derive(Debug)]
enum ClientMsg {
    Done,
}

/// Message used to signal memory pressure is in place
#[derive(Debug)]
enum ProviderMsg {
    Done,
}

/// The target amount of memory to be consumed by the forked memory hogs.
const TARGET_CONSUMPTION: u64 = 800 * 1024 * 1024;

#[derive(Debug)]
pub struct MemoryPressure {
    client_tx: Sender<ClientMsg>,
    client_rx: Receiver<ProviderMsg>,
    // Use Option to overcome moving provider_thread out from a mutable reference (in drop)
    provider_thread: Option<JoinHandle<()>>,
}

impl MemoryPressure {
    pub fn new(requirement: u64) -> Option<MemoryPressure> {
        let (client_tx, provider_rx) = channel();
        let (provider_tx, client_rx) = channel();

        match MemoryPressure::create_pressure(requirement, provider_tx, provider_rx) {
            Some(provider_thread) => {
                client_rx.recv().unwrap();
                Some(MemoryPressure {
                    client_tx,
                    client_rx,
                    provider_thread: Some(provider_thread),
                })
            }
            None => None,
        }
    }

    fn create_pressure(
        requirement: u64,
        tx: Sender<ProviderMsg>,
        rx: Receiver<ClientMsg>,
    ) -> Option<JoinHandle<()>> {
        let mut sys_info = System::new_all();
        sys_info.refresh_system();

        // If we've got no free swap, there's no point in carrying on. We'll either get
        // horribly wedged and/or unresponsive. Other processes might get nuked as well.
        if sys_info.get_free_swap() == 0 {
            return None;
        } else {
            Some(spawn(move || {
                mem_hog(requirement, tx, rx);
            }))
        }
    }
}

impl Drop for MemoryPressure {
    fn drop(&mut self) {
        // tell the pressure thread to shutdown the 'hog' processes.
        self.client_tx.send(ClientMsg::Done).unwrap();

        // join the provider thread using the Option 'trick'.
        self.provider_thread.take().map(|jh| jh.join().unwrap());

        println!("MemoryPressure shutdown!!!")
    }
}

fn mem_hog(requirement: u64, tx: Sender<ProviderMsg>, rx: Receiver<ClientMsg>) {
    let mut sys_info = System::new_all();
    sys_info.refresh_system();

    // sysinfo memory values are in Kb, so canovert to bytes
    let total_mem = (sys_info.get_total_memory() * 1024) as u64;
    let available_mem = (sys_info.get_available_memory() * 1024) as u64;
    let free_swap = (sys_info.get_free_swap() * 1024) as u64;

    // Find out how much memory we're using. We're going to fork some memory hog processes in a bit,
    // so get a feel for how much memory we consume, so we can subtract it from TARGET_CONSUMPTION
    // to get the amount of memory to allocate (hog_alloc) in/for the hog processes.
    let pid = getpid().into();
    let process = sys_info.get_process(pid).unwrap();
    let proc_mem = (process.memory() * 1024) as u64;
    let mut hog_alloc = TARGET_CONSUMPTION;

    println!(
        "tm = {} am = {} fs = {} pm = {} ha = {}",
        total_mem, available_mem, free_swap, proc_mem, hog_alloc
    );

    // We need this many hogs to consume available memory
    let n_hogs = (available_mem / TARGET_CONSUMPTION) + 1;
    // Approximate the amount of swap space we'll need to accomadate the memory this process will need in
    // addition to the hogs.
    let mut swap_requirement = requirement + (n_hogs * TARGET_CONSUMPTION) - available_mem;

    println!("swap_req = {}", swap_requirement);

    // Adjust our swap requirement so its at least feasible.
    if swap_requirement > free_swap {
        while swap_requirement > free_swap {
            swap_requirement -= free_swap;
            hog_alloc -= free_swap / n_hogs;
        }
    }

    // Now make it less half of what's available
    if swap_requirement > (free_swap / 2) {
        hog_alloc -= free_swap / (2 * n_hogs);
    }

    println!("adjusted hog_alloc = {}", hog_alloc);

    // We're going to manipulate a Vec<usize> in the hogs, so figure out what size
    // it needs to be to consume the required amount of memory.
    let hog_vec_size: u64 = hog_alloc / std::mem::size_of::<u64>() as u64;

    println!(
        "Launching {} hogs with mem size {} (= {})",
        n_hogs,
        hog_alloc,
        n_hogs * hog_alloc
    );

    // Allocate the hog vec. The fork will 'replicate' this in the hogs. N.B. This process
    // has a copy of the vec which we didn't take account of in our memory
    // consumption 'calculations'.
    let mut hog_vec: Vec<u64> = (0..hog_vec_size).into_iter().collect();

    // Fork the hogs
    let hogs = fork_hogs(n_hogs, &mut hog_vec);

    loop {
        sys_info.refresh_system();
        let available_mem = (sys_info.get_available_memory() * 1024) as usize;

        if (available_mem as f64 / total_mem as f64) < 0.05 {
            break;
        }
    }

    // Signal that the hogs are up and running.
    tx.send(ProviderMsg::Done).unwrap();

    // Wait for the client to signal they're done.
    rx.recv().unwrap();

    // We could 'manage' the hogs a bit better i.e. check if their doing their job.
    // In which case we'd want a loop like the
    // loop {
    //     if match rx.recv_timeout(Duration::from_millis(100)) {
    //         Ok(ClientMsg::Done) | Err(RecvTimeoutError::Disconnected) => true,
    //         Err(RecvTimeoutError::Timeout) => false,
    //     } {
    //         break;
    //     } else {
    //         // Do some management here ... i.e. add or delete some hogs
    //     }
    // }

    kill_hogs(&hogs);
}

fn fork_hogs(mut n_hogs: u64, v: &Vec<u64>) -> Vec<Pid> {
    let mut hogs = vec![];

    while n_hogs > 0 {
        if !fork_hog(v.to_vec(), &mut hogs) {
            sleep(Duration::from_millis(50));
            continue;
        }
        sleep(Duration::from_millis(50));
        n_hogs -= 1;
    }
    hogs
}

fn fork_hog(v: Vec<u64>, hogs: &mut Vec<Pid>) -> bool {
    match unsafe { fork() }.expect("Error: Fork Failed") {
        Parent { child } => {
            // Give the hog a chance to warm up.
            sleep(Duration::from_millis(500));
            let r = waitpid(child, Some(WaitPidFlag::WNOHANG));
            match r {
                Ok(StillAlive) => {
                    // println!("hog {} running.", child);
                    hogs.push(child);
                    true
                }
                Ok(Signaled(child, ..)) => {
                    println!("hog {} died.", child);
                    waitpid(child, None).unwrap();
                    false
                }
                Err(_) | Ok(_) => false,
            }
        }
        Child => {
            hog(v);
            false
        }
    }
}

fn kill_hogs(hogs: &Vec<Pid>) {
    for pid in hogs {
        kill(*pid, SIGKILL).unwrap();
    }
    for pid in hogs {
        waitpid(*pid, None).unwrap();
    }
}

fn hog(mut mem: Vec<u64>) {
    let rng: HeapRng = HeapRng::new();
    let index_range = Uniform::new_inclusive(1, mem.len() - 1);
    let mut index_iter = index_range.sample_iter(rng);
    let pgsz = page_size::get();
    let delta = pgsz / std::mem::size_of::<u64>();
    let end = mem.len();
    let mut i = 0;
    let mut acc = 0;

    loop {
        acc += mem[i];
        i += delta;
        if i >= end {
            mem[0] = acc;
            break;
        }
    }

    loop {
        for _i in 0..10000 {
            let trg = index_iter.next().unwrap();
            let src1 = index_iter.next().unwrap();
            let src2 = index_iter.next().unwrap();
            // Should really do this unsafe .. but unchecked_add is 'unstable'
            // unsafe {
            //     mem[trg] = mem[src1].unchecked_add(mem[src2]);
            // }
            mem[trg] = mem[src1] + mem[src2];
            sleep(Duration::from_millis(30));
        }
    }
}
