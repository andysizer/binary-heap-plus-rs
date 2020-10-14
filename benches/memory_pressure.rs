
pub mod mem_pressure {

    use nix::sys::signal::Signal::*;
    use nix::sys::wait::WaitStatus::*;
    use nix::sys::wait::{waitpid, WaitPidFlag};
    use nix::unistd::Pid;
    use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
    use std::thread::{sleep, spawn, JoinHandle};
    use std::time::Duration;
    use std::marker::Sync;

    use nix::sys::signal::*;
    use nix::unistd::fork;
    use nix::unistd::getpid;
    use nix::unistd::ForkResult::*;
    use sysinfo::{System, SystemExt};

    

    /// Message used to shut down memory pressure
    #[derive(Debug)]
    enum ClientMsg {
        Done,
    }

    unsafe impl Sync for ClientMsg {}

    /// Message used to signal memory pressure is in place
    #[derive(Debug)]
    enum ProviderMsg {
        Done,
    }

    unsafe impl Sync for ProviderMsg {}

    #[derive(Debug)]
    pub struct MemoryPressure {
        client_tx: Sender<ClientMsg>,
        client_rx: Receiver<ProviderMsg>,
        // Use Option to overcome moving provider_thread out from a mutable reference (in drop)
        provider_thread: Option<JoinHandle<()>>,
    }

    impl MemoryPressure {
        pub fn new() -> MemoryPressure {
            let (client_tx, provider_rx) = channel();
            let (provider_tx, client_rx) = channel();
            let mem_pressure = MemoryPressure { 
                client_tx, 
                client_rx, 
                provider_thread: Some(spawn(move || {
                    mem_eater(provider_tx, provider_rx);
                })) 
            };
            mem_pressure.client_rx.recv().unwrap();
            mem_pressure              
        }

        fn mem_eater(&self, tx: Sender<ProviderMsg>, rx: Receiver<ClientMsg>) {
            let mut eaters = fork_eaters();
            tx.send(ProviderMsg::Done).unwrap();
    
            loop {
                if match rx.recv_timeout(Duration::from_millis(10)) {
                    Ok(ClientMsg::Done) | Err(RecvTimeoutError::Disconnected) => true,
                    Err(RecvTimeoutError::Timeout) => false,
                } 
                {
                    break;
                } else {
                    prune_eaters(&mut eaters);
                }
            }
            kill_eaters(&eaters);
        }
    }


    impl Drop for MemoryPressure {
        fn drop(&mut self) {
            // tell the pressure thread to shutdown the 'eater' processes.
            self.client_tx.send(ClientMsg::Done).unwrap();

            // join the provider thread using the Option 'trick'.
            self.provider_thread.take().map(|jh| jh.join().unwrap());

            println!("MemoryPressure shutdown!!!")
        }
    }

    fn mem_eater(tx: Sender<ProviderMsg>, rx: Receiver<ClientMsg>) {
        let mut eaters = fork_eaters();
        tx.send(ProviderMsg::Done).unwrap();

        loop {
            if match rx.recv_timeout(Duration::from_millis(10)) {
                Ok(ClientMsg::Done) | Err(RecvTimeoutError::Disconnected) => true,
                Err(RecvTimeoutError::Timeout) => false,
            } 
            {
                break;
            } else {
                prune_eaters(&mut eaters);
            }
        }
        kill_eaters(&eaters);
    }

    fn fork_eaters() -> Vec<Pid> {
        //let (mem_avail, swap_used) = parse_proc_meminfo();
        let mut sys_info = System::new();
        let mut max_alloc: u64 = 1024 * 1024 * 1024;
        let alloc_delta: u64 = max_alloc / 10;
        let swap_target = 1.0 / 10.0;
        let mem_target = 5.0 / 100.0;

        let mut eaters = vec![];

        loop {
            sys_info.refresh_system();

            let total_mem = sys_info.get_total_memory();
            let available_mem = sys_info.get_available_memory();
            let total_swap = sys_info.get_total_swap();
            let free_swap = sys_info.get_free_swap();
            let free_swap_ratio = free_swap as f64 / total_swap as f64;
            let free_mem_ratio = available_mem as f64 / total_mem as f64;

            let alloc_request;

            if free_mem_ratio > mem_target {
                alloc_request = std::cmp::min(max_alloc, available_mem * 1024);
            } else if free_swap_ratio > swap_target {
                alloc_request = std::cmp::min(max_alloc, free_swap * 1024);
            } else {
                break;
            }
            if !fork_eater(alloc_request, &mut eaters) {
                max_alloc = std::cmp::max(alloc_delta, max_alloc - alloc_delta);
            }

            sleep(Duration::from_millis(10));
        }
        eaters
    }

    fn fork_eater(n: u64, eaters: &mut Vec<Pid>) -> bool {
        match unsafe { fork() }.expect("Error: Fork Failed") {
            Parent { child } => {
                sleep(Duration::from_millis(1));
                let r = waitpid(child, Some(WaitPidFlag::WNOHANG));
                match r {
                    Ok(StillAlive) => {
                        // println!("Eater {} running.", child);
                        eaters.push(child);
                        true
                    },
                    Ok(Signaled(child, ..)) => {
                        println!("Eater {} died.", child);
                        waitpid(child, None).unwrap();
                        false
                    },
                    Err(_) | Ok(_) => false,
                }
            }
            Child => {
                eater(n);
                false
            }
        }
    }

    fn prune_eaters(eaters: &mut Vec<Pid>) {
        //let (mem_avail, swap_used) = parse_proc_meminfo();
        let mut sys_info = System::new();
        let swap_target = 1.0 / 3.0;

        loop {
            sys_info.refresh_system();

            let total_swap = sys_info.get_total_swap();
            let free_swap = sys_info.get_free_swap();
            let free_swap_ratio = free_swap as f64 / total_swap as f64;


            if free_swap_ratio < swap_target {
                print!("Pruning: r={} t={} ", free_swap_ratio, swap_target);
                if let Some(pid) = eaters.pop() {
                    println!(" {}", pid);
                    kill(pid, SIGKILL).unwrap();
                    waitpid(pid, None).unwrap();
                }
            } else {
                break;
            }
            sleep(Duration::from_millis(100));
        }
    }

    fn kill_eaters(eaters: &Vec<Pid>) {
        for pid in eaters {
            kill(*pid, SIGKILL).unwrap();
        }
        for pid in eaters {
            waitpid(*pid, None).unwrap();
        }
    }

    fn eater(n: u64) {
        let _pid = getpid();
        let page_sz = page_size::get();
        let rl = page_sz / 4;

        // println!("eater {} eating {}.", pid, n);
        let size = n as usize / 8;
        let mut mem: Vec<u64> = Vec::with_capacity(size);

        // println!("eater {} pushing.", pid);
        for i in 0..size {
            mem.push(i as u64);
            if i > 0 && i % rl == 0 {
                // println!("eater {} pushing.", pid);
                sleep(Duration::from_millis(500));
            }
        }
        // println!("eater {} pushed {}.", pid, n);
        sleep(Duration::from_millis(500));

        // println!("eater {} moving {}.", pid, n);
        loop {
            let first = mem[0];
            for i in 0..size - 2 {
                mem[i] = mem[i + 1];
                if i > 0 && i % rl == 0 {
                    // println!("eater {} moving.", pid);
                    sleep(Duration::from_millis(500));
                }
            }
            mem[size - 1] = first;
            // println!("eater {} sleeping.", pid);
            sleep(Duration::from_secs(1));
            // println!("eater {} waking.", pid);
        }
    }
}
