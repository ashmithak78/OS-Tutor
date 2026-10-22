"""
build_full_dataset.py
=====================
Comprehensive Dataset & Knowledge-Base Builder for OS-Tutor.
1. Compiles the curated Alpaca-style fine-tuning dataset (dataset.json) and splits
   it reproducibly into train.json, val.json, and test.json.
2. Extracts, cleans, and chunks course material from 'os.pdf'.
3. Embeds chunks with all-MiniLM-L6-v2 and stores them into persistent ChromaDB ('os_notes'),
   ensuring all stale data from previous PDFs is completely removed.
"""

import os
import sys
import re
import json
import random
import warnings
warnings.filterwarnings("ignore")

# Force offline mode for transformers/huggingface to avoid sandbox DNS retries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
import torch


data = [
    # --- 1. CORE PROCESS ABSTRACTION & VIRTUALIZATION ---
    {
        "instruction": "What is a process?",
        "input": "",
        "output": "A process is simply a running program. A program by itself is just inert instructions sitting on disk; the OS takes those bytes, loads them into memory, and gets them running, which is what turns a program into a process. Example: when you double-click a music player icon, the OS loads it from disk and makes it a running process. Exam tip: don't confuse 'program' (static, on disk) with 'process' (dynamic, running in memory)."
    },
    {
        "instruction": "What is the difference between a program and a process?",
        "input": "",
        "output": "A program is a static collection of instructions and passive data residing in an executable file on secondary storage (disk). A process is an active, dynamic instance of a program in execution, with allocated CPU registers, program counter, and memory address space. Example: 'gcc' sitting in /usr/bin is a program; running 'gcc main.c' creates an active process. Exam tip: on exams, state clearly that a program is passive while a process is active."
    },
    {
        "instruction": "List the three basic process states and describe each.",
        "input": "",
        "output": "Running: the process is currently executing instructions on the CPU. Ready: the process is prepared to run, but the OS scheduler has chosen not to allocate CPU time to it right now. Blocked: the process cannot run until an external event occurs, such as completion of an I/O request. Exam tip: moving ready to running is called 'scheduled'; moving running to ready is called 'descheduled'."
    },
    {
        "instruction": "What is CPU virtualization and how does the operating system achieve it?",
        "input": "",
        "output": "CPU virtualization is the illusion created by the OS that each process has its own private, dedicated processor, even when the system has only one physical CPU core. The OS achieves this primarily through time-sharing: running one process for a time slice, pausing it, saving its state, and running another. Example: listening to music while compiling code simultaneously on a single-core machine. Exam tip: distinguish time-sharing (sharing one resource across time) from space-sharing (dividing a resource like RAM across space)."
    },
    {
        "instruction": "Explain time-sharing versus space-sharing in operating systems.",
        "input": "",
        "output": "Time-sharing is a resource multiplexing technique where different entities use the entire resource sequentially for short intervals of time. Space-sharing is where a resource is divided into distinct portions and allocated simultaneously to multiple entities. Example: CPU cores are time-shared among processes, whereas physical RAM and disk blocks are space-shared. Exam tip: remember that CPU scheduling is primarily time-sharing, while memory paging is space-sharing."
    },
    {
        "instruction": "What are the components that comprise the machine state of a process?",
        "input": "",
        "output": "A process's machine state consists of everything in the hardware that defines its execution: its memory address space (instructions, stack, heap, static data), CPU registers (including the Program Counter, Stack Pointer, and general-purpose registers), and persistent I/O state such as a table of open file descriptors. Example: when paused, the OS saves registers like EIP/RIP to remember the exact instruction to resume next. Exam tip: always mention the Program Counter (PC) and Stack Pointer (SP) as critical registers."
    },
    {
        "instruction": "What is a Process Control Block (PCB) and what information does it store?",
        "input": "",
        "output": "A Process Control Block (PCB), known in Linux as 'struct task_struct', is the kernel data structure used by the OS to track all information about an individual process. It stores the process ID (PID), process state (ready, running, blocked), CPU register contents for context switches, memory management metadata (page table pointers), scheduling priority, and open file tables. Example: when a process is descheduled, the OS dumps its register values directly into its PCB. Exam tip: memorize at least four PCB fields for written exams: PID, registers, state, and memory pointers."
    },
    {
        "instruction": "Describe the state transitions a process undergoes during its lifecycle.",
        "input": "",
        "output": "A process moves: 1) Initial/New to Ready upon creation and loading; 2) Ready to Running when selected by the scheduler; 3) Running to Ready when preempted or descheduled; 4) Running to Blocked when initiating an I/O operation or waiting on an event; 5) Blocked to Ready when the requested I/O completes; 6) Running to Terminated upon exit. Exam tip: note that a blocked process NEVER transitions directly back to Running; it must first re-enter the Ready queue."
    },
    {
        "instruction": "Why does a blocked process transition to Ready instead of directly to Running?",
        "input": "",
        "output": "When an I/O event completes, the CPU is typically already busy executing another process that is currently in the Running state. The unblocked process cannot simply seize the CPU immediately without violating the scheduler's policy, so the OS moves it to the Ready queue to await its next scheduled turn. Example: downloading a file finishes while you are typing in a text editor; the downloader becomes Ready, waiting for the scheduler. Exam tip: a direct transition from Blocked to Running does not exist in standard OS state diagrams."
    },
    {
        "instruction": "What is a context switch and what overhead does it introduce?",
        "input": "",
        "output": "A context switch is the low-level mechanism by which the OS saves the execution context (registers, program counter, kernel stack pointer) of the currently running process and restores the execution context of the next process to be run. Its overhead includes direct costs (executing assembly save/restore instructions and flushing or updating MMU page tables) and indirect costs (CPU cache and TLB misses as the new process warms up the cache). Example: switching between two CPU-intensive tasks every 10ms. Exam tip: direct overhead is in microseconds, but cache-pollution indirect overhead often dominates real-world performance."
    },

    # --- 2. LIMITED DIRECT EXECUTION & PRIVILEGE LEVELS ---
    {
        "instruction": "What is Limited Direct Execution (LDE) and why is it used?",
        "input": "",
        "output": "Limited Direct Execution (LDE) is the foundational design pattern where the OS lets user programs run directly on the physical CPU hardware for maximum speed, but limits their authority through hardware-enforced privilege rings (user vs kernel mode) and system calls. Without direct execution, programs would be sluggishly emulated; without limits, a program could commandeer the entire machine. Example: matrix multiplication runs native assembly instructions, but opening a disk file requires a trap into kernel mode. Exam tip: LDE achieves both high performance and operating system control."
    },
    {
        "instruction": "What is the difference between user mode and kernel mode?",
        "input": "",
        "output": "User mode is a restricted processor privilege level where user applications run; direct access to hardware, I/O devices, and arbitrary physical memory addresses is prohibited. Kernel mode (or supervisor mode) is an unrestricted privilege level where the OS runs with complete access to all physical instructions, memory, and devices. Example: trying to execute an 'in/out' assembly instruction in user mode triggers a general protection fault. Exam tip: switching between user and kernel mode requires hardware trap/interrupt support."
    },
    {
        "instruction": "Explain the role of the trap instruction and the trap table.",
        "input": "",
        "output": "A trap instruction is a special machine instruction executed by user software that simultaneously elevates CPU privilege from user mode to kernel mode and jumps to a pre-designated address in the OS. The trap table (or interrupt vector table) is initialized by the OS at boot time while in kernel mode, telling the hardware exactly which kernel handler routines correspond to each system call or interrupt number. Example: calling 'int 0x80' or 'syscall' on x86 triggers the trap mechanism. Exam tip: user code can never choose the target address of a trap; it only specifies the system call number, preventing malicious jumps."
    },
    {
        "instruction": "Why must the trap table be configured at boot time in kernel mode?",
        "input": "",
        "output": "If user-level programs were allowed to alter the trap table, an unprivileged program could rewrite the entry point of the trap handler to point to its own arbitrary code, effectively granting itself root/kernel privileges and disabling all security checks. Boot-time initialization ensures only the trusted kernel sets these jump targets. Example: a security exploit where an attacker redirects vector 128 to unmapped user memory. Exam tip: this is a classic exam question on OS security guarantees."
    },
    {
        "instruction": "What happens during a return-from-trap instruction?",
        "input": "",
        "output": "The return-from-trap instruction (such as 'iret' or 'sysret' on x86) de-elevates CPU privilege from kernel mode back to user mode, pops the saved user-space registers and program counter off the kernel stack, and resumes user-space execution right after the original system call site. Example: when read() finishes fetching disk bytes, return-from-trap returns control to your C code. Exam tip: remember that trap pushes registers onto the kernel stack, and return-from-trap pops them back off."
    },
    {
        "instruction": "How does the operating system regain control of the CPU if a user program enters an infinite loop?",
        "input": "",
        "output": "In non-cooperative systems, the OS uses a hardware timer interrupt configured at boot. Every few milliseconds, the hardware timer chip fires an electronic interrupt, causing the CPU to automatically pause the running program, elevate to kernel mode, and jump to the OS timer interrupt handler, allowing the scheduler to decide whether to preempt the looping program. Example: a program with 'while(1);' will not freeze your computer because the timer interrupt repeatedly kicks in. Exam tip: cooperative multitasking relies on the program yielding voluntarily (via sys_yield); preemptive multitasking relies on timer interrupts."
    },
    {
        "instruction": "What is the difference between cooperative multitasking and non-cooperative (preemptive) multitasking?",
        "input": "",
        "output": "In cooperative multitasking, the OS trusts user applications to periodically yield control of the CPU by issuing system calls or explicit yield operations. In non-cooperative (preemptive) multitasking, the OS forcefully regains control at periodic intervals via hardware timer interrupts regardless of what the user program does. Example: early Mac OS was cooperative and could freeze if an app crashed; modern Linux and Windows are strictly preemptive. Exam tip: cooperative multitasking fails completely if an application hangs in an infinite loop without issuing system calls."
    },

    # --- 3. PROCESS API (FORK, EXEC, WAIT) ---
    {
        "instruction": "How does the fork() system call work in UNIX-like operating systems?",
        "input": "",
        "output": "The fork() system call creates an almost exact duplicate copy of the calling process, known as the child. The child receives its own unique PID, its own copy of the parent's address space (code, data, stack, heap), and copies of open file descriptors. It returns twice: returning 0 to the newly created child process, and returning the child's PID to the parent process. Example: 'pid_t p = fork(); if (p == 0) { /* child */ } else { /* parent */ }'. Exam tip: students frequently forget that fork returns TWICE from a single invocation."
    },
    {
        "instruction": "Why does fork() return different values to the parent and child processes?",
        "input": "",
        "output": "Returning different values is how identical copies of code determine whether they are running as the parent or the child. The child gets 0 so it knows it is the child (it can query its own PID with getpid()), while the parent gets the child's positive PID so it can track, wait for, or manage that specific child. If fork fails, it returns -1. Example: checking 'if (rc < 0) exit(1); else if (rc == 0) ...'. Exam tip: the return value of fork() is 0 for child, PID > 0 for parent, -1 for error."
    },
    {
        "instruction": "What does the exec() system call family do, and how does it differ from fork()?",
        "input": "",
        "output": "While fork() creates a duplicate copy of the current process, exec() overwrites the existing process's memory space with a completely new program loaded from disk (re-initializing code, data, heap, and stack). The PID of the process does not change, and if exec succeeds, it never returns to the calling code because that code has been overwritten. Example: calling execvp('ls', args) replaces the current process with the 'ls' binary. Exam tip: fork creates a new process; exec transforms the current process into a new program."
    },
    {
        "instruction": "Why did UNIX designers separate process creation into fork() followed by exec() instead of a single call?",
        "input": "",
        "output": "Separating fork() and exec() gives the shell the opportunity to execute code between the fork and the exec. During this intermediate window, the shell can manipulate file descriptors, set up environment variables, change working directories, and configure UNIX pipelines or I/O redirection before the target program begins running. Example: redirecting standard output to a file using 'close(STDOUT_FILENO); open(\"out.txt\", ...)' right before calling exec(). Exam tip: always mention shell I/O redirection as the primary motivation for decoupling fork and exec."
    },
    {
        "instruction": "What is the purpose of the wait() system call?",
        "input": "",
        "output": "The wait() system call suspends the calling parent process until one of its child processes terminates. It allows the parent to collect the child's exit status code and enables the OS kernel to clean up the terminated child's remaining resources and PCB. Example: a shell uses wait() to pause until a command finishes executing before printing the next prompt. Exam tip: if a parent does not call wait(), a terminated child becomes a 'zombie'."
    },
    {
        "instruction": "What is a zombie process and how is it resolved?",
        "input": "",
        "output": "A zombie process is a process that has completed execution (via exit()) but still retains an entry in the OS process table because its parent has not yet read its exit status code using wait(). The OS keeps the PCB and exit code so the parent can inspect it. It is resolved when the parent finally calls wait() or waitpid(), or if the parent dies, in which case init/systemd (PID 1) adopts the zombie and reaps it. Example: a process with state 'Z' in 'ps aux'. Exam tip: zombies do not consume CPU or RAM, only a slot in the kernel PID/process table."
    },
    {
        "instruction": "What is an orphan process and what happens to it in UNIX?",
        "input": "",
        "output": "An orphan process is a process that continues executing after its parent process has terminated without waiting for it. The operating system handles orphan processes by having the root ancestor process—traditionally 'init' (PID 1) or 'systemd' in modern Linux—adopt them. PID 1 periodically reaps terminated adopted children to prevent permanent zombies. Example: launching a background daemon from a terminal and closing the terminal window. Exam tip: contrast orphans (still actively executing without a parent) with zombies (dead, waiting to be reaped)."
    },

    # --- 4. CPU SCHEDULING ALGORITHMS & METRICS ---
    {
        "instruction": "Define turnaround time and response time in CPU scheduling.",
        "input": "",
        "output": "Turnaround time is the total time from a job's arrival in the system to its ultimate completion: T_turnaround = T_completion - T_arrival. Response time is the time from a job's arrival until it is scheduled on the CPU for the very first time: T_response = T_firstrun - T_arrival. Example: a batch job wants minimal turnaround time; an interactive text editor wants minimal response time. Exam tip: turnaround time measures batch performance, while response time measures interactive responsiveness."
    },
    {
        "instruction": "What is First-In, First-Out (FIFO) scheduling and what is the convoy effect?",
        "input": "",
        "output": "FIFO (First-In, First-Out) or FCFS schedules jobs strictly in the order they arrive in the ready queue. The convoy effect occurs when a long-running, CPU-bound process arrives just before several short, I/O-bound processes, forcing all short processes to wait behind the heavyweight job and drastically degrading average turnaround time. Example: one customer at a grocery store checkout with 100 items while three customers behind them each have a single apple. Exam tip: the convoy effect is the primary drawback of FIFO scheduling."
    },
    {
        "instruction": "Explain Shortest Job First (SJF) and why non-preemption limits its effectiveness.",
        "input": "",
        "output": "Shortest Job First (SJF) schedules jobs in order of their execution duration, prioritizing the shortest job to minimize average turnaround time. However, if SJF is non-preemptive and a long job arrives first, subsequent short jobs arriving just after execution starts must wait until the long job finishes entirely, recreating the convoy effect. Example: Job A (100s) arrives at t=0, Jobs B and C (10s each) arrive at t=10; non-preemptive SJF runs A to completion first. Exam tip: SJF is only optimal if all jobs arrive at the exact same moment."
    },
    {
        "instruction": "What is Shortest Time-to-Completion First (STCF) scheduling?",
        "input": "",
        "output": "STCF (also known as Preemptive Shortest Job First or PSJF) adds preemption to SJF. Whenever a new job arrives in the ready queue, the scheduler compares the remaining execution time of the currently running job against the new job's total run time; if the new job requires less time to complete, the current job is preempted and the new job is scheduled immediately. Example: Job A has 90s left when Job B arrives needing only 10s; STCF preempts A and runs B. Exam tip: STCF achieves provably optimal average turnaround time for independent jobs."
    },
    {
        "instruction": "How does Round Robin (RR) scheduling work and what is the trade-off in choosing the time slice length?",
        "input": "",
        "output": "Round Robin (RR) runs each ready job for a fixed duration called a time slice (or scheduling quantum), moving the job to the back of the ready queue if it has not finished by the end of its slice. The trade-off: a short time slice provides excellent response time for interactive tasks but incurs heavy context switch overhead; a long time slice amortizes context switch costs but makes response time sluggish, approaching FIFO. Example: time slices are typically chosen between 10ms and 100ms. Exam tip: RR optimizes response time at the direct expense of turnaround time."
    },
    {
        "instruction": "How does the scheduler incorporate I/O operations into CPU scheduling?",
        "input": "",
        "output": "When a process initiates an I/O request (such as reading from disk or network), it does not use the CPU during the I/O wait. The scheduler moves it from the Running state to the Blocked state and schedules another ready process on the CPU. When the I/O finishes, an interrupt moves the I/O process back to Ready, effectively breaking long tasks into smaller sub-bursts of CPU computation interleaved with I/O. Example: treating a 50ms process with 10ms I/O intervals as five 10ms CPU bursts. Exam tip: overlapping CPU execution and I/O wait is essential for maximum hardware utilization."
    },
    {
        "instruction": "What is a Multi-Level Feedback Queue (MLFQ) and what problem does it solve?",
        "input": "",
        "output": "MLFQ is an adaptive priority-based scheduling algorithm designed to optimize both turnaround time for long batch jobs and response time for interactive jobs without requiring advance knowledge of job runtimes. It maintains multiple priority queues with different time slices: jobs that repeatedly use their full time slice are deemed CPU-bound and demoted to lower queues, while jobs that frequently yield for I/O remain at high priority. Example: your terminal keystrokes respond instantly at top priority, while background video rendering sinks to low priority. Exam tip: MLFQ learns job behavior dynamically from past CPU usage."
    },
    {
        "instruction": "What is priority starvation in MLFQ and how is it prevented using priority boost?",
        "input": "",
        "output": "Priority starvation occurs when an endless stream of short interactive jobs monopolizes high-priority queues, preventing CPU-bound jobs in lower queues from ever receiving CPU time. It is prevented by applying a periodic 'Priority Boost': after a fixed time period S (e.g., every few seconds), the OS moves all processes in the system to the highest-priority queue. Example: resetting all processes to priority 0 every 1 second ensures batch jobs make steady progress. Exam tip: priority boost guarantees that no process starves and allows CPU-bound jobs that become interactive to regain high priority."
    },

    # --- 5. MULTIPLE CHOICE QUESTIONS (MCQ) ---
    {
        "instruction": "Which of the following process state transitions is INVALID in standard operating systems?\nA) Running to Ready\nB) Blocked to Running\nC) Running to Blocked\nD) Blocked to Ready",
        "input": "",
        "output": "Correct Answer: B) Blocked to Running. Explanation: In standard operating system scheduling, when an event or I/O operation completes, the unblocked process is moved from the Blocked state to the Ready state. It cannot jump directly onto the CPU because another process may already be running. Exam tip: watch out for this trick on exams; Blocked always leads to Ready, never directly to Running."
    },
    {
        "instruction": "What value does fork() return to the newly created child process upon success?\nA) The child's PID\nB) The parent's PID\nC) 0\nD) 1",
        "input": "",
        "output": "Correct Answer: C) 0. Explanation: fork() returns 0 to the child process so that the child can easily identify itself as the child. The parent process receives the actual process identifier (PID > 0) of the new child. Exam tip: memorize this pairing: Parent receives Child PID; Child receives 0."
    },
    {
        "instruction": "What happens to the Process ID (PID) when a process calls execvp()?\nA) The PID is incremented by 1\nB) A new PID is assigned by the kernel\nC) The PID remains unchanged\nD) The process becomes PID 1",
        "input": "",
        "output": "Correct Answer: C) The PID remains unchanged. Explanation: The exec() family replaces the address space and instructions of the calling process with a new binary program, but it does not create a new process entity. Hence, the PID remains identical. Exam tip: fork creates a new PID; exec keeps the same PID."
    },
    {
        "instruction": "Which scheduling algorithm provably minimizes average turnaround time when all jobs arrive simultaneously?\nA) First-In, First-Out (FIFO)\nB) Round Robin (RR)\nC) Shortest Job First (SJF)\nD) Priority Scheduling",
        "input": "",
        "output": "Correct Answer: C) Shortest Job First (SJF). Explanation: By executing jobs with the shortest execution time first, early completions dramatically reduce the cumulative waiting time of the batch, achieving minimal average turnaround time. Exam tip: SJF minimizes average turnaround time assuming simultaneous arrivals; STCF minimizes it when jobs arrive dynamically."
    },
    {
        "instruction": "What is the primary benefit of a hardware timer interrupt in operating systems?\nA) It keeps track of the system clock for file timestamps\nB) It prevents user processes from monopolizing the CPU indefinitely\nC) It speeds up context switching between threads\nD) It eliminates the need for system calls",
        "input": "",
        "output": "Correct Answer: B) It prevents user processes from monopolizing the CPU indefinitely. Explanation: The periodic timer interrupt forces the CPU to transfer control back to the operating system kernel, enabling preemptive scheduling even if a user program enters an infinite loop. Exam tip: timer interrupts are the cornerstone of preemptive multitasking."
    },
    {
        "instruction": "A terminated process whose parent has not yet called wait() is known as a(n):\nA) Orphan process\nB) Zombie process\nC) Daemon process\nD) Blocked process",
        "input": "",
        "output": "Correct Answer: B) Zombie process. Explanation: A zombie process has finished execution and released its memory, but retains its exit status and entry in the process table until its parent reads that status via wait(). Exam tip: remember: Orphan = parent died first; Zombie = child died first, unharvested."
    },
    {
        "instruction": "Which component of process context switch overhead is considered an 'indirect' cost?\nA) Saving hardware registers to the kernel stack\nB) Restoring hardware registers from the kernel stack\nC) CPU cache and TLB warm-up misses\nD) Updating the process state variable in the PCB",
        "input": "",
        "output": "Correct Answer: C) CPU cache and TLB warm-up misses. Explanation: Saving and restoring registers are direct assembly instructions executed during the switch. Cache pollution (having to reload data and instructions into CPU caches for the newly scheduled process) represents indirect overhead. Exam tip: indirect costs often significantly exceed direct switch costs in modern processors."
    },
    {
        "instruction": "In Round Robin scheduling, what is the consequence of choosing an excessively small time slice?\nA) Average response time will increase dramatically\nB) CPU time will be wasted on frequent context switch overhead\nC) The algorithm will behave identically to FIFO\nD) High-priority processes will starve low-priority processes",
        "input": "",
        "output": "Correct Answer: B) CPU time will be wasted on frequent context switch overhead. Explanation: If the time slice is too small (e.g., 10 microseconds), the time spent saving and restoring registers and cache lines becomes a substantial fraction of total CPU time, reducing overall throughput. Exam tip: time slice must be significantly larger than context switch overhead."
    },
    {
        "instruction": "Which system call allows a parent process to collect the exit status of a terminated child?\nA) exit()\nB) fork()\nC) wait()\nD) kill()",
        "input": "",
        "output": "Correct Answer: C) wait(). Explanation: The wait() and waitpid() system calls block the parent until a child terminates, returning the child's termination status and allowing the OS to deallocate the child's PCB. Exam tip: exit() produces the status code; wait() consumes and reaps it."
    },
    {
        "instruction": "Which of the following operations CANNOT be executed directly in user mode?\nA) Adding two numbers in CPU registers\nB) Writing data directly to a physical hard drive sector\nC) Calling a user-defined C function\nD) Branching to a loop label",
        "input": "",
        "output": "Correct Answer: B) Writing data directly to a physical hard drive sector. Explanation: Direct hardware I/O instructions are privileged operations restricted to kernel mode to prevent malicious or accidental disruption of system storage. User mode must request this via system calls. Exam tip: all direct hardware I/O is strictly privileged."
    },

    # --- 6. VIVA & SHORT TECHNICAL QUESTIONS ---
    {
        "instruction": "Why does a system call require a mode switch from user mode to kernel mode?",
        "input": "",
        "output": "Direct Answer: A system call requests services involving protected resources (like disk, network, or memory mapping) that user programs cannot access directly due to hardware privilege restrictions. Technical Detail: The CPU switches privilege levels via a software interrupt/trap, enabling the kernel to perform rigorous safety checks before executing the requested hardware operation. Exam tip: mode switch changes the hardware execution level; context switch changes the active running process."
    },
    {
        "instruction": "What is the key difference between a mode switch and a context switch?",
        "input": "",
        "output": "Direct Answer: A mode switch changes the CPU privilege level between user mode and kernel mode while remaining within the context of the same process. Technical Detail: A context switch transfers execution from one process to an entirely different process, requiring saving and loading full register state and memory mappings. Exam tip: every context switch involves mode switches, but a mode switch does not necessarily cause a context switch."
    },
    {
        "instruction": "What is the Program Counter (PC) and why is it critical during a context switch?",
        "input": "",
        "output": "Direct Answer: The Program Counter (PC, or instruction pointer) contains the memory address of the next instruction to be fetched and executed by the CPU. Technical Detail: During a context switch, saving the PC in the PCB ensures that when the process is later rescheduled, the CPU resumes execution at the exact instruction where it was interrupted. Exam tip: if the PC were not saved, the process would have no way of knowing where to resume execution."
    },
    {
        "instruction": "What is the difference between CPU-bound and I/O-bound processes?",
        "input": "",
        "output": "Direct Answer: A CPU-bound process spends the vast majority of its time performing computations on the processor, whereas an I/O-bound process spends most of its time waiting for I/O operations (disk, network, keyboard) to complete. Technical Detail: CPU-bound processes use their full time slices, while I/O-bound processes quickly yield the CPU after short bursts. Exam tip: schedulers like MLFQ prioritize I/O-bound jobs to maximize device concurrency and UI responsiveness."
    },
    {
        "instruction": "What is Copy-on-Write (COW) and how does it optimize fork()?",
        "input": "",
        "output": "Direct Answer: Copy-on-Write is an optimization where the OS marks the parent's memory pages as read-only and shares them with the child upon fork(), copying a page only when either process attempts to write to it. Technical Detail: Because fork() is often immediately followed by exec(), duplicating all physical RAM pages is wasteful; COW avoids unnecessary memory copying. Exam tip: COW transforms fork() from an expensive O(memory size) operation into a rapid O(page table copy) operation."
    },
    {
        "instruction": "What is starvation in operating system scheduling?",
        "input": "",
        "output": "Direct Answer: Starvation occurs when a runnable process is perpetually denied CPU time because other processes consistently receive higher scheduling priority. Technical Detail: Non-preemptive priority scheduling and unmanaged multi-level queues suffer from starvation unless aging or periodic priority boosting mechanisms are implemented. Exam tip: distinguish starvation (process can theoretically run if others stop) from deadlock (processes are circularly blocked waiting for each other)."
    },
    {
        "instruction": "What is an interrupt vector table and where does it reside?",
        "input": "",
        "output": "Direct Answer: An interrupt vector table (or IDT) is a dispatch table in kernel memory containing memory addresses of interrupt service routines (ISRs) for each hardware and software interrupt. Technical Detail: It is set up by the OS during boot in privileged mode and indexed directly by interrupt numbers fired by hardware or trap instructions. Exam tip: examiners often ask whether this table can be modified by user programs—the answer is strictly NO."
    },
    {
        "instruction": "What does a process address space consist of?",
        "input": "",
        "output": "Direct Answer: A process address space is the virtual memory view of a process, containing code (text), static/global data, a dynamic heap growing upward, and a call stack growing downward. Technical Detail: The text segment holds machine instructions; heap handles dynamically allocated memory (malloc); stack handles local variables, parameters, and function call frames. Exam tip: remember that the stack and heap grow toward each other in conventional memory layouts."
    },
    {
        "instruction": "What happens when a process attempts to execute a privileged instruction in user mode?",
        "input": "",
        "output": "Direct Answer: The processor hardware refuses to execute the instruction and immediately generates an internal hardware exception (such as a General Protection Fault on x86). Technical Detail: The CPU elevates to kernel mode and vectors to the OS exception handler, which terminates the offending process with a signal like SIGSEGV or SIGILL. Exam tip: hardware enforcement guarantees that unprivileged programs cannot bypass OS controls."
    },
    {
        "instruction": "Why does a shell need to call fork() before calling execvp()?",
        "input": "",
        "output": "Direct Answer: If the shell called execvp() directly without forking, the shell program itself would be permanently overwritten by the commanded program, terminating the shell session as soon as the command finished. Technical Detail: Forking creates a child process where execvp() can safely replace memory while the parent shell waits and remains alive for future commands. Exam tip: this is one of the most frequently asked viva questions in operating systems."
    }
]

# Add more expanded entries to reach 105+ high quality questions
expanded_entries = [
    {
        "instruction": "Explain the difference between a trap, an interrupt, and an exception.",
        "input": "",
        "output": "A trap is an intentional, synchronous software event generated by a program to request an OS service via a system call. An interrupt is an asynchronous hardware event generated by external devices (e.g., keyboard, network card, timer). An exception is a synchronous fault condition caused by invalid program behavior (e.g., division by zero, page fault). Example: calling read() is a trap; a timer tick is an interrupt; dividing by zero is an exception. Exam tip: remember that interrupts are asynchronous (external to the CPU), while traps and exceptions are synchronous (caused by executing an instruction)."
    },
    {
        "instruction": "What is the role of the Stack Pointer (SP) in a process?",
        "input": "",
        "output": "The Stack Pointer is a CPU register that tracks the top address of the current process's call stack. It adjusts automatically as function parameters, return addresses, and local variables are pushed and popped during execution. Example: calling a recursive function decrements the SP on x86 as stack frames are pushed. Exam tip: stack overflow occurs when the SP collides with or exceeds the boundary of allocated stack memory."
    },
    {
        "instruction": "What is the difference between a monolithic kernel and a microkernel?",
        "input": "",
        "output": "A monolithic kernel runs all OS services (file systems, device drivers, networking, scheduler) in supervisor/kernel mode inside a single address space. A microkernel keeps only minimal core mechanisms (IPC, basic scheduling, virtual memory) in kernel mode, running file systems and device drivers as user-space servers. Example: Linux is monolithic; seL4 and QNX are microkernels. Exam tip: monolithic kernels offer higher performance due to direct function calls; microkernels offer better isolation and crash resilience."
    },
    {
        "instruction": "What is the purpose of the dup2() system call in UNIX shells?",
        "input": "",
        "output": "The dup2(oldfd, newfd) system call duplicates an open file descriptor onto another descriptor number, closing newfd first if it was open. It is primarily used by UNIX shells to implement input/output redirection between fork() and exec(). Example: 'dup2(file_fd, STDOUT_FILENO)' redirects all standard printf output to a file. Exam tip: dup2 is the key system call behind bash redirection symbols like '>' and '<'."
    },
    {
        "instruction": "Why can't an operating system simply use Shortest Job First (SJF) in a general-purpose desktop OS?",
        "input": "",
        "output": "SJF requires knowing the exact run time of each process in advance, which is impossible in a general-purpose system where user interactions and job lengths are unpredictable. Additionally, long jobs would suffer severe starvation if short jobs arrive continuously. Example: the OS cannot predict whether a user will type for 5 seconds or 5 hours in an editor. Exam tip: general-purpose OSes use adaptive algorithms like MLFQ instead, which approximate job length from past behavior."
    },
    {
        "instruction": "What is the convoy effect in CPU scheduling?",
        "input": "",
        "output": "The convoy effect is a phenomenon in non-preemptive FCFS/FIFO scheduling where a long, CPU-bound process holds the CPU while multiple short, I/O-bound processes wait idly behind it in the ready queue. This leads to poor resource utilization and high average turnaround time. Example: several cars stuck on a single-lane road behind a slow tractor. Exam tip: preemption solves the convoy effect by forcing the long job to yield."
    },
    {
        "instruction": "What is cooperative scheduling and why did modern operating systems abandon it?",
        "input": "",
        "output": "Cooperative scheduling relies on running processes to voluntarily yield CPU control periodically. Modern OSes abandoned it because an uncooperative or buggy process stuck in an infinite loop without system calls would freeze the entire computer, destroying system reliability and security. Example: early Windows 3.1 and Mac OS 9 were cooperative; a frozen app required a hard reboot. Exam tip: modern preemptive scheduling uses timer interrupts to ensure the OS always retains control."
    },
    {
        "instruction": "What is a system call number and how is it used during a trap?",
        "input": "",
        "output": "A system call number is an integer index assigned to each kernel function (e.g., sys_read = 0 on x86-64). When a user program issues a trap, it loads this number into a designated CPU register (such as %rax). The kernel trap handler inspects this register and uses a dispatch table to jump to the correct kernel implementation. Example: on Linux x86_64, writing 1 to %rax and executing 'syscall' calls sys_write. Exam tip: passing a number rather than an address prevents user programs from jumping to arbitrary kernel code."
    },
    {
        "instruction": "What information does the OS save onto the kernel stack during an interrupt?",
        "input": "",
        "output": "During a hardware interrupt, the CPU automatically saves the user process's Program Counter (PC), CPU status/flags register (EFLAGS), Stack Pointer (SP), and segment registers onto the process's kernel stack before switching execution to the interrupt handler. Example: this saved snapshot allows the exact resumption of the user program upon return-from-trap. Exam tip: the hardware pushes this minimal state automatically before the OS handler software executes."
    },
    {
        "instruction": "What is aging in priority scheduling algorithms?",
        "input": "",
        "output": "Aging is a technique to prevent starvation in priority scheduling by gradually increasing the priority of processes that spend extended time waiting in the ready queue. Eventually, even the lowest-priority process accumulates sufficient priority to run on the CPU. Example: incrementing a job's priority by 1 for every 100ms it spends waiting. Exam tip: aging ensures fairness and bounded waiting times in priority systems."
    },
    {
        "instruction": "What is the difference between turnaround time and throughput?",
        "input": "",
        "output": "Turnaround time is a latency metric measuring the elapsed time from arrival to completion for an individual process. Throughput is a rate metric measuring the total number of completed processes per unit of time across the entire system. Example: a system that finishes 10 jobs per minute has a throughput of 10 jobs/min, while each job took an average of 6 seconds turnaround. Exam tip: maximizing throughput does not always minimize turnaround time."
    },
    {
        "instruction": "Explain the concept of CPU burst vs I/O burst in process execution.",
        "input": "",
        "output": "Process execution alternates between cycles of CPU execution (CPU burst) and waiting for I/O operations (I/O burst). An I/O-bound process has frequent short CPU bursts separated by long I/O bursts; a CPU-bound process has rare, prolonged CPU bursts. Example: a database server querying disk alternates rapidly between short query parsing (CPU burst) and disk reads (I/O burst). Exam tip: schedulers exploit the CPU-burst duration to classify and prioritize jobs."
    },
    {
        "instruction": "What is the purpose of the init process in UNIX systems?",
        "input": "",
        "output": "The init process (PID 1, or systemd in modern Linux) is the first user-space process created during boot and acts as the direct or indirect ancestor of all other processes in the system. Its fundamental OS responsibility is adopting orphan processes and calling wait() on them to reap their zombie entries. Example: when a parent dies, its children are reparented to PID 1. Exam tip: PID 1 can never be killed and runs until the OS shuts down."
    },
    {
        "instruction": "What is a context switch latency and what factors affect it?",
        "input": "",
        "output": "Context switch latency is the time required to switch execution from one process to another. Factors affecting it include CPU architecture, register file size, OS kernel efficiency, memory bus speed, and the size of cache and TLB invalidations. Example: context switches typically take between 1 and 10 microseconds on modern x86 hardware. Exam tip: excessive context switching leads to thrashing and CPU wasted in non-productive work."
    },
    {
        "instruction": "Why does Round Robin scheduling yield poor turnaround times when all jobs are of similar length?",
        "input": "",
        "output": "When all jobs have identical lengths, Round Robin switches between them repeatedly so that they all progress at nearly the same rate and all finish near the very end of the total time. Under FIFO, jobs finish sequentially, giving much lower average turnaround time. Example: three 10-second jobs under FIFO finish at 10, 20, 30 (average 20s); under RR with 1s quantum, they finish at 28, 29, 30 (average 29s). Exam tip: RR optimizes response time, but is strictly worse than FIFO for jobs of equal duration."
    },
    {
        "instruction": "What is the difference between preemptive and non-preemptive scheduling?",
        "input": "",
        "output": "Preemptive scheduling allows the OS to interrupt an actively running process and move it to the ready queue before it finishes its CPU burst. Non-preemptive scheduling allows a process to hold the CPU until it explicitly yields, blocks on I/O, or terminates. Example: Round Robin and STCF are preemptive; FIFO and standard SJF are non-preemptive. Exam tip: preemptive scheduling requires hardware timer interrupt support."
    },
    {
        "instruction": "What is the primary role of the CPU scheduler versus the dispatcher?",
        "input": "",
        "output": "The scheduler is the policy engine that decides WHICH process from the ready queue should run next based on algorithmic criteria (e.g., priority, remaining time). The dispatcher is the low-level mechanism that actually executes the context switch, switches CPU modes, and jumps to the selected process. Example: the scheduler picks process B; the dispatcher loads B's registers and executes return-from-trap. Exam tip: scheduler = policy; dispatcher = mechanism."
    },
    {
        "instruction": "Explain the trade-off between throughput and fairness in operating systems.",
        "input": "",
        "output": "Throughput is maximized by minimizing context switch overhead and letting long jobs run uninterrupted. Fairness requires dividing CPU time equitably among all processes, which requires frequent context switches and time slices that waste cycles on overhead. Example: running a 1-hour job without interruption maximizes throughput but is completely unfair to interactive users. Exam tip: OS schedulers must balance throughput (system efficiency) and fairness (user responsiveness)."
    },
    {
        "instruction": "What is the purpose of the waitpid() system call over wait()?",
        "input": "",
        "output": "While wait() blocks until ANY child process terminates, waitpid(pid, status, options) allows a parent to specify the EXACT child PID to wait for. It also supports non-blocking checks using the WNOHANG flag, allowing the parent to poll child status without pausing. Example: 'waitpid(child_pid, &status, WNOHANG)' checks if a background job has finished while allowing the parent to continue executing. Exam tip: waitpid provides fine-grained control and non-blocking operation."
    },
    {
        "instruction": "What does the command 'kill -9 <PID>' do at the operating system level?",
        "input": "",
        "output": "The command sends signal number 9 (SIGKILL) to the specified process. Unlike regular signals (such as SIGTERM or SIGINT), SIGKILL cannot be caught, blocked, or ignored by user code; the OS kernel unconditionally terminates the process immediately. Example: stopping an unresponsive program that fails to close gracefully. Exam tip: SIGKILL and SIGSTOP are the only signals in UNIX that cannot be caught or ignored."
    }
]

# Generate variations to comfortably surpass 100 entries
viva_topics = [
    ("How does the OS manage file descriptors during fork()?",
     "Child processes inherit duplicate copies of the parent's file descriptor table, pointing to the same underlying open file descriptions in the kernel.",
     "This means seek offsets and file status flags are shared between parent and child.",
     "Always remember: file descriptor numbers are duplicated, but open file entries are shared in kernel memory."),
    ("What is an address space in an operating system?",
     "An address space is the virtual memory abstraction provided by the OS, giving each process the illusion of owning a contiguous block of private physical memory.",
     "The OS memory management unit (MMU) translates virtual addresses within this space into physical RAM frames.",
     "Never confuse virtual address space with actual physical RAM capacity."),
    ("What happens during a TLB shootdown?",
     "A TLB shootdown is a synchronization mechanism in multi-core systems where one core requests all other cores to flush invalid Translation Lookaside Buffer entries after modifying page tables.",
     "It is coordinated via inter-processor interrupts (IPIs) and introduces multi-core overhead.",
     "TLB shootdowns represent a major scalability challenge on large multi-core processors."),
    ("Why is Round Robin considered fair?",
     "Round Robin is considered fair because it distributes CPU time evenly among all ready processes in equal time slices without bias toward job length or arrival order.",
     "Every process with N jobs in the ready queue is guaranteed to run at least once every N time slices.",
     "Fairness in Round Robin sacrifices turnaround time for interactive jobs."),
    ("What is the difference between a process and a thread?",
     "A process has its own independent address space and system resources, while threads within the same process share that address space, heap, and open files.",
     "Threads have their own private stacks and registers but communicate easily via shared memory.",
     "Context switching between threads is faster than between processes because page tables do not need to be switched."),
    ("What is convoy effect prevention in modern CPU schedulers?",
     "Convoy effect prevention uses preemptive time-slicing so that no single long job can monopolize the CPU while short jobs wait.",
     "When the time slice expires, the long job is descheduled and moved to the back of the queue.",
     "Preemption is the standard cure for the convoy effect."),
    ("What is the difference between hard affinity and soft affinity in multi-core CPU scheduling?",
     "Soft affinity indicates the scheduler will try to keep a process on the same CPU core to preserve cache warmth, but may migrate it if another core is idle. Hard affinity strictly binds a process to specific CPU cores via system calls like sched_setaffinity.",
     "Hard affinity prevents cache thrashing for latency-critical applications.",
     "Affinity is primarily used to optimize CPU cache hit rates."),
    ("What is the purpose of the /proc virtual file system in Linux?",
     "/proc is a pseudo-filesystem generated dynamically by the kernel that exposes kernel data structures and process state information as human-readable text files.",
     "For example, /proc/<pid>/status displays current process state, memory usage, and thread counts.",
     "/proc exists entirely in RAM; it does not consume physical disk space."),
    ("What is the difference between a trap and a software interrupt?",
     "A trap and a software interrupt are functionally identical; both are synchronous CPU instructions that switch the CPU to kernel mode and invoke an OS vector handler.",
     "System calls are implemented using software interrupts or dedicated syscall instructions.",
     "Both use the interrupt vector table to locate the appropriate handler."),
    ("What is a race condition in process concurrency?",
     "A race condition is an undesirable situation where the outcome of concurrent operations depends on the uncontrollable timing or interleaving of process execution.",
     "It occurs when multiple processes access shared memory without proper mutual exclusion synchronization.",
     "Synchronization primitives like mutexes and semaphores are used to prevent race conditions.")
]

for q, direct, tech, tip in viva_topics:
    data.append({
        "instruction": q,
        "input": "",
        "output": f"Direct Answer: {direct} Technical Detail: {tech} Exam tip: {tip}"
    })

data.extend(expanded_entries)

# Add more MCQs to ensure a rich balance of all 3 styles
mcq_topics = [
    ("Which system call is used to replace the current process image with a new process image?\nA) fork()\nB) execvp()\nC) wait()\nD) exit()",
     "B) execvp()",
     "execvp() overwrites the code and memory of the calling process with a new program without changing the PID.",
     "fork creates a new process; exec replaces the existing one."),
    ("In UNIX, what process becomes the parent of an orphan process?\nA) The kernel process (PID 0)\nB) The init/systemd process (PID 1)\nC) The bash shell\nD) The scheduler process",
     "B) The init/systemd process (PID 1)",
     "When a parent process terminates before its child, the operating system reparents the orphan to PID 1, which reaps it upon exit.",
     "init/systemd is the universal parent for abandoned processes."),
    ("Which scheduling metric is defined as the time from job arrival until the job runs for the first time?\nA) Turnaround time\nB) Waiting time\nC) Response time\nD) Completion time",
     "C) Response time",
     "Response time is T_firstrun - T_arrival. It measures the initial responsiveness of the system, crucial for interactive tasks.",
     "Turnaround time measures total completion time; response time measures first run."),
    ("What hardware mechanism enables an operating system to implement preemptive scheduling?\nA) Translation Lookaside Buffer\nB) Direct Memory Access\nC) Hardware timer interrupt\nD) Memory Management Unit",
     "C) Hardware timer interrupt",
     "The timer interrupt fires periodically, allowing the kernel to preempt the currently executing process regardless of its state.",
     "Preemptive scheduling cannot exist without a hardware timer interrupt."),
    ("Which of the following is NOT shared between threads of the same process?\nA) Heap memory\nB) Open file descriptors\nC) Global variables\nD) CPU registers and stack",
     "D) CPU registers and stack",
     "Each thread must possess its own private call stack and register set (including Program Counter and Stack Pointer) to execute independently.",
     "Threads share code, data, heap, and files, but have private stacks and registers."),
    ("What state is a process in while waiting for a disk read to complete?\nA) Ready\nB) Running\nC) Blocked\nD) Terminated",
     "C) Blocked",
     "A process awaiting I/O cannot execute on the CPU until the I/O device completes the operation and signals an interrupt.",
     "Blocked processes cannot be scheduled until they transition back to Ready."),
    ("What is the primary advantage of Shortest Time-to-Completion First (STCF) over Shortest Job First (SJF)?\nA) It eliminates the convoy effect by allowing preemption\nB) It requires no knowledge of job length\nC) It eliminates all context switch overhead\nD) It guarantees equal CPU sharing",
     "A) It eliminates the convoy effect by allowing preemption",
     "STCF preempts a long-running job when a shorter job arrives, preventing short jobs from being stranded behind long jobs.",
     "STCF is the preemptive variant of SJF."),
    ("What does a return value of -1 from fork() indicate?\nA) The child process completed with an error\nB) The fork failed and no child process was created\nC) The process is running in kernel mode\nD) The process has become an orphan",
     "B) The fork failed and no child process was created",
     "When fork() returns -1, the operating system failed to create a new process (e.g., due to process table exhaustion or memory limits).",
     "Check fork() return value for -1 to handle process creation errors."),
    ("What occurs when a user program accesses memory outside its allocated address space?\nA) The CPU switches to cooperative mode\nB) The hardware generates a segmentation fault exception\nC) The program is placed in the Blocked state\nD) The scheduler reduces its priority",
     "B) The hardware generates a segmentation fault exception",
     "The MMU detects an invalid memory access and triggers a page fault or segmentation fault exception, causing the kernel to send SIGSEGV.",
     "Memory protection violations result in hardware traps, not soft errors."),
    ("What is the primary purpose of Multi-Level Feedback Queue (MLFQ) priority boosting?\nA) To maximize memory cache utilization\nB) To prevent starvation of CPU-bound processes in low-priority queues\nC) To increase the time slice of interactive tasks\nD) To eliminate the need for timer interrupts",
     "B) To prevent starvation of CPU-bound processes in low-priority queues",
     "Priority boosting periodically moves all processes to the top queue, guaranteeing that long-running batch jobs will never starve.",
     "Boost rule resets process priority to eliminate starvation.")
]

for q, ans, exp, tip in mcq_topics:
    data.append({
        "instruction": q,
        "input": "",
        "output": f"Correct Answer: {ans}. Explanation: {exp}. Exam tip: {tip}"
    })

# Add more concept explanations to reach 105+
additional_concepts = [
    ("What is memory virtualization?",
     "Memory virtualization is an operating system mechanism where each process is given the illusion of possessing a large, private, contiguous address space that is completely isolated from other processes.",
     "The operating system, with hardware support from the Memory Management Unit (MMU) and page tables, transparently translates virtual addresses generated by the CPU into physical RAM addresses.",
     "A pointer in program A with address 0x1000 and a pointer in program B with address 0x1000 point to completely distinct physical memory locations.",
     "Memory virtualization guarantees process isolation and protection; no process can overwrite another process's data without explicit sharing."),
    ("What is an address translation mechanism?",
     "Address translation is the hardware process where the Memory Management Unit (MMU) maps every virtual memory address issued by a running process to a physical RAM address.",
     "The CPU uses base and bounds registers or multi-level page tables to perform this translation at hardware speed for every instruction fetch, load, and store.",
     "When an instruction executes 'mov 0x20, %eax', the MMU hardware translates 0x20 into physical address 0x100020 using the process's base register.",
     "Address translation is performed in hardware on every memory reference, while the OS maintains the translation tables in memory."),
    ("What is a segmentation fault and why does it occur?",
     "A segmentation fault (SIGSEGV) is an error generated by the operating system when a process attempts to access a virtual memory address that is invalid, unmapped, or protected by access restrictions.",
     "The hardware MMU detects that the target virtual address has no valid page table entry or violates permissions (e.g., trying to write to a read-only code page) and triggers a fault trap into the kernel.",
     "Dereferencing a NULL pointer (address 0x0) triggers a segfault because the OS intentionally leaves page 0 unmapped to catch software bugs.",
     "A segfault indicates a memory protection violation detected by hardware MMU and enforced by the OS kernel."),
    ("What is the role of the TLB (Translation Lookaside Buffer)?",
     "The Translation Lookaside Buffer (TLB) is a small, ultra-fast hardware cache located directly on the CPU chip that stores recent virtual-to-physical address translations.",
     "Because accessing page tables in main memory for every memory instruction would cut performance in half, the MMU checks the TLB first (TLB hit); only if missing (TLB miss) does it walk the page table in RAM.",
     "Modern processors achieve TLB hit rates of 99% or higher, making virtual memory almost as fast as physical memory.",
     "On a context switch, the OS must flush the TLB or use Address Space Identifiers (ASIDs) to prevent the new process from using the old process's translations."),
    ("What is a page fault and how does the operating system handle it?",
     "A page fault is an exception raised by the MMU hardware when a process accesses a valid virtual page that is currently not resident in physical RAM (e.g., swapped out to disk or not yet loaded).",
     "The CPU traps into the kernel page fault handler, which locates the requested page on secondary storage (swap space), allocates a physical RAM frame, reads the page from disk, updates the page table, and restarts the faulting instruction.",
     "Demand paging loads executable pages into RAM only when the program first tries to execute them, avoiding loading unused code.",
     "A page fault is not an error; it is a normal, transparent mechanism used by virtual memory systems to support virtual spaces larger than physical RAM."),
    ("What is thrashing in operating system memory management?",
     "Thrashing is a pathological system state that occurs when the aggregate working sets of all running processes exceed the available physical RAM, causing the OS to spend more time swapping pages in and out of disk than executing instructions.",
     "Because disk access is thousands of times slower than RAM, system throughput plummets to near zero and the CPU sits mostly idle waiting for disk I/O.",
     "Opening 50 heavy applications simultaneously on a system with 4GB RAM causes the machine to become completely unresponsive as disk activity spikes to 100%.",
     "Thrashing can only be resolved by terminating processes, reducing working set size, or adding physical RAM."),
    ("What is the principle of locality in computer systems?",
     "The principle of locality states that programs tend to access a relatively small portion of their address space at any given time, exhibiting two distinct behaviors: temporal locality and spatial locality.",
     "Temporal locality means memory locations accessed recently are likely to be accessed again soon (e.g., loop variables). Spatial locality means memory locations near recently accessed addresses are likely to be accessed soon (e.g., array traversal).",
     "A 'for' loop iterating through an array exhibits both spatial locality (contiguous array elements) and temporal locality (loop counter variable).",
     "Virtual memory paging, CPU caches, and TLBs all depend directly on the principle of locality to achieve acceptable performance."),
    ("What is copy-on-write (COW) paging and when is it used?",
     "Copy-on-write is a memory optimization where multiple processes share the same physical pages marked read-only, duplicating a page only when one process attempts to modify it.",
     "When a write instruction executes on a shared page, the MMU raises a protection fault. The OS allocates a new physical frame, copies the page contents, updates the writing process's page table with write permission, and resumes execution.",
     "In UNIX fork(), the child initially shares all memory pages with the parent; if the child immediately executes exec(), no memory pages are ever duplicated.",
     "COW saves substantial memory and CPU cycles during process creation in UNIX."),
    ("What is internal fragmentation versus external fragmentation?",
     "Internal fragmentation occurs when allocated memory blocks are larger than requested, leaving wasted unused space INSIDE the allocated partition. External fragmentation occurs when free memory is broken into small scattered blocks, leaving total free space sufficient to satisfy a request, but no single block large enough.",
     "Paging eliminates external fragmentation by using fixed-size blocks (pages), but incurs minor internal fragmentation on the last page of an allocation.",
     "If a process needs 4097 bytes and page size is 4096 bytes, two pages (8192 bytes) are allocated, leaving 4095 bytes of internal fragmentation on the second page.",
     "Paging eliminates external fragmentation; variable-sized segmentation causes external fragmentation."),
    ("What is a daemon process in UNIX?",
     "A daemon is a long-running background process that operates without a controlling terminal, typically launched at system boot to provide ongoing system services.",
     "Daemons typically fork, dissociate from the controlling terminal (setsid), redirect stdin/stdout/stderr to /dev/null, and run continuously in a loop responding to network or system requests.",
     "Common examples include sshd (SSH daemon), cron (scheduled tasks), and httpd (web server).",
     "Daemons can be identified in 'ps' output by a '?' in the TTY column, indicating no attached terminal.")
]

def compile_qa_dataset():
    """Compiles the 100+ curated QA dataset and writes dataset.json + splits."""
    print("=" * 65)
    print("1. COMPILING INSTRUCTION FINE-TUNING DATASET")
    print("=" * 65)
    for title, d, expl, ex, tip in additional_concepts:
        data.append({
            "instruction": title,
            "input": "",
            "output": f"{d} {expl} Example: {ex}. Exam tip: {tip}"
        })

    print(f"Total dataset entries compiled: {len(data)}")
    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Saved dataset.json successfully!")

    # Reproducible train/val/test split (80/10/10, Seed 42)
    SEED = 42
    random.seed(SEED)
    shuffled = data.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]
    test_data = shuffled[n_train + n_val:]

    with open("train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open("val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open("test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"Dataset split complete: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")


def find_source_pdf():
    """Locates os.pdf in data/ or root directory."""
    candidates = [
        os.path.join("data", "os.pdf"),
        "os.pdf"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Could not find 'os.pdf' in 'data/' or project root.")


def clean_page_text(text):
    """
    Cleans raw PDF text extraction artifacts:
    - Normalizes common unicode ligatures and dashes
    - Reconnects hyphenated word breaks across lines
    - Removes isolated running headers/footers and page numbers
    - Normalizes consecutive whitespace
    """
    if not text:
        return ""
    # Ligatures and typography
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\u2014", " - ").replace("\u2013", "-")
    # Hyphenated line breaks (e.g. vir- \n tualization -> virtualization)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Remove isolated running headers / page numbers
    lines = [l.strip() for l in text.split("\n")]
    filtered = []
    for l in lines:
        if re.match(r"^\d+$", l):
            continue
        # Running headers like "10 THE ABSTRACTION : THE PROCESS" or "DISTRIBUTED SYSTEM SECURITY 15"
        if re.match(r"^\d+\s+[A-Z\s:—–-]+$", l) or re.match(r"^[A-Z\s:—–-]+\s+\d+$", l):
            continue
        filtered.append(l)
    text = " ".join(filtered)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text_by_sentence(text, target_size=450, min_size=100):
    """
    Splits text into concise, self-contained chunks on sentence boundaries.
    Avoids arbitrary character cuts or broken sentences to preserve academic clarity
    and prevent bloated retrieval contexts.
    """
    if not text:
        return []
    raw_sentences = re.split(r"(?<=[.?!])\s+(?=[A-Z0-9\"\'\u2018\u201c])", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    chunks = []
    curr = []
    curr_len = 0
    for s in sentences:
        if curr_len + len(s) + 1 > target_size and curr_len >= min_size:
            chunk_str = " ".join(curr).strip()
            if len(chunk_str) >= min_size:
                chunks.append(chunk_str)
            curr = [s]
            curr_len = len(s)
        else:
            curr.append(s)
            curr_len += len(s) + 1
    if curr:
        chunk_str = " ".join(curr).strip()
        if len(chunk_str) >= min_size:
            chunks.append(chunk_str)
    return chunks


def build_vector_store():
    """
    Extracts text from os.pdf, cleans and chunks passages, removes stale data,
    computes embeddings with all-MiniLM-L6-v2, and builds the ChromaDB collection.
    """
    print("\n" + "=" * 65)
    print("2. BUILDING VECTOR KNOWLEDGE BASE FROM os.pdf")
    print("=" * 65)

    pdf_path = find_source_pdf()
    print(f"Reading source PDF: {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"Total pages in PDF: {total_pages}")

    all_chunks = []
    all_metadata = []
    all_ids = []
    seen_chunks = set()

    extracted_pages = 0
    chunk_counter = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            raw_text = page.extract_text()
        except Exception:
            raw_text = ""

        if not raw_text or not raw_text.strip():
            continue

        extracted_pages += 1
        cleaned = clean_page_text(raw_text)
        page_chunks = chunk_text_by_sentence(cleaned, target_size=450, min_size=100)

        for c in page_chunks:
            # Normalize for de-duplication
            norm_key = re.sub(r"\W+", "", c.lower())
            if norm_key in seen_chunks or len(norm_key) < 40:
                continue
            seen_chunks.add(norm_key)

            all_chunks.append(c)
            all_metadata.append({"file": "os.pdf", "page": page_num})
            all_ids.append(f"os_p{page_num}_{chunk_counter}")
            chunk_counter += 1

    print(f"Extracted {extracted_pages}/{total_pages} non-empty pages.")
    avg_len = sum(len(c) for c in all_chunks) // max(1, len(all_chunks))
    print(f"Generated {len(all_chunks)} unique, coherent chunks (avg length: {avg_len} chars).")

    # Connect to ChromaDB and wipe stale data
    print("\nConnecting to ChromaDB and removing stale collections...")
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    try:
        chroma_client.delete_collection("os_notes")
        print("Removed previous 'os_notes' collection (stale notes.pdf data purged).")
    except Exception:
        print("No prior 'os_notes' collection to delete.")

    collection = chroma_client.create_collection("os_notes")

    # Device selection and embedding
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading embedding model 'all-MiniLM-L6-v2' on {device}...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    print(f"Encoding {len(all_chunks)} chunks...")
    batch_size = 64
    all_embeddings = model.encode(all_chunks, batch_size=batch_size, show_progress_bar=True).tolist()

    # Add to ChromaDB in batches of 500
    print("Writing new embeddings to ChromaDB collection 'os_notes'...")
    chroma_batch_size = 500
    for i in range(0, len(all_chunks), chroma_batch_size):
        end = i + chroma_batch_size
        collection.add(
            ids=all_ids[i:end],
            documents=all_chunks[i:end],
            embeddings=all_embeddings[i:end],
            metadatas=all_metadata[i:end]
        )

    print(f"Successfully saved {collection.count()} chunks into ChromaDB ('chroma_db/os_notes').")
    print("All chunks originate exclusively from 'os.pdf'. Stale data completely removed.")


if __name__ == "__main__":
    compile_qa_dataset()
    build_vector_store()
    print("\n" + "=" * 65)
    print("DATASET & VECTOR DATABASE BUILD COMPLETE!")
    print("=" * 65)

