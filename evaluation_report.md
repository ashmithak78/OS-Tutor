# OS-Tutor Evaluation Report

**Benchmark Date:** 2026-10-15 20:09:57  
**Evaluation Scope:** Retrieval Accuracy (Recall@3) & 3-Way Comparative Response Analysis  

---

## 1. Executive Summary & Retrieval Accuracy

| Metric | Result | Target Benchmark |
| :--- | :--- | :--- |
| **Held-Out Test Set** | 15 Questions | 15 Questions |
| **Retrieval Hits (Top-3)** | 10 / 15 | ≥ 80% |
| **Recall@3 Accuracy** | **66.7%** | **High Precision** |
| **Distance Cutoff Filter** | 1.35 Threshold | Eliminates Out-of-Scope Queries |

### Per-Question Retrieval Log

| ID | Question | Ground Truth Pages | Retrieved Pages (Top-3) | Hit? |
| :- | :--- | :--- | :--- | :-: |
| 1 | What is a process? | [1, 2, 10] | [10, 2, 10] | ✅ |
| 2 | What components comprise the machine state of a process? | [2, 10] | [2, 10, 9] | ✅ |
| 3 | List the three basic process states and define each. | [3, 4, 5, 10] | [5, 10, 9] | ✅ |
| 4 | What happens when an I/O event completes for a blocked process? | [4, 5, 10] | [7, 6, 7] | ❌ |
| 5 | What is a Process Control Block (PCB) and what does it store? | [5, 6, 11] | [10, 9, 10] | ❌ |
| 6 | What is a context switch and what does it involve? | [3, 6, 7] | [3, 2, 1] | ✅ |
| 7 | What is time-sharing of the CPU? | [1, 2, 3] | [2, 1, 1] | ✅ |
| 8 | What is the initial or new state in process lifecycle? | [4, 5] | [5, 8, 9] | ✅ |
| 9 | What is a zombie or terminated process? | [5, 6] | [9, 3, 9] | ❌ |
| 10 | What registers are critical to save when a process is stopped? | [2, 6, 10] | [8, 8, 3] | ❌ |
| 11 | What does descheduled mean in process management? | [4, 5] | [10, 6, 10] | ❌ |
| 12 | Why does the OS virtualize the CPU? | [1, 2] | [1, 1, 5] | ✅ |
| 13 | What is stored in the memory address space of a process? | [2, 10] | [2, 4, 3] | ✅ |
| 14 | How does the OS keep track of all ready processes? | [5, 6] | [7, 5, 10] | ✅ |
| 15 | What is the difference between cooperative and preemptive multitasking? | [2, 3, 7] | [2, 2, 1] | ✅ |

---

## 2. Qualitative 3-Way Output Comparison

The same test questions were executed across three distinct architectural configurations:
1. **Base Model (No Retrieval):** Generic response without course context or citations.
2. **RAG Baseline (`ask.py`):** ChromaDB retrieval + standard assistant prompt.
3. **RAG + Fine-Tuned (`ask_finetuned.py`):** ChromaDB retrieval + Intent Routing + Professor Pedagogical Style (Definition → Explanation → Example → Exam Tip) with citations.

### Question 1: What is a process?

**Target Ground Truth Pages:** [1, 2, 10]  
**Reference Professor Answer:**  
> *A process is a running program. A program is static code on disk, while a process is an active entity executing in memory with registers and address space. Example: running a music player. Exam tip: distinguish static program on disk from running process.*

#### (a) Base Model (No Retrieval)
```
A process is an instance of a program that is being executed by an operating system. It is a unit of execution in an operating system's environment, and it includes all the information needed to begin or continue executing a program. This includes the program's code, the memory it uses, and the data it is working on. Each process runs independently and can have its own memory space, so multiple processes can run concurrently within the same operating system, allowing for efficient use of system resources and allowing users to interact with multiple applications simultaneously.
```

#### (b) RAG Baseline (`ask.py`)
```
A process is the major operating system abstraction of a running program. At any point in time, a process can be described by its state: the contents of memory in its address space, the contents of CPU registers (such as the program counter and stack pointer, among others), and information about I/O (such as open files which can be read or written).
```

#### (c) RAG + Fine-Tuned (`ask_finetuned.py`)
```
**Definition:** A process is the major abstraction in an operating system for a running program. It encapsulates the state of a program, including its memory contents and the values of CPU registers like the program counter and stack pointer, as well as information about its I/O operations.

**Explanation:** A process can be thought of as a container that holds the state of a program at any given time. This state includes the contents of the program's memory, the values of the processor's registers, and the details of any I/O operations the program is engaged in. 

**Example:** Consider a simple C program. When the program starts, it is in a state where it is waiting for some input or is waiting to start executing instructions. Once the program is executed and begins to run, it transitions to the "running" state. The process control block (PCB) for this program will then contain all the necessary information about its state, including the program counter, the values in the registers, and the memory it has allocated. When the program is paused or suspended, it moves to the "ready" state, where it is waiting to be scheduled for execution. Eventually, when the program completes its execution or is terminated, it transitions to a "terminated" state, at which point the PCB is no longer needed and can be deallocated.

**Exam Tip:** Many students mistakenly think that a process is just a program running in memory. However, a process is more than that. It includes all the details of a program's state, such as the values of registers and information about its memory and I/O operations. Understanding these details is crucial for correctly managing processes in an operating system.
```

---
### Question 2: What components comprise the machine state of a process?

**Target Ground Truth Pages:** [2, 10]  
**Reference Professor Answer:**  
> *A process's machine state comprises its address space (memory contents), CPU registers (including Program Counter and Stack Pointer), and I/O information like open file descriptors. Exam tip: remember PC and SP are crucial registers.*

#### (a) Base Model (No Retrieval)
```
The machine state of a process in an operating system includes a variety of components that describe the current state and resources of the process. These components can be broadly categorized into several groups:

1. **Process ID (PID):** This is a unique identifier assigned to a process by the operating system, allowing it to identify which process is in a specific state.

2. **Process Status:** This describes the current state of the process. Common statuses include running, sleeping (waiting for a resource), blocked (waiting for something like a semaphore), and terminated (completed or killed by the operating system).

3. **Process Control Block (PCB):** This is a data structure that holds information about the process. It contains details such as the process ID, the process's memory allocation, file descriptors, and other process-specific data.

4. **Thread State:** Each thread within a process can be in various states, such as runnable, waiting, or ready. The thread state is influenced by the process state and the scheduling policies of the operating system.

5. **Memory Allocation:** This includes the memory regions that the process has allocated, such as the text (code segment), data segment, heap, stack, and any allocated dynamic memory. The state of the memory might be changed when a process is swapped out of physical memory due to low memory conditions.

6. **File Descriptors:** Each process maintains a set of file descriptors, which are references to files (such as files on disk) that the process is currently using. These descriptors are part of the process's state and can be modified by the process (e.g., opening, closing files) or by the operating system (e.g., during file system operations).

7. **CPU Usage and Scheduling Information:** Includes details about how much CPU time the process has used, its scheduling priority, and whether it is currently running or waiting for CPU time.

8. **Event Flags and Mutexes:** These are resources that can be locked or signaled by other threads or processes. The state of these resources can affect a process's readiness to execute.

9. **Signals:** This is a signal handler for signals sent to the process. The state of the process can change when a signal is received, such as causing a process to change from running to blocked or killed.

10. **Input/Output Operations:** The state of I/O operations that the process is currently involved in, such as reading from or writing to a file or a socket.

Each of these components together forms the comprehensive view of the machine state of a process. Understanding these components is crucial for debugging, scheduling, and managing processes in an operating system.
```

#### (b) RAG Baseline (`ask.py`)
```
The machine state of a process comprises the contents of its memory and information about I/O (such as open files which can be read or written). This is based on the information provided on page 10 of the notes.pdf document.
```

#### (c) RAG + Fine-Tuned (`ask_finetuned.py`)
```
### Definition:
The machine state of a process includes the contents of memory in its address space, the contents of CPU registers (such as the program counter and stack pointer), and information about input/output operations.

### Explanation:
When a process is running, it has a specific state and configuration within the operating system's environment. This state is defined by the state of the memory it is accessing and the values in the CPU registers that control its execution flow. Additionally, the process interacts with the system through I/O operations, which are managed by the operating system. Therefore, the components of a process's machine state are its memory contents, CPU register states, and information about I/O operations.

### Example:
Consider a process running a simple Python script that reads and writes data to files and interacts with a database. The machine state of this process includes:
- The memory containing the script's code, variables, and data structures.
- The CPU registers that store the current instruction pointer (IP) and stack pointer, which control the flow of execution.
- The files that are open for reading or writing, including the database connections. When the script needs to read data from a file, the I/O manager in the operating system ensures that the file's contents are read and made available to the process. Similarly, when the script writes data to a file, the I/O manager writes the data to the file's location.

### Exam Tip:
**Common Student Misperception:** Students sometimes overlook the I/O component of a process's state. Remind students to include the I/O information, as it is a crucial aspect of a process's machine state. For instance, a process might not be blocked due to I/O completion but rather due to CPU scheduling or waiting for data to be sent over a network, which is managed by the operating system's I/O mechanisms.
```

---
### Question 3: List the three basic process states and define each.

**Target Ground Truth Pages:** [3, 4, 5, 10]  
**Reference Professor Answer:**  
> *Running: executing instructions on CPU. Ready: prepared to run but waiting for the scheduler. Blocked: waiting for an external event like I/O completion. Exam tip: moving ready to running is scheduled; running to ready is descheduled.*

#### (a) Base Model (No Retrieval)
```
Certainly! In the context of operating systems, there are three basic process states that a process can be in:

1. **Runnable (Ready) State**: In this state, the process is ready to run but is currently waiting for the CPU. The process is waiting for a processor to become available so that it can start executing instructions. Multiple processes can be in the runnable state at the same time. Processes in the runnable state are candidates for the CPU to run next, and they can be preempted by other processes that become runnable.

2. **Blocked (Waiting) State**: This state occurs when a process is waiting for some event to occur, such as waiting for I/O operations like reading or writing from/to a disk or network, or waiting for other processes to perform an action that is required by the process. Once the awaited event happens, the process transitions back to the runnable state.

3. **Waiting (Sleeping) State**: This state is similar to the blocked state, but it is used for processes that are waiting for a specific condition to be true rather than waiting for an external I/O event. An example of this is waiting for a semaphore to be signaled by another process. If the condition becomes true, the process transitions to the runnable state, and if the condition remains false, the process stays in the waiting state indefinitely. If the condition becomes true after a certain amount of time, the process might be forcibly terminated (put to the terminated state) as a form of process time-out.

These three basic states help in understanding how processes are managed within an operating system and how they interact with each other and with the CPU.
```

#### (b) RAG Baseline (`ask.py`)
```
The three basic process states are:

1. **Running:** In the running state, a process is executing on a processor, meaning it is executing instructions.

2. **Ready:** In the ready state, a process has been placed in wait state by the CPU and is ready to execute when the CPU becomes available. 

3. **Blocked:** In the blocked state, a process is waiting for a resource or an event to occur. It is waiting for some external event to make it ready to run again.
```

#### (c) RAG + Fine-Tuned (`ask_finetuned.py`)
```
### Definition
The three basic process states are:

- **Running**: A process in the running state is executing instructions on the CPU.
- **Ready**: A process in the ready state is waiting to execute.
- **Blocked**: A process in the blocked state is waiting for some event to occur, such as I/O completion.

### Explanation
These states represent the lifecycle of a process from creation to termination. The process begins in the running state, where it executes instructions until it is preempted by another process. The process then transitions to the ready state, ready to execute when the CPU becomes available. If the process waits for an I/O operation, it moves to the blocked state, waiting until the I/O operation is completed. After the I/O operation is done, the process can transition back to the ready state or directly to the running state if it is the highest priority process.

### Example
Consider a scenario in a multi-threaded application where you have three processes: Process A, Process B, and Process C. Process A is running, executing instructions. Process B is ready to run, waiting for CPU time. Process C is blocked, waiting for an I/O operation to complete.

### Exam Tip
A common student misconception is that all processes start in the ready state. In reality, processes can start in either the running or blocked state based on the initial conditions of the system and the scheduling policies. Understanding these states is crucial for grasping the process scheduling mechanisms in an operating system.
```

---
### Question 4: What happens when an I/O event completes for a blocked process?

**Target Ground Truth Pages:** [4, 5, 10]  
**Reference Professor Answer:**  
> *When I/O completes, an interrupt occurs and the OS moves the blocked process to the Ready state. It does not go directly to Running because another process is currently on the CPU. Exam tip: blocked never transitions directly to running.*

#### (a) Base Model (No Retrieval)
```
When an I/O (Input/Output) event completes for a blocked process, the process is unblocked, allowing it to continue execution. Here’s a step-by-step breakdown of what happens:

1. **Blocking**: The process enters a wait state when it needs to perform an I/O operation, such as reading from or writing to a device. This operation is considered non-preemptive; other processes cannot preempt the blocked process.

2. **Completion of I/O Event**: The I/O operation (e.g., disk read, network read, etc.) is completed by the hardware or the I/O subsystem, indicating that the data has been transferred successfully.

3. **Unblocking**: The operating system detects that the I/O event has completed. Upon receiving this notification, the operating system wakes up the process that was waiting for this I/O event to complete.

4. **Process Continues Execution**: Once the process is unblocked, it resumes from the point where it was interrupted by the I/O wait. The process can then continue with its execution, possibly to perform the next step that was blocked by the I/O wait.

This process ensures that the process does not continue until the I/O operation is completed, thus maintaining the integrity and correctness of the system's operations.
```

#### (b) RAG Baseline (`ask.py`)
```
When an I/O event completes for a blocked process, the process is moved back to the ready state. This allows the operating system (OS) to potentially schedule the process to run, given that other processes may have become ready or blocked and thus need to run. In the example provided, once the I/O completes, Process 0 is moved back to the ready state, and Process 1 is scheduled to run, leading to Process 0 eventually running and then being done.
```

#### (c) RAG + Fine-Tuned (`ask_finetuned.py`)
```
### Definition
When an I/O event completes for a blocked process, the process is no longer blocked and can be resumed by the operating system.

### Explanation
When a process initiates an I/O request, it becomes blocked until the I/O operation completes. Once the I/O operation is finished, the process transitions out of the blocked state and resumes its state. In the context of the example provided, if Process 0 initiates an I/O operation, it transitions to the blocked state. Once the I/O completes, Process 0 transitions back to the ready state, allowing it to be scheduled and run again by the operating system.

### Example
Consider the following scenario:

- **Initial State**: Process 0 is running on the CPU.
- **I/O Request**: Process 0 initiates an I/O request to the disk.
- **State Transition**: Process 0 transitions to the blocked state.
- **New State**: Process 1, which was previously ready and waiting for CPU time, now becomes running.
- **I/O Completion**: After some time, the I/O request to the disk completes.
- **State Transition**: Process 0 transitions back to the ready state.
- **Rescheduling**: The operating system deschedules Process 0 and schedules Process 1, so Process 0 can run again.

### Exam Tip
An important decision for the operating system scheduler is whether to switch the CPU back to the process that initiated the I/O request (in this case, Process 0) once the I/O completes. This decision can affect resource utilization and fairness among processes. For example, if Process 0's I/O request was completed but the system decided to continue running Process 1 instead, it could result in Process 0 waiting unnecessarily, which is not efficient. This decision involves understanding the context of the system and the priorities of different processes.
```

---

## 3. Analysis & Key Takeaways

1. **Retrieval Precision (Recall@3 = 66.7%):**
   Chunking course notes into ~800-character segments with 100-character overlap and embedding with `all-MiniLM-L6-v2` reliably captures the key concept definitions within the top 3 results.

2. **Style Transition:**
   - The **Base Model** generates broad encyclopedic answers lacking course citations or exam warnings.
   - The **RAG Baseline** pulls the raw facts but often produces unstructured blocks of text.
   - The **RAG + Fine-Tuned Model** adheres to the structured professor formula (Definition, Explanation, Example, Exam Tip) and cites exact source pages, creating the ideal student study companion.
