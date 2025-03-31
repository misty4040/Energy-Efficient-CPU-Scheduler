import numpy as np
from typing import List, Dict, Tuple
import simpy
import random
import matplotlib.pyplot as plt

class Process:
    def __init__(self, pid: int, arrival_time: float, burst_time: float, priority: int):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.priority = priority
        self.remaining_time = burst_time
        self.energy_consumption = 0.0
        self.wait_time = 0.0
        self.response_time = 0.0

class EnergyEfficientScheduler:
    def __init__(self, num_cores: int = 1):
        self.num_cores = num_cores
        self.current_time = 0
        self.total_energy = 0
        self.processes = []
        self.completed_processes = []
        
    def calculate_energy(self, process: Process) -> float:
        """Calculate energy consumption based on process characteristics"""
        # Base energy consumption
        base_energy = process.burst_time * 0.1  # Base energy per unit time
        
        # Energy penalty for priority (higher priority = more energy)
        priority_penalty = process.priority * 0.05
        
        # Energy savings for longer burst times (batch processing)
        burst_savings = max(0, (process.burst_time - 1) * 0.02)
        
        return base_energy + priority_penalty - burst_savings
    
    def dynamic_voltage_scaling(self, load: float) -> float:
        """Adjust CPU voltage based on current load"""
        # Voltage scales between 0.8V (min) and 1.2V (max)
        min_voltage = 0.8
        max_voltage = 1.2
        return min_voltage + (max_voltage - min_voltage) * (load / 100)
    
    def schedule(self, processes: List[Process]) -> List[Process]:
        """Main scheduling algorithm"""
        # Sort processes by energy efficiency score
        processes.sort(key=lambda p: (
            self.calculate_energy(p) / p.burst_time,  # Energy efficiency
            -p.priority,  # Higher priority first
            p.arrival_time  # Arrival time
        ))
        
        return processes
    
    def simulate(self, processes: List[Process], simulation_time: float):
        """Run the simulation"""
        env = simpy.Environment()
        cores = [simpy.Resource(env, capacity=1) for _ in range(self.num_cores)]
        
        def process_runner(env, process, core):
            with core.request() as request:
                yield request
                
                # Calculate energy consumption
                process.energy_consumption = self.calculate_energy(process)
                self.total_energy += process.energy_consumption
                
                # Simulate process execution
                yield env.timeout(process.burst_time)
                
                # Update statistics
                process.wait_time = env.now - process.arrival_time - process.burst_time
                process.response_time = env.now - process.arrival_time
                
                self.completed_processes.append(process)
        
        for process in processes:
            env.process(process_runner(env, process, random.choice(cores)))
        
        env.run(until=simulation_time)
        
        return self.completed_processes

def generate_test_processes(num_processes: int) -> List[Process]:
    """Generate test processes for simulation"""
    processes = []
    for i in range(num_processes):
        arrival_time = random.uniform(0, 100)
        burst_time = random.uniform(1, 20)
        priority = random.randint(1, 5)
        processes.append(Process(i, arrival_time, burst_time, priority))
    return processes

def plot_results(completed_processes):
    """Plot the scheduling results"""
    # Sort processes by completion time
    completed_processes.sort(key=lambda p: p.arrival_time + p.burst_time)
    
    # Create timeline plot
    plt.figure(figsize=(12, 6))
    
    # Plot each process
    for process in completed_processes:
        start = process.arrival_time
        end = start + process.burst_time
        plt.barh(process.pid, process.burst_time, left=start, height=0.5,
                label=f'P{process.pid} (E={process.energy_consumption:.2f})')
    
    plt.xlabel('Time Units')
    plt.ylabel('Process ID')
    plt.title('CPU Scheduling Timeline')
    plt.grid(True, axis='x')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    # Initialize scheduler
    scheduler = EnergyEfficientScheduler(num_cores=2)
    
    # Generate test processes
    processes = generate_test_processes(20)
    
    # Run simulation
    completed = scheduler.simulate(processes, simulation_time=200)
    
    # Calculate statistics
    avg_energy = sum(p.energy_consumption for p in completed) / len(completed)
    avg_wait = sum(p.wait_time for p in completed) / len(completed)
    avg_response = sum(p.response_time for p in completed) / len(completed)
    
    print(f"\nSimulation Results:")
    print(f"Total Energy Consumption: {scheduler.total_energy:.2f} units")
    print(f"Average Energy per Process: {avg_energy:.2f} units")
    print(f"Average Wait Time: {avg_wait:.2f} units")
    print(f"Average Response Time: {avg_response:.2f} units")
    
    # Plot results
    plot_results(completed)

if __name__ == "__main__":
    main()
