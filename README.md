# Energy-Efficient CPU Scheduling Algorithm

An advanced CPU scheduling algorithm designed to minimize energy consumption while maintaining good performance for mobile and embedded systems.

## Features

- Energy-aware process scheduling
- Dynamic Voltage and Frequency Scaling (DVFS)
- Priority-based scheduling with energy optimization
- Multi-core support
- Simulation capabilities
- Performance metrics tracking

## Key Components

1. **Energy Efficiency Calculation**
   - Base energy consumption per unit time
   - Priority-based energy penalty
   - Burst time optimization
   - Dynamic voltage scaling

2. **Scheduling Algorithm**
   - Combines energy efficiency with priority
   - Considers process arrival times
   - Multi-core load balancing
   - Dynamic resource allocation

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the simulation:
```bash
python energy_scheduler.py
```

## Performance Metrics

- Total Energy Consumption
- Average Energy per Process
- Average Wait Time
- Average Response Time

## Project Structure

- `energy_scheduler.py`: Main scheduling algorithm implementation
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation
