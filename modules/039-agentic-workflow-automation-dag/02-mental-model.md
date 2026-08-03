# 02 - Mental Model: The Assembly Line with Traffic Lights

## The Metaphor
Imagine a highly advanced, automated **assembly line** building a complex product. 

- **The Stations (Nodes)**: Each station performs a specific operation on the product. Some stations can work in parallel, but others must wait for previous stations to finish.
- **The Conveyor Belts (Edges)**: The belts connect the stations, moving the product strictly forward. The belts never loop back—it's a Directed Acyclic Graph (DAG).
- **The Product (State)**: The item being built represents the shared `State`. As it moves from station to station, it gets updated and modified.
- **Traffic Lights (Human-in-the-Loop)**: Some stations handle highly sensitive operations (e.g., final quality approval). These stations have a red traffic light. The assembly line pauses, and a human operator must inspect the product and press a green button (approval) for the line to resume.
- **Save Points (Checkpoints & Rollbacks)**: Before a station modifies the product, it takes a perfect 3D scan (Checkpoint). If the machine breaks the product (Failure), it rolls back to the 3D scan, allowing a retry without losing the entire product's progress.

## Why this model?
Agentic workflows often suffer from unpredictability and infinite loops. By forcing them into an "assembly line with traffic lights" (a State Graph DAG), we maintain the flexibility of agents while enforcing strict boundaries, error recovery, and human oversight.
