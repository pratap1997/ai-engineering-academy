# Mental Model: The Chess Grandmaster

Think of MCTS and Tree of Thoughts like a chess grandmaster planning their moves. 
When looking at the board, the grandmaster doesn't just play the first move that comes to mind (which is akin to zero-shot or greedy generation). Instead, they:
- Consider a few candidate moves (Expansion).
- Imagine the sequence of counter-moves and responses (Simulation/Rollout).
- Evaluate if the resulting board state is advantageous.
- Update their mental estimation of the original candidate move (Backpropagation).
- Choose to either explore an alternative candidate or dive deeper into the current one (Selection).
