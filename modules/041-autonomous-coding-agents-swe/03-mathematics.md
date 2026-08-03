# Mathematics of Code Repair

## Edit Distance (Levenshtein)
To quantify how much an agent changed a file, we measure the Levenshtein distance:
$$ d(i,j) = \min(d(i-1,j)+1, d(i,j-1)+1, d(i-1,j-1) + \text{cost}) $$

## Diff Similarity Ratio
$$ R = \frac{2M}{T} $$
Where $M$ is the number of matching characters, and $T$ is the total number of characters in both sequences.

## Repair Convergence Probability
The probability of eventually passing the test suite after $k$ attempts, assuming independent success probability $p$ on each repair attempt:
$$ P(\text{pass} | k \text{ attempts}) = 1 - (1-p)^k $$
