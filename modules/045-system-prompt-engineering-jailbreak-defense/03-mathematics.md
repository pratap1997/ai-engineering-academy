# Mathematics of Prompt Defense

## Attack Success Rate (ASR)
A standard metric to evaluate jailbreak defense is the Attack Success Rate:
$$ASR = \frac{N_{jailbreak}}{N_{attempts}}$$
Where $N_{jailbreak}$ is the number of successful jailbreaks and $N_{attempts}$ is the total number of attacks.

## Delimiter Entropy
When using random nonces to enclose untrusted input, the probability of an attacker guessing or colliding with the delimiter is:
$$P_{collision} = \frac{1}{\Sigma^L}$$
Where $\Sigma$ is the alphabet size (e.g., 62 for alphanumeric) and $L$ is the length of the nonce.

## Dual-LLM Guardrail Confidence
A smaller, specialized LLM evaluates input for malicious intent, outputting a safety confidence score $C \in [0, 1]$. We enforce a security threshold $\theta_{sec}$:
If $C \ge \theta_{sec}$, process the input. Otherwise, reject.
