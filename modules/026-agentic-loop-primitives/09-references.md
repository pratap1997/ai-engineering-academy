# References: Agentic Loop Primitives

The following papers represent the foundational literature for Agentic Loops, Tool Use, and Planning in modern AI systems.

1. **ReAct: Synergizing Reasoning and Acting in Language Models**
   - *Authors:* Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
   - *Provenance:* [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
   - *Significance:* Introduced the ReAct paradigm, demonstrating that interleaving reasoning traces with task-specific actions improves both interpretability and success rates compared to acting or reasoning alone.

2. **Toolformer: Language Models Can Teach Themselves to Use Tools**
   - *Authors:* Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom (2023)
   - *Provenance:* [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
   - *Significance:* Demonstrated that language models can be trained to decide when to call APIs and how to parse the responses in a self-supervised manner, formalizing the tool-use paradigm.

3. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**
   - *Authors:* Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (2023)
   - *Provenance:* [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
   - *Significance:* Extended the linear chain of reasoning into a tree search, allowing agents to explore multiple reasoning paths, evaluate them, and backtrack if necessary.

4. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
   - *Authors:* Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
   - *Provenance:* [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
   - *Significance:* The precursor to ReAct. It established that forcing models to output intermediate reasoning steps drastically improves performance on complex reasoning tasks.

5. **Finite-time Analysis of the Multiarmed Bandit Problem (UCB1)**
   - *Authors:* Peter Auer, Nicolo Cesa-Bianchi, Paul Fischer (2002)
   - *Provenance:* Journal of Machine Learning Research, 3(May):235-256.
   - *Significance:* Provides the mathematical foundation (UCB1 algorithm) used in advanced agentic loops for optimal tool selection balancing exploration and exploitation.

6. **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation**
   - *Authors:* Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang (2023)
   - *Provenance:* [arXiv:2308.08155](https://arxiv.org/abs/2308.08155)
   - *Significance:* Introduced a comprehensive framework for multi-agent loops, demonstrating how ReAct-like agents can converse and collaborate to solve complex engineering tasks.
