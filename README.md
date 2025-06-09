# Thomas Joshi's Guide to AI Engineering Mastery

## MCP

### [MCP Summit](https://mcpdevsummit.ai/#page-1)
1. OpenAI - Nick Cooper - Future of MCP 
    * Specialization: MCP tools that can make policy decisions, language translation
    * Federation: Merging results from different servers
    * Steering: Information changes over time, and needs to update model context as it reasons

2. Acorn Labs - Darren Sherpder CTO - Why Are Agents So Hard?
    * MCP replaces all components taking it from retrieval, tool calling, and writing from [Anthropic Agents Guide](https://www.anthropic.com/engineering/building-effective-agents) to replacing everythign with an MCP tool
    * Tools used in his workflow: playwright for testing, [FastMCP](https://github.com/jlowin/fastmcp), [Nanobot](https://github.com/nanobot-ai/nanobot) 
    * Async coding agents are super easy to build because you can just give the file system and commands to manipulate file systems
    * Outstanding problem: distributed tracing

3. Paypal - Brendan Lane, Nitin Sharma - Building an MCP Server for Agentic Commerce
    * You can do online shopping through Claude, describe the item you are looking to purchase, it finds the item, and authenticates with PayPal within the Claude chat window --> changes non exploratory ecommerce
    * Any chat interface can call get_paypal_mcp_tools to enable commerce enabled chatbots
    * How to design an MCP Server? 
        * Tool ownership & scope: Who owns the tool? Includes traditional access management, logging, schema definition
        * Tool function & behavior: What does the tool do? Single repsonsiblity just like microservices design
        * Tool evaluation & life cycle: How does the tool evolve? Monitoring usage patterns like 7-day activity, CI validation

4. Dagger - Solomon Hykes (Founder of Docker) - Typed Composition with MCP
    * Problem with LLMs if you want to interact with its environment, you often have a container which is a good container but it can containimate your workspace if you let it run in the background. 
    * Raw tool calling straight to endpoint or over MCP which is nice decoupling with an ecosystem of tools? The tools are in a name space that are flat. The problem is that you when you have shell commands which return objects instead of primitive types, you quickly have a contaninamted environment
    * [Goose tool](https://github.com/block/goose): a local, extensible, open source AI agent that automates engineering tasks
    
5. Amazon Web Services Generative AI - Nicholas Aldridge (MCP Steering Committee Member) - Multi-Agent Collaboration in MCP
    *  Nested MCPs: In tool calling you can call another MCP
    * [Spring AI Tool](https://github.com/spring-projects/spring-ai): apply to the AI domain Spring ecosystem design principles such as portability and modular design and promote using POJOs as the building blocks of an application to the AI domain
    * [Open Protocols for Agent Interoperability Part 1: Inter-Agent Communication on MCP](https://aws.amazon.com/blogs/opensource/open-protocols-for-agent-interoperability-part-1-inter-agent-communication-on-mcp/)

6. Cisco - Arjun Sambamoorthy - Securing MCP in an Agentic World
    * Tool Poisoning: Client calls a malicious MCP server which invjects malicious prompt into the LLM
    * Parameter Abuse: use parameters to inject malicious code into the LLM
    * Wrong Tool Selection: two tools which similar descriptions
    * Indirect Poisoning: Webpage given a malicious instruction to the MCP server which goes back to the client

7. LLama Index - Laurie Voss - MCP vs. ACP vs. A2A: Comparing Agent Protocols
    * [A Survey of AI Agent Protocols](https://arxiv.org/pdf/2504.16736)
    * [Agent Network Protocol](https://agent-network-protocol.com/)
    * AITP: Agent Interaction & Transaction Protocol
    * [ACP: Agent Connect Protocol](https://github.com/agent-network-protocol/AgentConnect)
    * [ACP IBM](https://workos.com/blog/ibm-agent-communication-protocol-acp)
    * [Eclipse LMOS](https://eclipse.dev/lmos/)
    * MCP lacks asynchorous, renegotiation (upgrade the protcol like in Agora), but it's done one thing well: it started with a small problem and got traction for one small problem and can address all other problem later

8. Andressen Horrowitz - Yoko Li - [What MCP Middleware Could Look Like](https://gamma.app/docs/MCP-Middleware-pbuir25swchroty?mode=present#card-2jnv25t7fjgqr73)
    * User preference space tool use: please do not execute shell commands
    * Need a middleware that will route, cache, search tools 
    * Machine-initiated workloads picking up

9. AWS - Antje Barth - Scaling MCP
* [AWS MCP Servers](https://github.com/awslabs/mcp)


### [MIT Nanda Project] (https://nanda.media.mit.edu/)


## Data Infrastructure and Distributed Systems
1. [Data Mesh](https://www.nextdata.com/)
    * Data is the runtime substrate for AI agents: Autonomous agents depend entirely on the integrity, semantics, and freshness of the data they consume — not just the model weights — making data governance and discoverability critical infrastructure, not optional add-ons.
    * Traditional data platforms are agent-hostile: Legacy architectures lack real-time metadata, semantic schemas, and access guarantees — causing agents to hallucinate, expose PII, or act on stale or irrelevant inputs, severely limiting their reliability in production.
    * Autonomous data products encapsulate trust and runtime safety: Nextdata OS standardizes access via APIs enriched with metadata, freshness guarantees, and semantic structure — enabling agents to reason and act autonomously while minimizing risk.
    * Scalable agent ecosystems require cross-product discovery: Agents must continuously discover and consume new data products. Nextdata OS enables this via discovery APIs, allowing agents to adapt to data evolution without human reconfiguration — a foundational capability for AI-native enterprises.
    * Interoperability and governance are non-negotiable: By aligning with standards like the MCP protocol and embedding policy enforcement at the product level, Nextdata OS supports safe, permissioned agent access — especially vital in regulated industries like finance and healthcare.
2. [CME 323: Distributed Algorithms and Optimization](https://web.stanford.edu/~rezab/classes/cme323/S16/)

## Inference Acceleration
1. [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180)
2. [FLEX ATTENTION: A PROGRAMMING MODEL FOR GENERATING OPTIMIZED ATTENTION KERNELS](https://arxiv.org/pdf/2412.05496)
3. [VLLM Attention Kernels](https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cuh)
4. [Attention Gym Paged Attention Implementation](https://github.com/pytorch-labs/attention-gym/blob/main/attn_gym/paged_attention/paged_attention.py)
5. [CONTEXT PARALLELISM FOR SCALABLE MILLION-TOKEN INFERENCE](https://arxiv.org/pdf/2411.01783)
6. [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/pdf/2002.03794)
7. [Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer)
8. [vLLM Office Hours #26 Intro to torch.compile and how it works with vLLM](https://www.youtube.com/watch?v=1aEFHpF69Lc)

## Distributed AI Inference
1. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
2. [Dynamo AI Tutorials](https://www.linkedin.com/posts/vikramsharmam_distributed-inference-101-getting-started-activity-7308019486517858305-JX7e?utm_source=share&utm_medium=member_ios&rcm=ACoAAB-7BosBbrFaamvv690_M7ruCd3EHmcHhg0)

## Synthetic Data Generation
1. [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/pdf/2504.04736)
2. [Physics of Language Models](https://physics.allen-zhu.com/part-4-architecture-design/part-4-1)

## Continual Learning
1. [Lifelong Learning of Large Language Model based Agents: A Roadmap](https://arxiv.org/pdf/2501.07278)
2. [TRACE: A COMPREHENSIVE BENCHMARK FOR CONTINUAL LEARNING IN LARGE LANGUAGE MODELS](https://openreview.net/pdf?id=xelrLobW0n)
3. [A Comprehensive Survey of Continual Learning: Theory, Method and Application](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10444954)

## PhD Defenses
1. [On Evaluation and Efficient Post-training for LLMs](https://docs.google.com/presentation/d/1-4qwacAMJ012Pv5W5xvTAhfqmF7M9CqN/edit?slide=id.p1#slide=id.p1)

## LLM Alignment
1. [BPO: Staying Close to the Behavior LLM Creates Better Online LLM Alignment.](https://arxiv.org/pdf/2406.12168)
2. [FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users](https://arxiv.org/pdf/2502.19312?)

## AI Reasoning and Agents
1. [Measuring AI Ability to Complete Long Tasks](https://arxiv.org/pdf/2503.14499) - [Code](https://github.com/METR/eval-analysis-public)
2. [Evolving Deeper LLM Thinking](https://arxiv.org/pdf/2501.09891)
3. [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
4. [OpenAI Evals](https://github.com/openai/evals)
5. [Deliberative Alignment: Reasoning Enables Safer Language Models](https://arxiv.org/pdf/2412.16339)

## Agent Browser Tools
1. [Stagehand Docs](https://docs.stagehand.dev/get_started/introduction)

## Agents
1. [ADVANCES AND CHALLENGES IN FOUNDATION AGENTS](https://arxiv.org/pdf/2504.01990)
2. [PhD Dissertation: Language Agents From Next-Token Prediction to Digital Automation](https://ysymyth.github.io/papers/Dissertation-finalized.pdf)
3. [Why Do Multi-Agent LLM Systems Fail?](https://export-test.arxiv.org/pdf/2503.13657)
4. [REST MEETS REACT: SELF-IMPROVEMENT FOR MULTI-STEP REASONING LLM AGENT](https://arxiv.org/pdf/2312.10003)
5. [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/pdf/2408.07199)
6. [Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions](https://arxiv.org/pdf/2505.00675)

## Vector Databases
1. [Primers -- Approximate Nearest Neighbors -- Similarity Search](https://vinija.ai/concepts/ann-similarity-search/)

## Coding Agents
1. [SWE-BENCH MULTIMODAL: DO AI SYSTEMS GENERALIZE TO VISUAL SOFTWARE DOMAINS?](https://arxiv.org/pdf/2410.03859)
2. [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)
3. [Code Generation with Small Language Models: A Deep Evaluation on Codeforces](https://arxiv.org/pdf/2504.07343)
4. [IDA-Bench: Evaluating LLMs on Interactive Guided Data Analysis](https://arxiv.org/pdf/2505.18223)

## Multimodal
1. [WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models](https://arxiv.org/pdf/2401.13919)
2. [Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V](https://arxiv.org/pdf/2310.11441)
3. [Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents](https://arxiv.org/pdf/2502.11357)

## Compound AI Systems and RAG
1. [HETEROGENEOUS SWARMS: Jointly Optimizing Model Roles and Weights for Multi-LLM Systems](https://arxiv.org/pdf/2502.04510)
2. [LLM-based Optimization of Compound AI Systems: A Survey](https://arxiv.org/pdf/2410.16392)
3. [Microsoft Trace - End-to-end Generative Optimization for AI Agents - DSPy Alternative](https://github.com/microsoft/Trace)
4. [Semantic Search with Weaviate's Query Agent, LlamaIndex and Comet](https://colab.research.google.com/drive/1Ef-VhKIJqj85K2fKa_sUR0N8WGJgik4x?usp=sharing&referrer=luma)


## Model Training
1. [Stanford CS336: Language Modeling from Scratch | Spring 2025](https://www.youtube.com/watch?v=ptFiH_bHnJw)
2. [LLM Training with DeepSpeed](https://github.com/microsoft/DeepSpeed)
3. [Gemini Flash Pretraining](https://vladfeinberg.com/2025/04/24/gemini-flash-pretraining.html)
4. [Deepmind Efficient Training of Large Language Models](https://www.youtube.com/watch?v=1MqlbPsWnAA)
5. [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448)
6. [Scaling Data-Constrained Language Models](https://vladfeinberg.com/assets/2025-04-24-princeton-talk.pdf)
7. [TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://arxiv.org/html/2410.06511v2)

## Reinforcement Learning
1. [SkyRL](https://github.com/NovaSky-AI/SkyRL)
2. [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)
3. [TinyZero](https://github.com/leaplab-thu/TinyZero)
4. [ZeroReason](https://github.com/Jiayi-Pan/TinyZero)
5. [rLLM](https://github.com/agentica-project/rllm)
6. [Spurious Rewards: Rethinking Training Signals in RLVR](https://rethink-rlvr.notion.site/Spurious-Rewards-Rethinking-Training-Signals-in-RLVR-1f4df34dac1880948858f95aeb88872f)

## AI Chip Manufacturing

1. [TCP: A Tensor Contraction Processor for AI Workloads](https://dli5ezlttyahz.cloudfront.net/FuriosaAI-tensor-contraction-processor-isca24.pdf?p=download/FuriosaAI-tensor-contraction-processor-isca24)
2. [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://arxiv.org/pdf/2505.09343)
3. [There’s plenty of room at the Top: What will drive computer performance after Moore’s law?](https://www.science.org/doi/10.1126/science.aam9744)

## AI Safety
1. [Reasoning Models Don’t Always Say What They Think](https://assets.anthropic.com/m/71876fabef0f0ed4/original/reasoning_models_paper.pdf)

## Old School Machine Learning
1. [Elements of Information Theory](https://cs-114.org/wp-content/uploads/2015/01/Elements_of_Information_Theory_Elements.pdf)
2. [Pattern Classification](http://cscog.likufanele.com/~calvo/Inteligencia_Artificial_files/Duda%20R%20O,%20Hart%20P%20E,%20Stork%20D%20G_Pattern%20Classification%20%282Ed%20Wiley%29.pdf)
3. [Approximation Algorithms](https://www.amazon.com/Approximation-Algorithms-Vijay-V-Vazirani/dp/3642084699)
4. [The Probabilistic Method](https://math.bme.hu/~gabor/oktatas/SztoM/AlonSpencer.ProbMethod3ed.pdf)
5. [Prediction, Learning, and Games](https://www.cambridge.org/core/books/prediction-learning-and-games/A05C9F6ABC752FAB8954C885D0065C8F)
6. [Neural Network Learning: Theoretical Foundations](https://www.cambridge.org/core/books/neural-network-learning/665C8C7EB5E2ABC5367A55ADB04E2866)
7. [Convex Optimization – Boyd and Vandenberghe](https://stanford.edu/~boyd/cvxbook/)
8. [Learning with Kernels](https://mcube.lab.nycu.edu.tw/~cfung/docs/books/scholkopf2002learning_with_kernels.pdf)
    8.1 Provides great intuition for why kernels are useful.

## Statistical Inference
1. [Mathematical Statistics](https://link.springer.com/book/10.1007/b97553)

## Philosophy of AI
1. [AI 2027](https://ai-2027.com/)

## Graph Learning
1. [Google at NeurIPs: Mining and Learning with Graphs at Scale](https://neurips.cc/Expo/Conferences/2020/workshop/20237)

## Compute Orchestration
1. [Cilantro: Performance-Aware Resource Allocation for Large LanguageGeneral Objectives via Online Feedback](https://www.usenix.org/system/files/osdi23-bhardwaj.pdf)

## Mathematic Foundations
1. Numerical Optimization by Jorge Nocedal Stephen Wright
2. Elements of Information Theory

