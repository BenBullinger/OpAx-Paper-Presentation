# Optimistic Active Exploration of Dynamical Systems

* Link: [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/77b5aaf2826c95c98e5eb4ab830073de-Abstract-Conference.html)

[TOC]


TODO:
* Intuition of the proof, in words
* Motivate the problem using experiments
* UCBVI
* Find some examples where OPAX fails

## Tentative Presentation Structure
1. Motivations (2-3 pages) ==Ben==
    - [ ] Benefit of model-based approach
	- [ ] Single-reward model-based RL does not perform well on other tasks
	**Transit smoothly to the problem description**
2. Problem ==Chung-En==
	- [x] Full description of the problem (1 page)
	- [x] Assumptions (1 page)
	- [ ] Challenges (1 page)
**Transit smoothly to how previous works handled these challenges**
3. Related work ==Chung-En==
	- [x] Active learning (1 page)
	- [x] Maximum entropy exploration (1 page)
	- [x] Reward-free RL (1 page)
**Transit smoothly to the contributions of this paper**
4. Contributions (1 page) ==Chung-En==
5. Algorithms ==Ben==
	- [ ] Maximize mutual information (1-2 pages)
        ==This part should already be explained by the frst presentation. But I think it's better for us to go through (recall) it briefly.==
	- [ ] Optimism in the face of uncertainty (1-2 pages)
	   ==I think we need to focus on this part. Applying OFU to dynamics instead of reward functions is a new technique to students (although this was taught in PAI).==
	- [ ] Full description of the algorithm (1 page)
6. Theory ==Chung-En==
	- [x] Recall our assumptions (1 page)
	- [x] Main GP guarantee (1 page)
	- [X] Benefit of optimism + its very short proof (1 page)
	- [x] Zero-shot performance (1 page)
7. Experiments (TBD) ==Ben==
    - [x] Results from paper
    - [ ] Own video (pendulum swing_up generalization when trained on keep_down)
9. Conclusions, limitations, and future directions (1-2 pages) ==Ben==

## Motivations

1. Although model-based RL methods can obtain a model of the underlying dynamical system, the learned model is more accurate on the state-action region that is relevant to the task. It does not generalize well to other reward functions.
2. Learning dynamical system (system identification) is useful. Can do planning in a zero-shot manner. Stability and sensitivity can also be inferred.
## Required background

1. Active learning: Learn an unknown function. We can query the function at any point. Want to minimize the number of queries. Use "optimism in the face of uncertainty" to guide exploration.
2. Optimal control

## Connections to the lecture

1. Model-based RL
2. Active learning in PAI


## Challenges

How to build a model?

* Building a model directly from physics is difficult. We need a data-driven approach.

How to interact with the environment to collect data?

* Simulating the system or querying the system at arbitrary state-action pairs are unrealistic. This is different from active learning.
* Learning only from offline data can be challenging (?)
* We can execute carefully designed policies and observe their trajectories.

How to design policies to explore the dynamics?

* Intuitively, designing good policies requires long-term planning. This is harder than active learning. (Supervised learning vs. RL)

## (Theoretical) Contributions

1. Propose an algorithm OPAX for learning non-linear dynamics with continuous state-action spaces. In each episode, OPAX quantifies epistemic uncertainty in the state-action spaces and use optimism to design an explorative policy.
2. The first to give convergence guarantees for a rich family of non-linear dynamical systems.

## Algorithmic techniques

1. Optimism in the face of uncertainty
2. Maximizing mutual information

## Theoretical results

### Assumptions

1. (Calibration) From the collected data, we can estimate the mean and variance of the dynamic with high confidence.
2. (Noise) Noise is i.i.d. zero-mean Gaussian with known variance.

We won't get into the details of building a calibrated model from data. The calibration assumption can be satisfied by using Gaussian processes.

### Main Guarantees

The maximum expected information gain at iteration $N$ of any policy is small, i.e., we cannot obtain much more information from the data. With probability at least $1-\delta$ (over the noise?),
$$\mathbb{E}\left[ \max_{\pi}\mathbb{E}_\tau[ I(f^\star;y_\tau | D_{1:N-1}) ]  \right] \lesssim \beta_N(\delta) T^{3/2}\sqrt{\frac{\mathcal{C}_N(f^\star)}{N}}.$$
In the special case of GP with kernel $k$, we have
$$\max_{\pi}\mathbb{E}_\tau\left[ \max_{z\in\tau}\lVert \sigma_N (z) \rVert^2_2  \right] \lesssim \beta_N(\delta) T^{3/2}\sqrt{\frac{\gamma_N(k)}{N}}.$$
For example, when using linear or RBF kernels, we have $\gamma_N(k) \lesssim \text{polylog}(N)$.

:::warning
When considering sub-Gaussian noise, the dependence on $\beta_N$ becomes $\beta_N^T$. According to the paper, this seems to be unavoidable (?).
:::


### Zero-shot performance on downstream tasks

Consider the following planning (optimal control) problem
$$\max_{\pi} J_c^{f^\star}(\pi)$$
where
$$J_c^{f}(\pi)=\mathbb{E}_\tau\left[ \sum_{t=0}^{T-1} c(x_t,\pi(x_t)) \right],\quad x_{t+1} = f(x_t,\pi(x_t)) + \omega_t.$$
We have for all $\pi$,
$$J_c^{f^\star}(\pi) - \min_{\hat{f}}J_c^{\hat{f}}(\pi)
\lesssim T\cdot\frac{\sqrt{d}\beta_{N-1}(\delta)}{\sigma^2}\cdot\mathbb{E}_{\tau^{\pi_N}}\left[  \sum_{t=0}^{T-1} \lVert \sigma_{N-1}(z_t) \rVert_2 \right],$$
where the minimum is over all $\hat{f}$ of the form $\hat{f}=\mu_N + \beta_N\sigma_N\eta$ for some $\eta:\mathcal{X}\to[-1,1]^d$. In other words $\hat{f}\in\{ \mu_N+\beta_N\sigma_N\eta: \eta:\mathcal{X}\to[-1,1]^d\}$.

:::info
The following result is presented in Lemma 13 in Appendix, and it answers one of my questions. However, the original proof uses proof by contradiction, which in my opinion is not easy to understand. We provide a simpler and more direct proof below.
:::

What does this guarantee mean? Suppose we use GP with kernel $k$ as the model. By the main guarantee, we have $$J_c^{f^\star}(\pi) - \min_{\hat{f}}J_c^{\hat{f}}(\pi) \lesssim T\cdot\frac{\sqrt{d}\beta_{N-1}(\delta)}{\sigma^2} \cdot T \cdot \sqrt{ \beta_N(\delta) T^{3/2}\sqrt{\frac{\gamma_N(k)}{N}} } = \frac{ \sqrt{d} \beta_N(\delta)^{3/2}T^{11/4}\gamma_N(k)^{1/4} }{ \sigma^2 N^{1/4} }.$$
If $$N\gtrsim \frac{ d^2\beta_N(\delta)^6 T^{11} \gamma_N(k) }{\sigma^8 \varepsilon^4},$$then we have $0\leq J_c^{f^\star}(\pi) - \min_{\hat{f}}J_c^{\hat{f}}(\pi) \leq \varepsilon$. This means that we can approximate the true expected cumulated return by the worst expected cumulative return over all possible dynamics: $$\min_{\hat{f}}J_c^{\hat{f}}(\pi) \leq J_c^{f^\star}(\pi) \leq \min_{\hat{f}}J_c^{\hat{f}}(\pi) + \varepsilon.$$By taking maximum over all policies, we get $$\max_{\pi}\min_{\hat{f}}J_c^{\hat{f}}(\pi) \leq J_c^{f^\star}(\pi^\star) \leq \max_{\pi}\min_{\hat{f}}J_c^{\hat{f}}(\pi) + \varepsilon.$$
where $\pi^\star$ is the optimal policy for the true dynamic $\pi^\star \in \arg\!\max_\pi J_c^{f^\star}(\pi)$. Let $\hat{\pi}$ be a $\delta$-maximin policy, i.e., $$\min_{\hat{f}}J_c^{\hat{f}}(\hat{\pi}) \geq \max_{\pi} \min_{\hat{f}} J_c^{\hat{f}}(\pi) - \delta.$$Then, we have $$J_c^{f^\star}(\pi^\star) \leq \min_{\hat{f}} J_c^{\hat{f}}(\hat{\pi}) + \varepsilon + \delta.$$Note that by the calibration assumption, we have $f^\star = \mu_N + \beta_N\sigma_N\eta^\star$ for some $\eta^\star:\mathcal{X}\to[-1,1]^d$ with high probability. Hence, we have $$J_c^{f^\star}(\pi^\star) - J_c^{f^\star}(\hat{\pi}) \leq \varepsilon + \delta.$$

:::info
The remaining question is: How to compute a $\delta$-maximin policy $\hat{\pi}$? I think this is not discussed in the paper.
:::


The problem looks like computing the Nash equilibrium of a two-agent zero-sum RL problem.

## Follow-up work


## Questions

1. Once we have learned a model, we can use it for planning. How good is the performance of downstream tasks with different reward functions? Does the performance change drastically with different reward functions? How to translate upper bound on the epistemic uncertainty to performance bound on the downstream tasks? (This is answered in the paper.)
2. What about continuous-time models? Can we reduce it to discrete-time models? (Think more carefully about this.) https://arxiv.org/abs/2102.04764
3. Can we consider adversarial noise?
4. Can we generalize the result to infinite horizon?
5. Construct a simple example such that model-based RL does not generalize to other tasks. (Think more carefully about this.)
6. ==What's the connection to non-linear RL that maximizes entropy? May also related to reward-free RL. What is the benefit of OPAX over these methods?== https://arxiv.org/abs/2103.06257
7. Regarding zero-shot performance, how to solve the maximin policy given the reward $r$? How to solve it without knowing $r$? Can we solve it efficiently?
8. OpAX requires the knowledge of the noise level $\sigma$. If we don't know it, what can we do?



## Related work
1. Single-reward model-based RL
2. System identification
3. GP and Active learning:
	* ==(Present before us)== Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design https://arxiv.org/abs/0912.3995: This will introduce active learning, mutual information, the information gain $\gamma$.
4. Intrinsic reward method: e.g., Maximum entropy exploration
	* ==(Present before us)== Provably Efficient Maximum Entropy Exploration https://proceedings.mlr.press/v97/hazan19a.html: proposed maximizing entropy for exploration. did not provide guarantees on downstream planning tasks (also pointed out by reward-free paper by Chi Jin) Model-free/based?
	* Maximum Entropy RL (Provably) Solves Some Robust RL Problems https://arxiv.org/abs/2103.06257: "one should apply MaxEnt RL to a pessimistic version of that reward function" "MaxEnt RL is not robust with respect to the reward function it is trained on." The objective in the zero-shot performance of OPAX is a special case of the robust RL objective. Theorem 4.1 is related. The limitations of Theorem 4.1 are that we don't know $\varepsilon>0$ and that the set $\mathcal{R}$ does not contain all reward functions. (If we let $\alpha\to\infty$, we can remove the use of reward functions. What is the guarantee?) Model-free/based?
5. Reward-free RL
	* Reward-Free Exploration for Reinforcement Learning https://proceedings.mlr.press/v119/jin20d.html: Model-based planning (require knowledge of reward functions, same as the zero-shot performance below). Discrete setting. Sample complexity $\lesssim \frac{S^2A\text{poly}(H)}{\varepsilon^2}$. Does not scale to large state-action spaces.
	* On Reward-Free Reinforcement Learning with Linear Function Approximation https://proceedings.neurips.cc/paper/2020/hash/ce4449660c6523b377b22a1dc2da5556-Abstract.html: Study reward-free RL in linear MDP and more general MDP. Show a hardness result: under the linear $Q^\star$ assumption and deterministic dynamics, reward-free RL requires exponential sample complexity. In other words, this may not be suitable for continuous dynamics.
	* On Reward-Free RL with Kernel and Neural Function Approximations: Single-Agent MDP and Markov Game https://proceedings.mlr.press/v139/qiu21d.html: as title suggests. Two-layer overparameterized NN.
	* Near Optimal Reward-Free Reinforcement Learning https://proceedings.mlr.press/v139/zhang21e.html: optimal upper bounds (up to logarithmic factors).
	*  On the Statistical Efficiency of Reward-Free Exploration in Non-Linear RL https://proceedings.neurips.cc/paper_files/paper/2022/hash/8433bb4f7477bf8202614ce1ae8b1169-Abstract-Conference.html: seems like a unify paper. Show exponential hardness under some linear structural assumptions, e.g., linear MDP.
	* Task-agnostic Exploration in Reinforcement Learning https://proceedings.neurips.cc/paper/2020/hash/8763d72bba4a7ade23f9ae1f09f4efc7-Abstract.html: a setting related to reward-free RL. Does not need to know the true reward function, only need to know samples from it. But can only apply to $N$ different reward functions.
6. Continuous time
	* Optimal Exploration for Model-Based RL in Nonlinear Systems https://proceedings.neurips.cc/paper_files/paper/2023/hash/31e018f43ab9c7065c058cc2c5848128-Abstract-Conference.html: model-based RL in non-linear dynamics. Require reward functions.
	* Maximum Entropy Optimal Control of Continuous-Time Dynamical Systems https://ieeexplore.ieee.org/abstract/document/9760123: Study optimal control with continuous dynamics. The objective is entropic regularized cost. Both the dynamic and cost are known.