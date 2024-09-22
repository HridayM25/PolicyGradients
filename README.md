## Vanilla Policy Gradient Method 

1. **Policy Gradient**: The gradient of the expected return with respect to the policy parameters:

\[ \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi}(s_t, a_t) \right] \]

Where:
- $\( J(\theta) \)$ is the objective function (e.g., expected return).
- $\( \pi_{\theta}(a_t|s_t) \)$ is the policy function parameterized by \( \theta \), representing the probability of taking action \( a_t \) in state \( s_t \).
- $\( A^{\pi}(s_t, a_t) \)$ is the advantage function, representing how much better or worse the action \( a_t \) is compared to the expected value under policy \( \pi \).
- $\( \tau \)$ represents a trajectory of state-action pairs.

## Actor-Critic Method 

1. **Policy Gradient** (Actor):
$\( \nabla_{\theta^{\pi}} J(\theta^{\pi})$= \mathbb{E}_{\tau} \left[ \sum_{t=0}^{T} \nabla_{\theta^{\pi}} \log \pi_{\theta^{\pi}}(a_t|s_t) A^{\pi}(s_t, a_t) \right] \]

2. **Value Function Update** (Critic):
\[ \delta_t = r_t + \gamma V^{\pi}(s_{t+1}) - V^{\pi}(s_t) \]
\[ V^{\pi}(s_t) = V^{\pi}(s_t) + \alpha_v \delta_t \]

Where:
- $\( \nabla_{\theta^{\pi}} J(\theta^{\pi}) \)$ is the gradient of the expected return with respect to the policy parameters \( \theta^{\pi} \).
- $\( \pi_{\theta^{\pi}}(a_t|s_t) \)$ is the policy function parameterized by \( \theta^{\pi} \), representing the probability of taking action \( a_t \) in state \( s_t \).
- $\( A^{\pi}(s_t, a_t) \)$ is the advantage function, representing how much better or worse the action \( a_t \) is compared to the expected value under policy \( \pi \).
- $\( \delta_t \)$ is the temporal difference error, representing the difference between the predicted value and the actual return.
- $\( r_t \)$ is the reward received after taking action \( a_t \) in state \( s_t \).
- $\( \gamma \)$ is the discount factor.
- $\( V^{\pi}(s_t) \)$ is the estimated value function for state \( s_t \).
- $\( \alpha_v \)$ is the learning rate for the value function.
