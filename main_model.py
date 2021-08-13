import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm

def prob_to_logodds(p):
    # p is the probability of being red
    if p == 1:
        p = 0.9999
    elif p == 0:
        p = 0.0001
    return np.log(p / (1 - p))

def logodds_to_prob(logodds):
    return np.exp(logodds) / (1 + np.exp(logodds))

def L_o(confidence_agents, choice_agents):
    l_o = 0
    for agent_num in range(4):
        l_o += confidence_agents[agent_num] * prob_to_logodds(choice_agents[agent_num])
    return l_o

def L_s(num_red_beads):
    return prob_to_logodds(num_red_beads / 8)

def F_func(L, w):
    return np.log((w * np.exp(L) + 1 - w) / ((1 - w) * np.exp(L) + w))

def simple_bayes(confidence_agents, choice_agents, num_red_beads):

    prior = L_o(confidence_agents, choice_agents)
    likelihood = L_s(num_red_beads)

    # # Specifying biases
    # mean_bias_sz = norm.rvs(0, 2)
    # mean_bias_ctrl = norm.rvs(0, 1)
    # sigma_bias = np.abs(norm.rvs(0, 1))
    # bias = norm.rvs(mean_bias_sz, sigma_bias)

    # final logodds of the probability of a red bean next
    logodds_r = likelihood + prior

    # return the probability of a red bead
    return logodds_to_prob(logodds_r)

def weighted_bayes(confidence_agents, choice_agents, num_red_beads, w_s, w_o):

    prior = F_func(L_o(confidence_agents, choice_agents), w_o)
    likelihood = F_func(L_s(num_red_beads), w_s)

    logodds_r = likelihood + prior

    return logodds_to_prob(logodds_r)

def circular_inference(confidence_agents, choice_agents, num_red_beads, w_s, w_o, a_s, a_o):

    l_o = L_o(confidence_agents, choice_agents)
    l_s = L_s(num_red_beads)

    F_o = F_func(a_o * l_o, w_o)
    F_s = F_func(a_s * l_s, w_s)

    prior = F_func(l_o + F_s + F_o, w_o)
    likelihood = F_func(l_s + F_s + F_o, w_s)

    logodds_r = likelihood + prior

    return logodds_to_prob(logodds_r)

confidence_agents = [0.5, 0.5, 0.5, 0.5]
choice_agents = [0.9, 0.9, 0.9, 0.9]
num_red_beads = 1
w_s = 0.7
w_o = 0.9
a_s = 0.5
a_o = 0.3
print("Prediction from Simple Bayes: ", simple_bayes(confidence_agents, choice_agents, num_red_beads))
print("Prediction from Weighted Bayes: ", weighted_bayes(confidence_agents, choice_agents, num_red_beads, w_s, w_o))
print("Prediction from Circular Inference: ", circular_inference(confidence_agents, choice_agents, num_red_beads, w_s, w_o, a_s, a_o))

p_simple_bayes = simple_bayes(confidence_agents, choice_agents, num_red_beads)
p_weighted_bayes = weighted_bayes(confidence_agents, choice_agents, num_red_beads, w_s, w_o)
p_circular_inference = circular_inference(confidence_agents, choice_agents, num_red_beads, w_s, w_o, a_s, a_o)

r_simple_bayes = bernoulli.rvs(p_simple_bayes, size=10)
r_weighted_bayes = bernoulli.rvs(p_weighted_bayes, size=10)
r_circular_inference = bernoulli.rvs(p_circular_inference, size=10)

print(r_simple_bayes)
print(r_weighted_bayes)
print(r_circular_inference)

# Let us first explore how parameter settings effect performance. We'll run through 