import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm
import csv

class Models:
    def prob_to_logodds(self, p):
        # p is the probability of being red
        if p == 1:
            p = 0.9999
        elif p == 0:
            p = 0.0001
        return np.log(p / (1 - p))

    def logodds_to_prob(self, logodds):
        return np.exp(logodds) / (1 + np.exp(logodds))

    def L_o(self, confidence_agents, choice_agents):
        l_o = 0
        for agent_num in range(4):
            l_o += confidence_agents[agent_num] * self.prob_to_logodds(choice_agents[agent_num])
        return l_o

    def L_s(self, num_red_beads):
        return self.prob_to_logodds(num_red_beads / 8)

    def F_func(self, L, w):
        return np.log((w * np.exp(L) + 1 - w) / ((1 - w) * np.exp(L) + w))

    def simple_bayes(self, confidence_agents, choice_agents, num_red_beads):
        # Returns a posterior probability for choosing red bead

        prior = self.L_o(confidence_agents, choice_agents)
        likelihood = self.L_s(num_red_beads)

        # # Specifying biases
        # mean_bias_sz = norm.rvs(0, 2)
        # mean_bias_ctrl = norm.rvs(0, 1)
        # sigma_bias = np.abs(norm.rvs(0, 1))
        # bias = norm.rvs(mean_bias_sz, sigma_bias)

        # final logodds of the probability of a red bean next
        logodds_r = likelihood + prior

        # return the probability of a red bead
        return self.logodds_to_prob(logodds_r)

    def weighted_bayes(self, confidence_agents, choice_agents, num_red_beads, w_s, w_o):
        # Returns a posterior probability for choosing red bead

        prior = self.F_func(self.L_o(confidence_agents, choice_agents), w_o)
        likelihood = self.F_func(self.L_s(num_red_beads), w_s)

        logodds_r = likelihood + prior

        return self.logodds_to_prob(logodds_r)

    def circular_inference(self, confidence_agents, choice_agents, num_red_beads, w_s, w_o, a_s, a_o):
        # Returns a posterior probability for choosing red bead

        l_o = self.L_o(confidence_agents, choice_agents)
        l_s = self.L_s(num_red_beads)

        F_o = self.F_func(a_o * l_o, w_o)
        F_s = self.F_func(a_s * l_s, w_s)

        prior = self.F_func(l_o + F_s + F_o, w_o)
        likelihood = self.F_func(l_s + F_s + F_o, w_s)

        logodds_r = likelihood + prior

        return self.logodds_to_prob(logodds_r)

if __name__ == "__main__":
    print("Running script")
