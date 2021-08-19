from main_model import Models
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
from scipy.stats import bernoulli

agent_choices_all = np.zeros((35, 4))
with open('agent_choices.csv', newline='') as csvfile:
    choices = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(choices):
        agent_choices_all[i] = [float(j) for j in row]

agent_confidences_all = np.zeros((35, 4))
with open('agent_confidences.csv', newline='') as csvfile:
    confidences = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(confidences):
        agent_confidences_all[i] = [float(j) for j in row]

participant_red_beads_all = np.zeros(35)
with open('particpant_red_beads.csv', newline='') as csvfile:
    red_beads = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(red_beads):
        participant_red_beads_all[i] = int(row[0])

models = Models()
posteriors_simple_bayes = []
posteriors_weighted_bayes = []
posteriors_circular_inference = []
sim_choices_simple_bayes = []
sim_choices_weighted_bayes = []
sim_choices_circular_inference = []
for trial_block in range(3):
    trial_indices = np.arange(35)
    np.random.shuffle(trial_indices)
    for trial in trial_indices:
        # Set the parameters to recover
        w_s = 0.6
        w_o = 0.9
        a_s = 0.2
        a_o = 0.2

        confidence_agents = agent_confidences_all[trial]
        choice_agents = agent_choices_all[trial]
        num_red_beads = participant_red_beads_all[trial]

        p_simple_bayes = models.simple_bayes(confidence_agents, choice_agents, num_red_beads)
        p_weighted_bayes = models.weighted_bayes(confidence_agents, choice_agents, num_red_beads, w_s, w_o)
        p_circular_inference = models.circular_inference(confidence_agents, choice_agents, num_red_beads, w_s, w_o, a_s, a_o)

        r_simple_bayes = bernoulli.rvs(p_simple_bayes, size=1)
        r_weighted_bayes = bernoulli.rvs(p_weighted_bayes, size=1)
        r_circular_inference = bernoulli.rvs(p_circular_inference, size=1)

        posteriors_simple_bayes.append(p_simple_bayes)
        posteriors_weighted_bayes.append(p_weighted_bayes)
        posteriors_circular_inference.append(p_circular_inference)

        sim_choices_simple_bayes.append(r_simple_bayes[0])
        sim_choices_weighted_bayes.append(r_weighted_bayes[0])
        sim_choices_circular_inference.append(r_circular_inference[0])

sim_choices_df = pd.DataFrame({
    'simple_bayes': sim_choices_simple_bayes,
    'weighted_bayes': sim_choices_weighted_bayes,
    'circular_inference': sim_choices_circular_inference
})
sim_choices_df.to_csv('sim_choices.csv')

# plt.hist(posteriors_simple_bayes)
# plt.hist(posteriors_weighted_bayes)
# plt.hist(posteriors_CI_bayes)
# plt.show()