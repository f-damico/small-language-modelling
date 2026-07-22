import torch
import random
from itertools import *


class BpRhm:
    def __init__(self, v, s, m, L, rules, device="cpu") -> None:
        """
        Initialize the random hierarchy model.
        v: vocabulary size. This code assumes that the number of classes is the same as the vocabulary size.
        s: size of the lower-level representations
        m: number of synonymic lower-level representations
        L: number of levels of the hierarchy
        rules: dictionary with the rules for each level of the hierarchy
        """
        self.v = v
        self.n = v
        self.s = s
        self.m = m
        self.L = L
        self.rules = rules
        self.device = device

        for key in self.rules.keys():
            self.rules[key] = self.rules[key].to(self.device)

    def initalize_messages(self, nu_up_L=None, nu_down_0=None):
        """
        Initialize the messages to uniform over the vocabulary
        """
        self.nu_up = {}
        self.nu_down = {}
        if nu_up_L is None:
            d = self.s**self.L
            self.nu_up[self.L] = (
                torch.ones(tuple([self.v] + [d]), device=self.device) / self.v
            )
        else:
            self.nu_up[self.L] = nu_up_L

        if nu_down_0 is None:
            self.nu_down[0] = (
                torch.ones(tuple([self.v] + [1]), device=self.device) / self.v
            )
        else:
            self.nu_down[0] = nu_down_0

        return self.nu_up, self.nu_down

    def set_evidence_to_leaf_messages(self, x_leaves, noise=0.0):
        """
        Set the evidence to the leaf messages, with possible uniform noise
        x_leaves: list with the data at the leaf level
        """
        x_leaves = x_leaves.flatten()
        nu = torch.zeros((self.v, len(x_leaves)), device=self.device)
        nu[x_leaves, torch.arange(len(x_leaves))] = 1.0 - noise
        nu = nu + noise / self.v
        return nu

    def set_masking_to_leaf_messages(self, x_leaves, masked):
        """
        Set the evidence to the leaf messages, with possible uniform noise
        x_leaves: list with the data at the leaf level
        """
        x_leaves = x_leaves.flatten()
        nu = torch.zeros((self.v, len(x_leaves)), device=self.device)
        nu[x_leaves, torch.arange(len(x_leaves))] = 1.0
        nu[:, masked] = 1.0 / self.v
        return nu

    def upward_rule_to_proba(self, l):
        """
        Compute the probability of the rules given the upward messages.
        l: level of the hierarchy
        return: probabilities of the rules. shape = (v, m, I)
        """

        R = self.rules[l]
        V = self.nu_up[l + 1].reshape(self.v, -1, self.s)

        V = V.permute(0, 2, 1)
        flatR = R.reshape(-1, self.s)
        proba_rules = V[flatR, torch.arange(self.s)].squeeze().prod(1)

        return proba_rules.reshape(self.v, self.m, -1)

    def BP_upward_iteration(self):
        """
        Compute the upward messages using the upward rules.
        """
        for l in range(self.L - 1, -1, -1):
            proba_rules = self.upward_rule_to_proba(l)
            self.nu_up[l] = proba_rules.sum(1)
            self.nu_up[l] = self.nu_up[l] / self.nu_up[l].sum(axis=0, keepdims=True)
            # if nu_up is nan, set to uniform
            # self.nu_up[l] = torch.where(
            #     torch.isnan(self.nu_up[l]), torch.ones_like(self.nu_up[l]) / self.v, self.nu_up[l]
            # )
        return self.nu_up

    def compute_downward_messages(self, l):
        """
        Compute the downward messages from layer l to layer l+1.
        """
        R = self.rules[l]
        Pclass = self.nu_down[l].reshape(self.n, -1)
        V = self.nu_up[l + 1].reshape(self.v, -1, self.s)

        flatR = R.reshape(-1, self.s)  # (n*m, s)

        proba_rules = (
            V[flatR.flatten(), :, torch.arange(self.s).repeat(self.n * self.m)]
            .reshape(self.n * self.m, self.s, -1)
            .permute(0, 2, 1)
        )  # (n*m, I, s)

        proba_rules = proba_rules.unsqueeze(-1) @ torch.ones(
            (1, self.s)
        ).to(self.device)  # (n*m, I, s, s) Create matrix (nu1, nu1; nu2, nu2)
        proba_rules[:, :, torch.arange(self.s), torch.arange(self.s)] = (
            1.0  # Set the diagonal to 1
        )
        proba_rules = proba_rules.prod(-2).reshape(
            self.n, self.m, -1, self.s
        )  # (n, m, I, s) Multiply the columns of the matrix of upward messages

        proba_rules = (
            Pclass[:, None, :, None] * proba_rules
        )  # (n, m, I, s) Multiply the class probabilities by the product of upward messages for each rule
        proba_rules = proba_rules.reshape(self.n * self.m, -1, self.s)  # (n*m, I, s)

        sum_rules = torch.zeros(self.v, *proba_rules.shape).to(self.device)  # (v, n*m, I, s)
        index_rules = torch.arange(self.n * self.m).repeat_interleave(self.s)
        index_patch = torch.arange(self.s).repeat(self.n * self.m)

        sum_rules[flatR.flatten(), index_rules, :, index_patch] = proba_rules[
            index_rules, :, index_patch
        ]  # (v, n*m, I, s)

        sum_rules = sum_rules.sum(1)  # (v, I, s) Sum over the rules

        return sum_rules

    def BP_downward_iteration(self):
        """
        Compute the upward messages using the upward rules.
        """
        for l in range(0, self.L):
            self.nu_down[l + 1] = self.compute_downward_messages(
                l
            )  # Compute the unnormalized messages
            self.nu_down[l + 1] = self.nu_down[l + 1] / self.nu_down[l + 1].sum(
                axis=0, keepdims=True
            )  # Normalize the messages
            # if nu_down is nan, set to uniform
            # self.nu_down[l + 1] = torch.where(
            #     torch.isnan(self.nu_down[l + 1]),
            #     torch.ones_like(self.nu_down[l + 1]) / self.v,
            #     self.nu_down[l + 1],
            # )
        return self.nu_down

    def compute_variable_marginals(self, l):
        marginals = self.nu_up[l].reshape(self.nu_up[l].shape[0], -1) * self.nu_down[
            l
        ].reshape(self.nu_down[l].shape[0], -1)
        marginals = marginals / marginals.sum(axis=0, keepdims=True)
        return marginals

    def compute_all_marginals(self):
        """
        Compute the marginals for all the variables of the tree.
        """
        marginals = {}
        for l in self.nu_up.keys():
            marginals[l] = self.compute_variable_marginals(l)
        return marginals

    def compute_rules_marginals(self, proba_rules, nu_down):
        proba_rules = nu_down.unsqueeze(1) * proba_rules
        proba_rules = proba_rules.reshape(
            proba_rules.shape[0] * proba_rules.shape[1], -1
        )
        proba_rules = proba_rules / proba_rules.sum(axis=0, keepdims=True)
        return proba_rules

    def sample_multinomial(self, p, g):
        """
        Sample rules given the probabilities p.

        p: probabilities of the rules shape = (v*m, I) or of the variables shape = (v, I)
        g: random generator
        """
        sampled_rules = torch.multinomial(p.T, num_samples=1, generator=g)
        return sampled_rules, g

    def update_nu_down(self, l, sampled_x):
        """
        Set the sampled values to the messages.
        """
        num_variables = sampled_x.shape[-1]
        self.nu_down[l] = torch.zeros((self.v, num_variables), device=self.device)
        self.nu_down[l][sampled_x.flatten(), torch.arange(num_variables)] = 1.0
        return self.nu_down[l]

    def BP_downward_sampling(self, seed=0):

        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        class_marginal = self.compute_variable_marginals(0)
        sampled_x, g = self.sample_multinomial(class_marginal, g)
        self.update_nu_down(0, sampled_x)

        for l in range(0, self.L):
            proba_rules = self.upward_rule_to_proba(l)
            rules_marginals = self.compute_rules_marginals(proba_rules, self.nu_down[l])
            sampled_rules, g = self.sample_multinomial(rules_marginals, g)

            index_rules = sampled_rules.repeat_interleave(self.s)
            index_patch = torch.arange(self.s, device=self.device).repeat(
                len(sampled_rules)
            )

            flatR = self.rules[l].reshape(-1, self.s)
            x = flatR[index_rules, index_patch]
            self.update_nu_down(l + 1, x)

        return x

    def run_BP_from_upward_messages(self, nu_up_L):
        """
        Sample from the evidence at the leaf level.
        The noise level is set so that the closest variables are sampled when BP meets inconsistencies in rules.
        """
        self.initalize_messages(nu_up_L=nu_up_L)
        self.BP_upward_iteration()
        self.BP_downward_iteration()
        marginals = self.compute_all_marginals()

        return marginals

    def sample_from_upward_messages(self, nu_up_L, seed=0):
        """
        Sample from the evidence at the leaf level.
        """
        self.initalize_messages(nu_up_L=nu_up_L)
        self.BP_upward_iteration()
        x = self.BP_downward_sampling(seed)
        return x

    def run_BP_from_evidence(self, x_leaves, noise=0.0):
        """
        Sample from the evidence at the leaf level.
        The noise level is set so that the closest variables are sampled when BP meets inconsistencies in rules.
        """
        B, d = x_leaves.shape
        x_leaves = x_leaves.flatten()

        nu_up_L = self.set_evidence_to_leaf_messages(x_leaves, noise=noise)
        marginals = self.run_BP_from_upward_messages(nu_up_L)

        for l in marginals.keys():
            marginals[l] = marginals[l].reshape(self.v, B, -1)

        return marginals

    def sample_from_evidence(self, x_leaves, seed=0):
        """
        Sample from the evidence at the leaf level.
        The noise level is set so that the closest variables are sampled when BP meets inconsistencies in rules.
        """
        B, d = x_leaves.shape
        x_leaves = x_leaves.flatten()

        nu_up_L = self.set_evidence_to_leaf_messages(x_leaves, noise=1e-3)
        x = self.sample_from_upward_messages(nu_up_L, seed=seed)

        return x.reshape(B, d)
    

def BP_countcorrect_upward(x, bp):
    """
    Check if the data x is correctly propagated upwards.

    x: data at the leaf level of the hierarchy. shape = (B, d)
    bp: belief propagation object

    return: Dictionary with the number of errors at each level of the hierarchy.
    """

    def _upward_rule_to_proba_(nu_up, l):
        """
        Compute the compatibility of the rules given the upward messages.
        l: level of the hierarchy
        return: probabilities of the rules. shape = (v, m, I)
        """

        R = bp.rules[l]
        V = nu_up.reshape(bp.v, -1, bp.s)

        V = V.permute(0, 2, 1)
        flatR = R.reshape(-1, bp.s)
        proba_rules = V[flatR, torch.arange(bp.s)].squeeze().prod(1)

        return proba_rules.reshape(bp.v, bp.m, -1)

    B, d = x.shape
    x = x.flatten()

    nu_up = {}
    frac_correct = {}
    rules_frequencies = {}

    nu_up[bp.L] = bp.set_evidence_to_leaf_messages(x, noise=0.0) # set the evidence to the leaf messages
    for l in range(bp.L - 1, -1, -1):
        proba_rules = _upward_rule_to_proba_(nu_up[l+1], l)
        nu_up[l] = proba_rules.sum(1) # sum over the m possible rules for each variable (v, B * s**l)

        proba_rules = proba_rules.reshape(bp.v, bp.m, B, -1) # proba rules for each variable (v, m, B * s**l)
        frac_correct[l] = proba_rules.sum(dim=(0,1,2)) / B
        rules_frequencies[l] = proba_rules.sum(dim=(2, 3)).flatten()  # (v*m)

    return frac_correct, rules_frequencies
