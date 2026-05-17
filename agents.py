"""Scientist agent for the Scientific Search ABM."""

from mesa import Agent
import networkx as nx


class Scientist(Agent):
    """A scientist who chooses research topics on a knowledge landscape."""

    def __init__(self, model, reputation, structural_insulation, position,
                 intrinsic_conservatism=None):
        super().__init__(model)
        self.reputation = reputation
        self.structural_insulation = structural_insulation
        self.position = position

        # Beta(3,2) so most agents lean conservative by disposition
        if intrinsic_conservatism is None:
            intrinsic_conservatism = float(model.rng_np.beta(3, 2))
        self.intrinsic_conservatism = intrinsic_conservatism

        self.selected_topic = None
        self.recognized_this_step = False
        self.total_recognized_novel = 0

    def get_rbc(self):
        return (self.reputation + self.structural_insulation) / 2

    def step(self):
        """Find reachable topics and choose one."""
        candidates = self._get_candidates()
        self.selected_topic = self._choose_topic(candidates)

    def _get_candidates(self):
        """Return reachable topics within search radius (high rbc = wider radius)."""
        G = self.model.landscape
        radius = self.model.base_radius + int(
            self.model.expansion_bonus * self.get_rbc()
        )
        lengths = nx.single_source_shortest_path_length(
            G, self.position, cutoff=radius
        )
        candidates = [(node, dist) for node, dist in lengths.items() if dist > 0]
        candidates.sort(key=lambda x: x[1])
        return candidates

    def _choose_topic(self, candidates):
        """Career threshold gate, then weighted utility maximization."""
        if not candidates:
            return self.position

        # threshold gate: high-rbc agents tolerate riskier topics
        career_thr = 0.7 - self.model.aspiration_bonus * self.get_rbc()

        viable = [
            node for node, _dist in candidates
            if self._career_viability(node) >= career_thr
        ]

        if viable:
            rbc_effect = self.model.aspiration_bonus * self.get_rbc()
            career_weight = (
                self.intrinsic_conservatism * (1.0 - 0.5 * rbc_effect)
                + (1.0 - self.intrinsic_conservatism) * (1.0 - rbc_effect)
            )
            career_weight = max(career_weight, 0.0)
            return max(viable, key=lambda n: self._topic_attractiveness(n, career_weight))

        # fallback: nothing viable, pick safest
        return max(candidates, key=lambda x: self._career_viability(x[0]))[0]

    def _topic_attractiveness(self, topic_node, career_weight):
        cv = self._career_viability(topic_node)
        ep = self.model.landscape.nodes[topic_node]["epistemic_potential"]
        return career_weight * cv + (1 - career_weight) * ep

    ## Channel A — anticipated field response (pre-choice deterrent)
    def _career_viability(self, topic_node):
        # how publishable does this topic look to the agent?
        # at rep=1 everything looks publishable (CV → 1)
        recog = self.model.landscape.nodes[topic_node]["recognizability"]
        bias = self.model.recognition_bias
        perceived_pub = bias * recog + (1 - bias) * 0.7
        return perceived_pub + self.reputation * (1 - perceived_pub)
