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

        # Beta(3,2) gives a right-leaning distribution so most agents
        # are moderately conservative by disposition
        if intrinsic_conservatism is None:
            intrinsic_conservatism = float(model.rng_np.beta(3, 2))
        self.intrinsic_conservatism = intrinsic_conservatism

        self.selected_topic = None
        self.recognized_this_step = False
        self.total_recognized_novel = 0

    @property
    def risk_bearing_capacity(self):
        """Mean of reputation and structural insulation. Float in [0, 1]."""
        return (self.reputation + self.structural_insulation) / 2

    def step(self):
        """Find reachable topics and choose one."""
        candidates = self._get_candidates()
        self.selected_topic = self._choose_topic(candidates)

    ## Bounded search — agents can only "see" nearby topics on the graph
    def _get_candidates(self):
        """Return list of (node, distance) pairs within search radius."""
        G = self.model.landscape
        radius = self.model.base_radius + int(
            self.model.expansion_bonus * self.risk_bearing_capacity
        )
        lengths = nx.single_source_shortest_path_length(
            G, self.position, cutoff=radius
        )
        candidates = [(node, dist) for node, dist in lengths.items() if dist > 0]
        candidates.sort(key=lambda x: x[1])
        return candidates

    ## Core decision rule — constrained weighted choice
    def _choose_topic(self, candidates):
        """Pick the best topic from candidates using career threshold + weighted utility."""
        if not candidates:
            return self.position

        # Career threshold gate: high-rbc agents tolerate riskier topics
        career_thr = 0.7 - self.model.aspiration_bonus * self.risk_bearing_capacity

        viable = [
            node for node, _dist in candidates
            if self._career_viability(node) >= career_thr
        ]

        if viable:
            # Blend intrinsic disposition with structural position: conservative
            # agents stay career-focused even with high rbc, while dispositionally
            # open agents shift toward epistemic interest as rbc grows.
            # aspiration_bonus amplifies how much rbc reduces career_weight
            rbc_effect = self.model.aspiration_bonus * self.risk_bearing_capacity
            career_weight = (
                self.intrinsic_conservatism * (1.0 - 0.5 * rbc_effect)
                + (1.0 - self.intrinsic_conservatism) * (1.0 - rbc_effect)
            )
            career_weight = max(career_weight, 0.0)
            return max(viable, key=lambda n: self._topic_attractiveness(n, career_weight))

        # Fallback: nothing passes threshold, pick safest option
        return max(candidates, key=lambda x: self._career_viability(x[0]))[0]

    def _topic_attractiveness(self, topic_node, career_weight):
        """Weighted sum of career viability and epistemic potential."""
        cv = self._career_viability(topic_node)
        ep = self.model.landscape.nodes[topic_node]["epistemic_potential"]
        return career_weight * cv + (1 - career_weight) * ep

    ## Channel A: anticipated field response
    def _career_viability(self, topic_node):
        """How publishable the agent expects this topic to be. Implements Channel A."""
        recog = self.model.landscape.nodes[topic_node]["recognizability"]
        bias = self.model.recognition_bias
        # perceived publishability: when bias is high, low-recog topics look risky
        perceived_pub = bias * recog + (1 - bias) * 0.7
        # reputation buffers against field bias: at rep=1, CV=1 for any topic
        return perceived_pub + self.reputation * (1 - perceived_pub)
