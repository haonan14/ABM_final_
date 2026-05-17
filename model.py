"""Scientific Search ABM — main model."""

from mesa import Model, DataCollector
from math import log
import numpy as np
import networkx as nx

from agents import Scientist


class ScientificSearchModel(Model):
    """
    Scientists choose research topics each step, recognition is resolved
    stochastically, and feedback effects update agent/topic attributes.
    Three channels produce collective conservatism:
      A — anticipated deterrence (pre-choice)
      B — differential recognition (post-choice)
      C — reputation pipeline (feedback loop)
    """

    def __init__(
        self,
        n_agents=100,
        n_topics=200,
        n_clusters=5,
        p_intra=0.3,
        p_inter=0.02,
        frac_established=0.50,
        frac_recognizable=0.30,
        base_radius=2,
        expansion_bonus=2,
        aspiration_bonus=0.4,
        recognition_base=0.3,
        recognition_bias=0.5,
        reputation_weight=0.3,
        crowding_penalty=0.05,
        rep_gain=0.01,
        rep_decay=0.003,
        domestication_rate=0.03,
        depletion_rate=0.005,
        regeneration_rate=0.002,
        pressure_increment=0.005,
        epistemic_floor=0.1,
        insulation_floor=0.05,
        turnover_rate=0.05,
        max_steps=200,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.rng_np = np.random.default_rng(seed)
        self.max_steps = max_steps
        self._step_count = 0

        self.n_agents = n_agents
        self.base_radius = base_radius
        self.expansion_bonus = expansion_bonus
        self.aspiration_bonus = aspiration_bonus
        self.recognition_base = recognition_base
        self.recognition_bias = recognition_bias
        self.reputation_weight = reputation_weight
        self.crowding_penalty = crowding_penalty
        self.rep_gain = rep_gain
        self.rep_decay = rep_decay
        self.domestication_rate = domestication_rate
        self.depletion_rate = depletion_rate
        self.regeneration_rate = regeneration_rate
        self.pressure_increment = pressure_increment
        self.epistemic_floor = epistemic_floor
        self.insulation_floor = insulation_floor
        self.turnover_rate = turnover_rate

        self.landscape = self._create_landscape(
            n_topics, n_clusters, p_intra, p_inter,
            frac_established, frac_recognizable,
        )

        ## Here I snapshot the initial topic values before any domestication/depletion
        ## occurs, so that my DV measurements aren't confounded by the dynamics themselves
        self._initial_recog = {}
        self._initial_ep = {}
        for n, d in self.landscape.nodes(data=True):
            self._initial_recog[n] = d["recognizability"]
            self._initial_ep[n] = d["epistemic_potential"]

        self._create_agents()

        ## Rolling buffers for smoothed display — I use a window of 10 steps
        ## to reduce the step-to-step noise that made the plots hard to read
        self._smooth_window = 10
        self._cc_buffer = []
        self._rer_buffer = []

        self.datacollector = DataCollector(
            model_reporters={
                "Collective_Conservatism": collective_conservatism,
                "Radical_Exploration_Rate": radical_exploration_rate,
                "Innovation_Inequality": innovation_inequality,
                "Domestication_Progress": domestication_progress,
                "CC_Smoothed": cc_smoothed,
                "RER_Smoothed": rer_smoothed,
            },
        )
        self.datacollector.collect(self)

    def _create_landscape(
        self, n_topics, n_clusters, p_intra, p_inter,
        frac_established, frac_recognizable,
    ):
        """Build knowledge landscape as stochastic block model graph."""
        n_established = int(n_topics * frac_established)
        n_recognizable = int(n_topics * frac_recognizable)
        n_radical = n_topics - n_established - n_recognizable

        ## Here I use networkx stochastic_block_model to create a modular network —
        ## I needed disciplinary clusters with realistic community structure and SBM
        ## lets me parameterize intra/inter connection density directly
        n_main = n_established + n_recognizable
        sizes = balanced_partition(n_main, n_clusters)
        probs = [
            [p_intra if i == j else p_inter for j in range(n_clusters)]
            for i in range(n_clusters)
        ]
        G = nx.stochastic_block_model(sizes, probs, seed=int(self.rng_np.integers(1e9)))

        est_ratio = frac_established / (frac_established + frac_recognizable)
        node_idx = 0
        for cluster_id, size in enumerate(sizes):
            cluster_nodes = list(range(node_idx, node_idx + size))
            self.rng_np.shuffle(cluster_nodes)
            n_est = int(len(cluster_nodes) * est_ratio)

            for n in cluster_nodes[:n_est]:
                G.nodes[n].update({
                    "node_type": "established",
                    "cluster": cluster_id,
                    "recognizability": float(self.rng_np.beta(8, 2)),
                    "epistemic_potential": float(self.rng_np.beta(2, 5)),
                    "times_selected": 0,
                })
            for n in cluster_nodes[n_est:]:
                G.nodes[n].update({
                    "node_type": "recognizable",
                    "cluster": cluster_id,
                    "recognizability": float(self.rng_np.beta(5, 3)),
                    "epistemic_potential": float(self.rng_np.beta(5, 3)),
                    "times_selected": 0,
                })
            node_idx += size

        ## Here I add radical topics as sparse fringe nodes with only 1-3 edges
        ## to the core network — this makes them structurally distant so only
        ## high-rbc agents with expanded search radius can reach them
        main_nodes = list(G.nodes())
        for i in range(n_radical):
            rid = n_main + i
            G.add_node(rid, **{
                "node_type": "radical",
                "cluster": -1,
                "recognizability": float(self.rng_np.beta(2, 7)),
                "epistemic_potential": float(self.rng_np.beta(7, 2)),
                "times_selected": 0,
            })
            n_conn = self.rng_np.integers(1, 4)
            targets = self.rng_np.choice(main_nodes, size=n_conn, replace=False)
            for t in targets:
                G.add_edge(rid, int(t))

        return G

    def _create_agents(self):
        """Create scientist population with right-skewed reputation.
        Beta(1,5) so most agents start low, matching early-career reality."""
        reps = self.rng_np.beta(1, 5, size=self.n_agents)
        insul_base = self.rng_np.beta(2, 4, size=self.n_agents)
        noise = self.rng_np.normal(0, 0.15, size=self.n_agents)
        insuls = np.clip(0.7 * insul_base + 0.3 * reps + noise, 0, 1)

        nodes_by_cluster = {}
        for node, data in self.landscape.nodes(data=True):
            c = data["cluster"]
            if c >= 0:
                nodes_by_cluster.setdefault(c, []).append(node)
        n_clusters = len(nodes_by_cluster)

        for i in range(self.n_agents):
            cid = i % n_clusters
            pos = self.rng_np.choice(nodes_by_cluster[cid])
            Scientist(
                model=self,
                reputation=float(reps[i]),
                structural_insulation=float(insuls[i]),
                position=int(pos),
            )

    def step(self):
        """One step: choose -> recognize -> regenerate -> turnover -> collect."""
        self.agents_by_type[Scientist].do("step")
        self._resolve_recognition()
        self._regenerate_ep()
        self._turnover()
        self.datacollector.collect(self)
        self._step_count += 1
        if self._step_count >= self.max_steps:
            self.running = False

    def _regenerate_ep(self):
        """Unresearched topics slowly regain EP — without this the frontier
        exhausts around step 300 and everything converges."""
        # TODO: might need to make regeneration rate topic-specific?
        for n, d in self.landscape.nodes(data=True):
            if d["node_type"] in ("recognizable", "radical"):
                cap = self._initial_ep[n]
                if d["epistemic_potential"] < cap:
                    d["epistemic_potential"] = min(
                        d["epistemic_potential"] + self.regeneration_rate, cap
                    )

    def _turnover(self):
        """Replace lowest-rep agents with fresh entrants. Without this everyone
        eventually reaches high rbc and conservatism disappears entirely."""
        n_retire = int(self.n_agents * self.turnover_rate)
        if n_retire == 0:
            return

        scientists = sorted(
            self.agents_by_type[Scientist],
            key=lambda a: a.reputation,
        )
        retirees = scientists[:n_retire]

        nodes_by_cluster = {}
        for node, data in self.landscape.nodes(data=True):
            c = data["cluster"]
            if c >= 0:
                nodes_by_cluster.setdefault(c, []).append(node)
        n_clusters = len(nodes_by_cluster)

        for agent in retirees:
            agent.remove()
            rep = float(self.rng_np.beta(1, 5))
            insul = float(np.clip(0.7 * self.rng_np.beta(2, 4) + 0.3 * rep
                                  + self.rng_np.normal(0, 0.15), 0, 1))
            cid = int(self.rng_np.integers(n_clusters))
            pos = self.rng_np.choice(nodes_by_cluster[cid])
            Scientist(
                model=self,
                reputation=rep,
                structural_insulation=insul,
                position=int(pos),
            )

    ## Channel B — the field actually does favor legible work
    def _resolve_recognition(self):
        """Resolve recognition for all agents this step."""
        scientists = list(self.agents_by_type[Scientist])

        # crowding penalty: more agents on same topic = lower recognition chance
        topic_counts = {}
        for agent in scientists:
            t = agent.selected_topic
            if t is not None:
                topic_counts[t] = topic_counts.get(t, 0) + 1

        for agent in scientists:
            topic = agent.selected_topic
            if topic is None:
                continue

            n_others = topic_counts.get(topic, 1) - 1
            p_recog = self._recognition_probability(agent, topic, n_others)
            recognized = self.rng_np.random() < p_recog

            agent.recognized_this_step = recognized
            tdata = self.landscape.nodes[topic]
            tdata["times_selected"] += 1

            if recognized:
                agent.reputation = min(agent.reputation + self.rep_gain, 1.0)
                if tdata["recognizability"] < 0.5:
                    agent.total_recognized_novel += 1
                # domestication + depletion
                tdata["recognizability"] = min(
                    tdata["recognizability"] + self.domestication_rate, 1.0
                )
                tdata["epistemic_potential"] = max(
                    tdata["epistemic_potential"] - self.depletion_rate,
                    self.epistemic_floor,
                )
                agent.position = topic
            else:
                # career pressure from failed risk-taking
                agent.structural_insulation = max(
                    agent.structural_insulation - self.pressure_increment,
                    self.insulation_floor,
                )

            # publish-or-perish decay
            agent.reputation = max(agent.reputation - self.rep_decay, 0.0)

    def _recognition_probability(self, agent, topic, n_others):
        """P(recognition) = base + bias*(recog-0.5) + rep_weight*rep - crowding."""
        recog = self.landscape.nodes[topic]["recognizability"]
        crowding = self.crowding_penalty * log(1 + n_others)
        p = (
            self.recognition_base
            + self.recognition_bias * (recog - 0.5)
            + self.reputation_weight * agent.reputation
            - crowding
        )
        return float(np.clip(p, 0.0, 1.0))


## ── DVs ──

def collective_conservatism(model):
    # use INITIAL recognizability so domestication doesn't confound the measure
    recogs = []
    for a in model.agents_by_type[Scientist]:
        if a.selected_topic is not None:
            recogs.append(model._initial_recog[a.selected_topic])
    return float(np.mean(recogs)) if recogs else 0.0


def radical_exploration_rate(model):
    n_radical = 0
    n_total = 0
    for a in model.agents_by_type[Scientist]:
        if a.selected_topic is not None:
            n_total += 1
            if model.landscape.nodes[a.selected_topic]["node_type"] == "radical":
                n_radical += 1
    return n_radical / n_total if n_total > 0 else 0.0


def innovation_inequality(model):
    """Gini of risk-bearing capacity. I use rbc rather than lifetime counts
    because turnover resets those to 0 and collapses the measure."""
    agent_rbcs = [a.get_rbc() for a in model.agents_by_type[Scientist]]
    if len(agent_rbcs) == 0 or sum(agent_rbcs) == 0:
        return 0.0
    sorted_rbcs = sorted(agent_rbcs)
    n = len(sorted_rbcs)
    x = sum(el * (n - ind) for ind, el in enumerate(sorted_rbcs)) / (n * sum(sorted_rbcs))
    return 1 + (1 / n) - 2 * x


def domestication_progress(model):
    deltas = []
    for n, d in model.landscape.nodes(data=True):
        if d["node_type"] in ("recognizable", "radical") and d["times_selected"] > 0:
            deltas.append(d["recognizability"] - model._initial_recog[n])
    return float(np.mean(deltas)) if deltas else 0.0


## smoothed versions for the dashboard — raw CC and RER jump around too much per step

def cc_smoothed(model):
    raw = collective_conservatism(model)
    model._cc_buffer.append(raw)
    if len(model._cc_buffer) > model._smooth_window:
        model._cc_buffer.pop(0)
    return float(np.mean(model._cc_buffer))


def rer_smoothed(model):
    raw = radical_exploration_rate(model)
    model._rer_buffer.append(raw)
    if len(model._rer_buffer) > model._smooth_window:
        model._rer_buffer.pop(0)
    return float(np.mean(model._rer_buffer))


def balanced_partition(n, k):
    base = n // k
    rem = n % k
    return [base + (1 if i < rem else 0) for i in range(k)]
