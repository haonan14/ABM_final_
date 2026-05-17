"""Scientific Search ABM — main model. Simulates scientists choosing topics on a knowledge landscape."""

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

        ## Store parameters as instance attributes
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

        ## Build environment and agents
        self.landscape = self._create_landscape(
            n_topics, n_clusters, p_intra, p_inter,
            frac_established, frac_recognizable,
        )

        # Snapshot initial values — needed for conservatism DV and visualization
        # so domestication/depletion don't confound the measurement or move the dots
        self._initial_recog = {
            n: d["recognizability"] for n, d in self.landscape.nodes(data=True)
        }
        self._initial_ep = {
            n: d["epistemic_potential"] for n, d in self.landscape.nodes(data=True)
        }

        self._create_agents()

        ## Rolling buffers for smoothed display (window=10)
        self._smooth_window = 10
        self._cc_buffer = []
        self._rer_buffer = []

        ## Data collection: raw DVs + smoothed versions for visualization
        self.datacollector = DataCollector(
            model_reporters={
                "Collective_Conservatism": _collective_conservatism,
                "Radical_Exploration_Rate": _radical_exploration_rate,
                "Innovation_Inequality": _innovation_inequality,
                "Domestication_Progress": _domestication_progress,
                "CC_Smoothed": _cc_smoothed,
                "RER_Smoothed": _rer_smoothed,
            },
        )
        self.datacollector.collect(self)

    ## Landscape construction
    def _create_landscape(
        self, n_topics, n_clusters, p_intra, p_inter,
        frac_established, frac_recognizable,
    ):
        """Build knowledge landscape as a stochastic block model graph."""
        n_established = int(n_topics * frac_established)
        n_recognizable = int(n_topics * frac_recognizable)
        n_radical = n_topics - n_established - n_recognizable

        # Stochastic block model creates a modular network with disciplinary
        # clusters, giving the landscape realistic community structure
        n_main = n_established + n_recognizable
        sizes = _balanced_partition(n_main, n_clusters)
        probs = [
            [p_intra if i == j else p_inter for j in range(n_clusters)]
            for i in range(n_clusters)
        ]
        G = nx.stochastic_block_model(sizes, probs, seed=int(self.rng_np.integers(1e9)))

        # Assign topic types within each cluster
        est_ratio = frac_established / (frac_established + frac_recognizable)
        node_idx = 0
        for cluster_id, size in enumerate(sizes):
            cluster_nodes = list(range(node_idx, node_idx + size))
            self.rng_np.shuffle(cluster_nodes)
            n_est = int(len(cluster_nodes) * est_ratio)

            # Established: high recognizability, low epistemic potential
            for n in cluster_nodes[:n_est]:
                G.nodes[n].update({
                    "node_type": "established",
                    "cluster": cluster_id,
                    "recognizability": float(self.rng_np.beta(8, 2)),
                    "epistemic_potential": float(self.rng_np.beta(2, 5)),
                    "times_selected": 0,
                })
            # Recognizably novel: moderate on both dimensions
            for n in cluster_nodes[n_est:]:
                G.nodes[n].update({
                    "node_type": "recognizable",
                    "cluster": cluster_id,
                    "recognizability": float(self.rng_np.beta(5, 3)),
                    "epistemic_potential": float(self.rng_np.beta(5, 3)),
                    "times_selected": 0,
                })
            node_idx += size

        # Radical topics: sparse fringe nodes with 1-3 edges to core
        # Structurally distant so only high-rbc agents can reach them
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

    ## Agent initialization
    def _create_agents(self):
        """Create scientist population with right-skewed reputation."""
        reps = self.rng_np.beta(1, 5, size=self.n_agents)
        insul_base = self.rng_np.beta(2, 4, size=self.n_agents)
        noise = self.rng_np.normal(0, 0.15, size=self.n_agents)
        insuls = np.clip(0.7 * insul_base + 0.3 * reps + noise, 0, 1)

        # Place agents round-robin across clusters
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

    ## Step logic
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
        """Unresearched topics slowly regain EP, modeling how new questions
        emerge from existing knowledge. Without this the frontier exhausts
        by ~t=300 and the model enters a degenerate steady state."""
        for n, d in self.landscape.nodes(data=True):
            if d["node_type"] in ("recognizable", "radical"):
                cap = self._initial_ep[n]
                if d["epistemic_potential"] < cap:
                    d["epistemic_potential"] = min(
                        d["epistemic_potential"] + self.regeneration_rate, cap
                    )

    def _turnover(self):
        """Replace lowest-rep agents with fresh entrants each step.
        Without this, the entire population eventually reaches high rbc
        and conservatism disappears as a steady-state outcome."""
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

    ## Channel B: differential recognition
    def _resolve_recognition(self):
        """Resolve recognition for all agents. Updates reputation, topic attributes, positions."""
        scientists = list(self.agents_by_type[Scientist])

        # Count agents per topic for crowding penalty
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
                # Domestication: recognized work makes topic more legible,
                # but capped by node_type — radical work can never become
                # fully mainstream (preserves structural risk)
                recog_cap = {"established": 1.0, "recognizable": 0.8, "radical": 0.4}
                cap = recog_cap[tdata["node_type"]]
                tdata["recognizability"] = min(
                    tdata["recognizability"] + self.domestication_rate, cap
                )
                # Depletion: researched topics lose epistemic potential
                tdata["epistemic_potential"] = max(
                    tdata["epistemic_potential"] - self.depletion_rate,
                    self.epistemic_floor,
                )
                agent.position = topic
            else:
                # Career pressure: failed attempts erode insulation
                agent.structural_insulation = max(
                    agent.structural_insulation - self.pressure_increment,
                    self.insulation_floor,
                )

            # Publish-or-perish: reputation decays every step
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


## ── Model-level reporters (dependent variables) ──

def _collective_conservatism(model):
    """Mean INITIAL recognizability of selected topics. Uses pre-domestication
    values so the DV isn't confounded by topics becoming legible over time."""
    recogs = []
    for a in model.agents_by_type[Scientist]:
        if a.selected_topic is not None:
            recogs.append(model._initial_recog[a.selected_topic])
    return float(np.mean(recogs)) if recogs else 0.0


def _radical_exploration_rate(model):
    """Fraction of agents on originally-radical topics this step."""
    n_radical = 0
    n_total = 0
    for a in model.agents_by_type[Scientist]:
        if a.selected_topic is not None:
            n_total += 1
            if model.landscape.nodes[a.selected_topic]["node_type"] == "radical":
                n_radical += 1
    return n_radical / n_total if n_total > 0 else 0.0


def _innovation_inequality(model):
    """Gini coefficient of risk-bearing capacity across agents."""
    vals = np.array([a.risk_bearing_capacity for a in model.agents_by_type[Scientist]], dtype=float)
    if vals.sum() == 0:
        return 0.0
    s = np.sort(vals)
    n = len(s)
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * s) - (n + 1) * np.sum(s)) / (n * np.sum(s)))


def _domestication_progress(model):
    """Mean recognizability gain for novel topics that have been explored."""
    deltas = []
    for n, d in model.landscape.nodes(data=True):
        if d["node_type"] in ("recognizable", "radical") and d["times_selected"] > 0:
            deltas.append(d["recognizability"] - model._initial_recog[n])
    return float(np.mean(deltas)) if deltas else 0.0


## ── Smoothed reporters (rolling mean to reduce step-to-step noise in viz) ──

def _cc_smoothed(model):
    """Rolling-mean CC for cleaner visualization."""
    raw = _collective_conservatism(model)
    model._cc_buffer.append(raw)
    if len(model._cc_buffer) > model._smooth_window:
        model._cc_buffer.pop(0)
    return float(np.mean(model._cc_buffer))


def _rer_smoothed(model):
    """Rolling-mean RER for cleaner visualization."""
    raw = _radical_exploration_rate(model)
    model._rer_buffer.append(raw)
    if len(model._rer_buffer) > model._smooth_window:
        model._rer_buffer.pop(0)
    return float(np.mean(model._rer_buffer))


def _balanced_partition(n, k):
    """Split n items into k groups as evenly as possible."""
    base = n // k
    rem = n % k
    return [base + (1 if i < rem else 0) for i in range(k)]
