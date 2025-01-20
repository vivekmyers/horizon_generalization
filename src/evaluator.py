import time
import jax
import numpy as np
from brax.training import acting
from brax.training.types import Metrics
from brax.training.types import PolicyParams

# This is an evaluator that behaves in the exact same way as brax Evaluator,
# but additionally it aggregates metrics with max, min.
# It also logs in how many episodes there was any success.
class CrlEvaluator(acting.Evaluator):
    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            (np.std, "_std"),
            (np.max, "_max"),
            (np.min, "_min"),
        ]

        total_distance = eval_state.metrics["total_distance"]
        bins = np.arange(0., 250., 5.)
        idx = np.digitize(total_distance, bins)
        assert (total_distance >= 0).all() and (total_distance <= 250).all()

        metrics["eval/total_distance"] = np.mean(total_distance)
        for bin_idx in range(len(bins)):
            metrics[f"eval/total_distance_bin{bin_idx}"] = np.mean(
                total_distance[idx == bin_idx]
            )

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(eval_metrics.episode_metrics[name])
                        if aggregate_episodes
                        else eval_metrics.episode_metrics[name]
                    )
                    for name in eval_metrics.episode_metrics
                }
            )

        for fn, suffix in aggregating_fns:
            for bin_idx in range(len(bins)):
                for name in eval_metrics.episode_metrics:
                    assert eval_metrics.episode_metrics[name].shape == total_distance.shape
                    vals = eval_metrics.episode_metrics[name][idx == bin_idx]

                    fn_ = lambda x: np.nan_to_num(fn(x) if len(x) > 0 else vals.mean())
                    metrics.update(
                        {
                            f"eval/episode_{name}{suffix}_bin{bin_idx}": (
                                fn_(vals) if aggregate_episodes else vals
                            )
                        }
                    )


        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(
                eval_metrics.episode_metrics["success"] > 0.0
            )
            for bin_idx in range(len(bins)):
                metrics[f"eval/episode_success_any_bin{bin_idx}"] = np.mean(
                    eval_metrics.episode_metrics["success"][idx == bin_idx] > 0.0
                )
        if "success_easy" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_easy"] = np.mean(
                eval_metrics.episode_metrics["success_easy"] > 0.0
            )
            for bin_idx in range(len(bins)):
                metrics[f"eval/episode_success_easy_bin{bin_idx}"] = np.mean(
                    eval_metrics.episode_metrics["success_easy"][idx == bin_idx] > 0.0
                )

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray
