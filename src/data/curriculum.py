import numpy as np
from typing import List, Dict, Any, Optional

class CurriculumTracker:
    """
    Tracks training and validation losses per operation category,
    computes adaptive sampling weights using Softmax over losses,
    and detects and mitigates category-level overfitting.
    """
    def __init__(
        self,
        categories: List[str],
        enabled: bool = True,
        ema_alpha: float = 0.2,
        temperature: float = 1.0,
        floor_prob: float = 0.01,
        overfit_ratio: float = 3.0,
        overfit_train_threshold: float = 0.3,
        overfit_decay: float = 0.1,
        update_interval: int = 100
    ):
        self.categories = categories
        self.num_categories = len(categories)
        self.enabled = enabled
        self.ema_alpha = ema_alpha
        self.temperature = temperature
        self.floor_prob = min(floor_prob, 1.0 / self.num_categories)
        self.overfit_ratio = overfit_ratio
        self.overfit_train_threshold = overfit_train_threshold
        self.overfit_decay = overfit_decay
        self.update_interval = update_interval
        
        # Initialize EMAs with a reasonable default (e.g. 1.0) so they don't start at 0
        self.train_losses = np.ones(self.num_categories, dtype=np.float32) * 1.5
        self.val_losses = np.ones(self.num_categories, dtype=np.float32) * 1.5
        
        # Track whether a category is currently flagged as overfitted
        self.overfitted_flags = np.zeros(self.num_categories, dtype=bool)
        
        # Compute initial sampling weights (uniform at first)
        self.sampling_probs = np.ones(self.num_categories, dtype=np.float32) / self.num_categories
        
        # Store probability history for logging/debugging
        self.prob_history = []
        self._record_history()

    def update_train_loss(self, cat_idx: int, loss: float):
        """
        Updates the train loss EMA for a given category.
        """
        if 0 <= cat_idx < self.num_categories:
            if np.isnan(loss) or np.isinf(loss):
                return
            self.train_losses[cat_idx] = (
                self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.train_losses[cat_idx]
            )

    def update_val_losses(self, val_losses_dict: Dict[str, float]):
        """
        Updates the validation loss EMA for categories.
        Expects a dictionary: {category_name: val_loss}
        """
        for cat, loss in val_losses_dict.items():
            if cat in self.categories:
                idx = self.categories.index(cat)
                if np.isnan(loss) or np.isinf(loss):
                    continue
                self.val_losses[idx] = (
                    self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.val_losses[idx]
                )

    def detect_overfitting(self):
        """
        Detects category-level overfitting:
        Train loss is low AND Val loss is significantly higher than Train loss.
        """
        for i in range(self.num_categories):
            t_loss = self.train_losses[i]
            v_loss = self.val_losses[i]
            
            # Overfitting condition
            if t_loss < self.overfit_train_threshold and v_loss > t_loss * self.overfit_ratio:
                self.overfitted_flags[i] = True
            else:
                self.overfitted_flags[i] = False

    def recompute_weights(self):
        """
        Recomputes sampling probabilities based on tracked losses.
        If curriculum is disabled, uses a flat uniform distribution.
        """
        if not self.enabled:
            self.sampling_probs = np.ones(self.num_categories, dtype=np.float32) / self.num_categories
            self._record_history()
            return
            
        # First, detect overfitting
        self.detect_overfitting()
        
        # Compute effective losses
        eff_losses = np.copy(self.val_losses)
        for i in range(self.num_categories):
            if self.overfitted_flags[i]:
                # Down-weight overfitted tasks so they aren't oversampled
                eff_losses[i] = self.val_losses[i] * self.overfit_decay
                
        # Numerical stability: subtract max before exponentiating
        # This is standard softmax scaling
        scaled_losses = eff_losses / self.temperature
        scaled_losses = scaled_losses - np.max(scaled_losses)
        exp_losses = np.exp(scaled_losses)
        
        softmax_probs = exp_losses / np.sum(exp_losses)
        
        # Incorporate floor probability
        # P_i = (1 - C * P_floor) * P_softmax_i + P_floor
        adjusted_probs = (1.0 - self.num_categories * self.floor_prob) * softmax_probs + self.floor_prob
        
        # Enforce exact normalization
        self.sampling_probs = adjusted_probs / np.sum(adjusted_probs)
        self._record_history()

    def _record_history(self):
        self.prob_history.append(np.copy(self.sampling_probs))

    def get_probabilities(self) -> np.ndarray:
        """
        Returns the current sampling probabilities for categories.
        """
        return self.sampling_probs

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns key curriculum metrics for logging.
        """
        metrics = {}
        for i, cat in enumerate(self.categories):
            metrics[f"curriculum/prob/{cat}"] = float(self.sampling_probs[i])
            metrics[f"curriculum/train_loss/{cat}"] = float(self.train_losses[i])
            metrics[f"curriculum/val_loss/{cat}"] = float(self.val_losses[i])
            metrics[f"curriculum/overfitted/{cat}"] = int(self.overfitted_flags[i])
        return metrics
