import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TemporalPattern:
    pattern_type: str
    frequency: float
    amplitude: float
    phase: float
    confidence: float
    affected_dimensions: List[str]
    discovery_time: datetime
    last_occurrence: datetime
    occurrence_count: int = 0


@dataclass
class EmotionalTransition:
    from_state: any
    to_state: any
    transition_time: float
    trigger_content: Optional[str]
    transition_smoothness: float
    significance: float


class RecurrentEmotionalPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, (hidden, cell) = self.lstm(x)
        
        attended, attention_weights = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=mask)
        
        normed = self.layer_norm(attended + lstm_out)
        
        predictions = self.projection(normed)
        
        return predictions, attention_weights


class TemporalDynamicsEngine:
    def __init__(
        self,
        emotional_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        sequence_length: int = 50,
        pattern_detection_window: int = 30
    ):
        self.emotional_dim = emotional_dim
        self.sequence_length = sequence_length
        self.pattern_detection_window = pattern_detection_window
        
        self.predictor = RecurrentEmotionalPredictor(
            input_dim=emotional_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=emotional_dim
        )
        
        self.emotional_sequence = deque(maxlen=sequence_length)
        self.transition_history: List[EmotionalTransition] = []
        self.detected_patterns: Dict[str, TemporalPattern] = {}
        
        self.rhythm_analyzers = self._initialize_rhythm_analyzers()
        self.prediction_confidence_history = deque(maxlen=100)
        
    def _initialize_rhythm_analyzers(self) -> Dict[str, Callable]:
        return {
            'circadian': lambda x: self._detect_periodic_pattern(x, 24.0, 6.0),
            'ultradian': lambda x: self._detect_periodic_pattern(x, 4.0, 1.0),
            'weekly': lambda x: self._detect_periodic_pattern(x, 168.0, 24.0),
            'burst': lambda x: self._detect_burst_pattern(x),
            'drift': lambda x: self._detect_drift_pattern(x),
            'oscillation': lambda x: self._detect_oscillation_pattern(x)
        }
    
    def update_sequence(self, emotional_state: any, timestamp: datetime):
        state_vector = emotional_state.to_vector()
        self.emotional_sequence.append({
            'vector': state_vector,
            'state': emotional_state,
            'timestamp': timestamp
        })
        
        if len(self.emotional_sequence) >= 2:
            transition = self._analyze_transition(
                self.emotional_sequence[-2],
                self.emotional_sequence[-1]
            )
            self.transition_history.append(transition)
        
        if len(self.emotional_sequence) >= self.pattern_detection_window:
            self._detect_temporal_patterns()
    
    def _analyze_transition(self, prev_entry: Dict, curr_entry: Dict) -> EmotionalTransition:
        time_delta = (curr_entry['timestamp'] - prev_entry['timestamp']).total_seconds()
        
        prev_vector = prev_entry['vector']
        curr_vector = curr_entry['vector']
        
        transition_magnitude = np.linalg.norm(curr_vector - prev_vector)
        
        velocity = (curr_vector - prev_vector) / (time_delta + 1e-6)
        smoothness = 1.0 / (1.0 + np.std(velocity))
        
        dimension_changes = np.abs(curr_vector - prev_vector)
        significance = np.mean(dimension_changes) * (1.0 + np.std(dimension_changes))
        
        return EmotionalTransition(
            from_state=prev_entry['state'],
            to_state=curr_entry['state'],
            transition_time=time_delta,
            trigger_content=None,
            transition_smoothness=float(smoothness),
            significance=float(significance)
        )
    
    def _detect_temporal_patterns(self):
        if len(self.emotional_sequence) < self.pattern_detection_window:
            return
        
        sequence_array = np.array([entry['vector'] for entry in self.emotional_sequence])
        timestamps = [entry['timestamp'] for entry in self.emotional_sequence]
        
        for dimension_idx in range(self.emotional_dim):
            dimension_series = sequence_array[:, dimension_idx]
            
            for pattern_name, analyzer in self.rhythm_analyzers.items():
                pattern = analyzer(dimension_series)
                
                if pattern is not None:
                    pattern_key = f"{pattern_name}_dim{dimension_idx}"
                    
                    if pattern_key in self.detected_patterns:
                        existing = self.detected_patterns[pattern_key]
                        existing.occurrence_count += 1
                        existing.last_occurrence = datetime.now()
                        existing.confidence = min(1.0, existing.confidence * 0.95 + pattern.confidence * 0.05)
                    else:
                        self.detected_patterns[pattern_key] = pattern
    
    def _detect_periodic_pattern(
        self,
        series: np.ndarray,
        expected_period: float,
        tolerance: float
    ) -> Optional[TemporalPattern]:
        if len(series) < 10:
            return None
        
        detrended = series - savgol_filter(series, min(len(series) // 3 * 2 + 1, 11), 2)
        
        fft = np.fft.fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended))
        power = np.abs(fft) ** 2
        
        positive_freq_idx = frequencies > 0
        frequencies = frequencies[positive_freq_idx]
        power = power[positive_freq_idx]
        
        if len(power) == 0:
            return None
        
        peak_idx = np.argmax(power)
        dominant_frequency = frequencies[peak_idx]
        dominant_period = 1.0 / (dominant_frequency + 1e-10)
        
        if abs(dominant_period - expected_period) > tolerance:
            return None
        
        amplitude = np.std(detrended)
        phase = np.angle(fft[peak_idx])
        
        signal_power = np.sum(power)
        noise_power = signal_power - power[peak_idx]
        snr = power[peak_idx] / (noise_power + 1e-10)
        confidence = min(1.0, snr / 10.0)
        
        if confidence < 0.3:
            return None
        
        return TemporalPattern(
            pattern_type='periodic',
            frequency=float(dominant_frequency),
            amplitude=float(amplitude),
            phase=float(phase),
            confidence=float(confidence),
            affected_dimensions=[],
            discovery_time=datetime.now(),
            last_occurrence=datetime.now()
        )
    
    def _detect_burst_pattern(self, series: np.ndarray) -> Optional[TemporalPattern]:
        if len(series) < 5:
            return None
        
        differences = np.diff(series)
        threshold = np.mean(np.abs(differences)) + 2 * np.std(differences)
        
        bursts = np.abs(differences) > threshold
        burst_count = np.sum(bursts)
        
        if burst_count < 2:
            return None
        
        burst_magnitude = np.mean(np.abs(differences[bursts]))
        burst_frequency = burst_count / len(series)
        
        confidence = min(1.0, burst_frequency * 5.0)
        
        return TemporalPattern(
            pattern_type='burst',
            frequency=float(burst_frequency),
            amplitude=float(burst_magnitude),
            phase=0.0,
            confidence=float(confidence),
            affected_dimensions=[],
            discovery_time=datetime.now(),
            last_occurrence=datetime.now()
        )
    
    def _detect_drift_pattern(self, series: np.ndarray) -> Optional[TemporalPattern]:
        if len(series) < 10:
            return None
        
        x = np.arange(len(series))
        coefficients = np.polyfit(x, series, 1)
        slope = coefficients[0]
        
        if abs(slope) < 0.01:
            return None
        
        fitted = np.polyval(coefficients, x)
        residuals = series - fitted
        r_squared = 1 - (np.sum(residuals**2) / np.sum((series - np.mean(series))**2))
        
        confidence = min(1.0, r_squared)
        
        if confidence < 0.5:
            return None
        
        return TemporalPattern(
            pattern_type='drift',
            frequency=0.0,
            amplitude=float(abs(slope)),
            phase=0.0,
            confidence=float(confidence),
            affected_dimensions=[],
            discovery_time=datetime.now(),
            last_occurrence=datetime.now()
        )
    
    def _detect_oscillation_pattern(self, series: np.ndarray) -> Optional[TemporalPattern]:
        if len(series) < 8:
            return None
        
        peaks, _ = find_peaks(series)
        troughs, _ = find_peaks(-series)
        
        total_extrema = len(peaks) + len(troughs)
        
        if total_extrema < 3:
            return None
        
        oscillation_frequency = total_extrema / len(series)
        
        if oscillation_frequency < 0.15:
            return None
        
        amplitude = np.std(series)
        regularity = 1.0 / (1.0 + np.std(np.diff(peaks)) + np.std(np.diff(troughs)) + 1e-6)
        
        confidence = min(1.0, oscillation_frequency * regularity * 2.0)
        
        return TemporalPattern(
            pattern_type='oscillation',
            frequency=float(oscillation_frequency),
            amplitude=float(amplitude),
            phase=0.0,
            confidence=float(confidence),
            affected_dimensions=[],
            discovery_time=datetime.now(),
            last_occurrence=datetime.now()
        )
    
    def predict_future_state(
        self,
        steps_ahead: int = 1,
        context: Optional[Dict[str, any]] = None
    ) -> Tuple[any, float]:
        if len(self.emotional_sequence) < 10:
            return None, 0.0
        
        sequence_array = np.array([entry['vector'] for entry in self.emotional_sequence])
        
        with torch.no_grad():
            x = torch.FloatTensor(sequence_array).unsqueeze(0)
            predictions, attention_weights = self.predictor(x)
            
            if steps_ahead == 1:
                predicted_vector = predictions[0, -1, :].numpy()
            else:
                current_sequence = sequence_array.copy()
                for step in range(steps_ahead):
                    x = torch.FloatTensor(current_sequence[-self.sequence_length:]).unsqueeze(0)
                    pred, _ = self.predictor(x)
                    next_state = pred[0, -1, :].numpy()
                    current_sequence = np.vstack([current_sequence, next_state])
                
                predicted_vector = current_sequence[-1]
        
        pattern_adjustment = self._apply_pattern_predictions(predicted_vector, steps_ahead)
        
        final_prediction = predicted_vector * 0.7 + pattern_adjustment * 0.3
        
        confidence = self._compute_prediction_confidence(predicted_vector, attention_weights)
        
        predicted_state = self._vector_to_emotional_state(final_prediction)
        
        return predicted_state, confidence
    
    def _apply_pattern_predictions(self, base_prediction: np.ndarray, steps_ahead: int) -> np.ndarray:
        adjustment = base_prediction.copy()
        
        for pattern_key, pattern in self.detected_patterns.items():
            if pattern.confidence < 0.5:
                continue
            
            dimension_idx = int(pattern_key.split('dim')[1])
            
            if pattern.pattern_type == 'periodic':
                phase_shift = pattern.frequency * steps_ahead * 2 * np.pi
                adjustment[dimension_idx] += pattern.amplitude * np.sin(pattern.phase + phase_shift) * 0.1
            
            elif pattern.pattern_type == 'drift':
                adjustment[dimension_idx] += pattern.amplitude * steps_ahead * 0.05
            
            elif pattern.pattern_type == 'oscillation':
                oscillation_phase = steps_ahead * pattern.frequency * 2 * np.pi
                adjustment[dimension_idx] += pattern.amplitude * np.sin(oscillation_phase) * 0.08
        
        return np.clip(adjustment, 0.0, 1.0)
    
    def _compute_prediction_confidence(
        self,
        prediction: np.ndarray,
        attention_weights: torch.Tensor
    ) -> float:
        if len(self.emotional_sequence) < 5:
            return 0.3
        
        recent_vectors = np.array([entry['vector'] for entry in list(self.emotional_sequence)[-5:]])
        variance = np.mean(np.var(recent_vectors, axis=0))
        stability_score = 1.0 / (1.0 + variance)
        
        attention_entropy = entropy(attention_weights[0, -1, :].numpy() + 1e-10)
        max_entropy = np.log(attention_weights.shape[-1])
        attention_focus = 1.0 - (attention_entropy / max_entropy)
        
        pattern_support = sum(
            pattern.confidence for pattern in self.detected_patterns.values()
            if pattern.confidence > 0.5
        ) / max(len(self.detected_patterns), 1)
        
        confidence = (
            stability_score * 0.4 +
            attention_focus * 0.3 +
            min(1.0, pattern_support) * 0.3
        )
        
        self.prediction_confidence_history.append(confidence)
        
        return float(confidence)
    
    def _vector_to_emotional_state(self, vector: np.ndarray) -> any:
        from core_engine import EmotionalState
        
        return EmotionalState(
            valence=float(vector[0]),
            arousal=float(vector[1]),
            dominance=float(vector[2]),
            intimacy=float(vector[3]),
            cognitive_load=float(vector[4]),
            temporal_urgency=float(vector[5]),
            social_orientation=float(vector[6]),
            existential_depth=float(vector[7]),
            aesthetic_resonance=float(vector[8]),
            semantic_coherence=float(vector[9])
        )
    
    def analyze_emotional_trajectory(self) -> Dict[str, any]:
        if len(self.emotional_sequence) < 5:
            return {}
        
        vectors = np.array([entry['vector'] for entry in self.emotional_sequence])
        
        complexity = self._compute_trajectory_complexity(vectors)
        volatility = self._compute_emotional_volatility(vectors)
        directionality = self._compute_directionality(vectors)
        cyclicity = self._compute_cyclicity(vectors)
        
        dominant_patterns = sorted(
            self.detected_patterns.values(),
            key=lambda p: p.confidence,
            reverse=True
        )[:3]
        
        transition_characteristics = self._analyze_transitions()
        
        return {
            'complexity': complexity,
            'volatility': volatility,
            'directionality': directionality,
            'cyclicity': cyclicity,
            'dominant_patterns': [
                {
                    'type': p.pattern_type,
                    'confidence': p.confidence,
                    'frequency': p.frequency,
                    'amplitude': p.amplitude
                }
                for p in dominant_patterns
            ],
            'transition_characteristics': transition_characteristics,
            'sequence_length': len(self.emotional_sequence),
            'prediction_reliability': float(np.mean(self.prediction_confidence_history)) if self.prediction_confidence_history else 0.0
        }
    
    def _compute_trajectory_complexity(self, vectors: np.ndarray) -> float:
        if len(vectors) < 3:
            return 0.0
        
        directions = np.diff(vectors, axis=0)
        direction_changes = np.sum(
            np.abs(np.diff(np.sign(directions), axis=0)) > 0
        ) / (len(vectors) - 2)
        
        path_length = np.sum(np.linalg.norm(directions, axis=1))
        direct_distance = np.linalg.norm(vectors[-1] - vectors[0])
        tortuosity = path_length / (direct_distance + 1e-6)
        
        dimension_entropy = entropy(np.var(vectors, axis=0) + 1e-10)
        
        complexity = (
            direction_changes * 0.4 +
            min(1.0, tortuosity / 5.0) * 0.4 +
            min(1.0, dimension_entropy / 3.0) * 0.2
        )
        
        return float(complexity)
    
    def _compute_emotional_volatility(self, vectors: np.ndarray) -> float:
        if len(vectors) < 2:
            return 0.0
        
        changes = np.diff(vectors, axis=0)
        magnitudes = np.linalg.norm(changes, axis=1)
        
        mean_change = np.mean(magnitudes)
        std_change = np.std(magnitudes)
        
        volatility = mean_change + std_change
        
        return float(min(1.0, volatility))
    
    def _compute_directionality(self, vectors: np.ndarray) -> Dict[str, float]:
        if len(vectors) < 2:
            return {}
        
        overall_direction = vectors[-1] - vectors[0]
        
        dimension_trends = {}
        dimension_names = [
            'valence', 'arousal', 'dominance', 'intimacy',
            'cognitive_load', 'temporal_urgency', 'social_orientation',
            'existential_depth', 'aesthetic_resonance', 'semantic_coherence'
        ]
        
        for i, name in enumerate(dimension_names):
            trend = overall_direction[i]
            if abs(trend) > 0.1:
                dimension_trends[name] = 'increasing' if trend > 0 else 'decreasing'
        
        return dimension_trends
    
    def _compute_cyclicity(self, vectors: np.ndarray) -> float:
        if len(vectors) < 10:
            return 0.0
        
        cyclicity_scores = []
        
        for dim in range(vectors.shape[1]):
            series = vectors[:, dim]
            autocorr = np.correlate(series - np.mean(series), series - np.mean(series), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            if len(autocorr) > 1:
                peaks, _ = find_peaks(autocorr[1:])
                if len(peaks) > 0:
                    cyclicity_scores.append(np.mean(autocorr[peaks]))
        
        return float(np.mean(cyclicity_scores)) if cyclicity_scores else 0.0
    
    def _analyze_transitions(self) -> Dict[str, any]:
        if not self.transition_history:
            return {}
        
        recent_transitions = self.transition_history[-20:]
        
        avg_transition_time = np.mean([t.transition_time for t in recent_transitions])
        avg_smoothness = np.mean([t.transition_smoothness for t in recent_transitions])
        avg_significance = np.mean([t.significance for t in recent_transitions])
        
        rapid_transitions = sum(1 for t in recent_transitions if t.transition_time < 60)
        significant_transitions = sum(1 for t in recent_transitions if t.significance > 0.5)
        
        return {
            'average_transition_time': float(avg_transition_time),
            'average_smoothness': float(avg_smoothness),
            'average_significance': float(avg_significance),
            'rapid_transition_ratio': rapid_transitions / len(recent_transitions),
            'significant_transition_ratio': significant_transitions / len(recent_transitions)
        }
