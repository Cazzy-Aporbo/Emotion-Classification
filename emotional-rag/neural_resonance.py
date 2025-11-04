import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class EmpathyProfile:
    cognitive_empathy: float
    affective_empathy: float
    compassionate_response: float
    perspective_taking: float
    emotional_contagion_susceptibility: float
    resonance_depth: float


@dataclass
class ResonanceSignature:
    primary_emotion: str
    intensity: float
    authenticity_score: float
    complexity_index: float
    harmonic_components: List[Tuple[str, float]]
    interference_patterns: List[str]


class MultiHeadEmotionalAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, emotional_bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        if emotional_bias:
            self.emotional_bias = nn.Parameter(torch.randn(num_heads, 1, 1))
        else:
            self.emotional_bias = None
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        emotional_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if self.emotional_bias is not None:
            scores = scores + self.emotional_bias
        
        if emotional_context is not None:
            emotional_modulation = self._compute_emotional_modulation(
                emotional_context, scores.shape
            )
            scores = scores * emotional_modulation
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        output = self.out_proj(attended)
        
        return output, attention_weights.mean(dim=1)
    
    def _compute_emotional_modulation(
        self,
        emotional_context: torch.Tensor,
        target_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, _ = target_shape
        
        modulation = torch.sigmoid(emotional_context.mean(dim=-1, keepdim=True))
        modulation = modulation.unsqueeze(1).unsqueeze(1)
        modulation = modulation.expand(batch_size, num_heads, seq_len, seq_len)
        
        return 0.5 + modulation * 0.5


class EmpathyEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.cognitive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.affective_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.compassion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.input_projection(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        pooled = x.mean(dim=1)
        
        cognitive = self.cognitive_head(pooled)
        affective = self.affective_head(pooled)
        compassion = self.compassion_head(pooled)
        
        return pooled, {
            'cognitive_empathy': cognitive,
            'affective_empathy': affective,
            'compassionate_response': compassion
        }


class ResonanceNetwork(nn.Module):
    def __init__(self, emotional_dim: int, hidden_dim: int):
        super().__init__()
        
        self.emotional_encoder = nn.Sequential(
            nn.Linear(emotional_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.resonance_attention = MultiHeadEmotionalAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            emotional_bias=True
        )
        
        self.harmonic_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, emotional_dim),
            nn.Softmax(dim=-1)
        )
        
        self.authenticity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        source_emotion: torch.Tensor,
        context_emotions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        source_encoded = self.emotional_encoder(source_emotion.unsqueeze(0))
        context_encoded = self.emotional_encoder(context_emotions)
        
        resonant_features, attention_weights = self.resonance_attention(
            source_encoded,
            context_encoded,
            context_encoded
        )
        
        harmonic_components = self.harmonic_analyzer(resonant_features)
        authenticity = self.authenticity_predictor(resonant_features)
        
        return {
            'resonant_features': resonant_features.squeeze(0),
            'harmonic_components': harmonic_components.squeeze(0),
            'authenticity_score': authenticity.squeeze(),
            'attention_weights': attention_weights.squeeze(0)
        }


class NeuralResonanceEngine:
    def __init__(
        self,
        emotional_dim: int = 10,
        hidden_dim: int = 256,
        empathy_layers: int = 4
    ):
        self.emotional_dim = emotional_dim
        self.hidden_dim = hidden_dim
        
        self.empathy_encoder = EmpathyEncoder(
            input_dim=emotional_dim * 2,
            hidden_dim=hidden_dim,
            num_layers=empathy_layers
        )
        
        self.resonance_network = ResonanceNetwork(
            emotional_dim=emotional_dim,
            hidden_dim=hidden_dim
        )
        
        self.empathy_profiles: Dict[str, EmpathyProfile] = {}
        self.resonance_history: List[ResonanceSignature] = []
        
        self.emotional_vocabulary = self._initialize_emotional_vocabulary()
        self.contagion_model = self._initialize_contagion_model()
        
    def _initialize_emotional_vocabulary(self) -> Dict[str, np.ndarray]:
        base_emotions = {
            'joy': np.array([0.9, 0.7, 0.7, 0.6, 0.3, 0.4, 0.7, 0.3, 0.8, 0.8]),
            'sadness': np.array([0.2, 0.3, 0.3, 0.5, 0.6, 0.3, 0.4, 0.6, 0.4, 0.5]),
            'anger': np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.8, 0.5, 0.4, 0.2, 0.4]),
            'fear': np.array([0.2, 0.8, 0.2, 0.4, 0.8, 0.9, 0.3, 0.7, 0.3, 0.4]),
            'surprise': np.array([0.6, 0.8, 0.5, 0.4, 0.7, 0.7, 0.5, 0.5, 0.6, 0.5]),
            'disgust': np.array([0.2, 0.6, 0.6, 0.2, 0.5, 0.5, 0.3, 0.4, 0.2, 0.4]),
            'trust': np.array([0.7, 0.4, 0.6, 0.8, 0.4, 0.3, 0.8, 0.5, 0.6, 0.7]),
            'anticipation': np.array([0.6, 0.7, 0.6, 0.5, 0.6, 0.7, 0.6, 0.5, 0.5, 0.6]),
            'serenity': np.array([0.7, 0.2, 0.6, 0.7, 0.2, 0.2, 0.6, 0.5, 0.7, 0.8]),
            'anxiety': np.array([0.3, 0.8, 0.3, 0.4, 0.8, 0.8, 0.4, 0.7, 0.3, 0.4]),
            'contentment': np.array([0.8, 0.3, 0.6, 0.7, 0.3, 0.2, 0.6, 0.4, 0.7, 0.8]),
            'frustration': np.array([0.3, 0.7, 0.5, 0.3, 0.7, 0.6, 0.4, 0.5, 0.3, 0.4]),
            'curiosity': np.array([0.6, 0.6, 0.5, 0.5, 0.7, 0.5, 0.5, 0.6, 0.6, 0.7]),
            'pride': np.array([0.8, 0.6, 0.8, 0.5, 0.4, 0.4, 0.6, 0.4, 0.7, 0.7]),
            'shame': np.array([0.2, 0.5, 0.2, 0.4, 0.7, 0.4, 0.3, 0.7, 0.3, 0.4]),
            'love': np.array([0.9, 0.6, 0.6, 0.9, 0.4, 0.3, 0.9, 0.5, 0.9, 0.8]),
            'loneliness': np.array([0.2, 0.4, 0.3, 0.2, 0.6, 0.5, 0.2, 0.7, 0.4, 0.5]),
            'excitement': np.array([0.9, 0.9, 0.7, 0.6, 0.5, 0.8, 0.7, 0.4, 0.7, 0.6])
        }
        return base_emotions
    
    def _initialize_contagion_model(self) -> Dict[str, float]:
        return {
            'baseline_susceptibility': 0.5,
            'intensity_amplification': 1.2,
            'decay_rate': 0.15,
            'transmission_threshold': 0.3
        }
    
    def compute_empathy_profile(
        self,
        observer_state: any,
        target_state: any,
        interaction_history: List[Dict[str, any]]
    ) -> EmpathyProfile:
        observer_vector = observer_state.to_vector()
        target_vector = target_state.to_vector()
        
        combined_input = np.concatenate([observer_vector, target_vector])
        
        with torch.no_grad():
            x = torch.FloatTensor(combined_input).unsqueeze(0).unsqueeze(0)
            _, empathy_components = self.empathy_encoder(x)
        
        cognitive_empathy = float(empathy_components['cognitive_empathy'].item())
        affective_empathy = float(empathy_components['affective_empathy'].item())
        compassionate_response = float(empathy_components['compassionate_response'].item())
        
        perspective_taking = self._compute_perspective_taking(
            observer_state, target_state
        )
        
        contagion_susceptibility = self._compute_contagion_susceptibility(
            observer_state, target_state, interaction_history
        )
        
        resonance_depth = self._compute_resonance_depth(
            observer_vector, target_vector
        )
        
        profile = EmpathyProfile(
            cognitive_empathy=cognitive_empathy,
            affective_empathy=affective_empathy,
            compassionate_response=compassionate_response,
            perspective_taking=perspective_taking,
            emotional_contagion_susceptibility=contagion_susceptibility,
            resonance_depth=resonance_depth
        )
        
        profile_key = f"{id(observer_state)}_{id(target_state)}"
        self.empathy_profiles[profile_key] = profile
        
        return profile
    
    def _compute_perspective_taking(self, observer: any, target: any) -> float:
        observer_vec = observer.to_vector()
        target_vec = target.to_vector()
        
        distance = np.linalg.norm(observer_vec - target_vec)
        similarity = 1.0 / (1.0 + distance)
        
        cognitive_alignment = 1.0 - abs(observer.cognitive_load - target.cognitive_load)
        social_alignment = 1.0 - abs(observer.social_orientation - target.social_orientation)
        
        perspective_capacity = (
            similarity * 0.4 +
            cognitive_alignment * 0.3 +
            social_alignment * 0.3
        )
        
        return float(perspective_capacity)
    
    def _compute_contagion_susceptibility(
        self,
        observer: any,
        target: any,
        history: List[Dict[str, any]]
    ) -> float:
        base_susceptibility = self.contagion_model['baseline_susceptibility']
        
        intimacy_factor = (observer.intimacy + target.intimacy) / 2
        
        arousal_differential = abs(observer.arousal - target.arousal)
        arousal_factor = 1.0 - arousal_differential
        
        if history:
            recent_interactions = min(len(history), 10)
            interaction_factor = recent_interactions / 10.0
        else:
            interaction_factor = 0.3
        
        emotional_openness = 1.0 - observer.dominance * 0.3
        
        susceptibility = (
            base_susceptibility * 0.3 +
            intimacy_factor * 0.25 +
            arousal_factor * 0.2 +
            interaction_factor * 0.15 +
            emotional_openness * 0.1
        )
        
        return float(np.clip(susceptibility, 0.0, 1.0))
    
    def _compute_resonance_depth(
        self,
        observer_vector: np.ndarray,
        target_vector: np.ndarray
    ) -> float:
        dimension_weights = np.array([1.2, 1.0, 0.8, 1.3, 0.7, 0.6, 1.1, 0.9, 1.0, 0.8])
        
        weighted_distance = np.sqrt(
            np.sum(dimension_weights * (observer_vector - target_vector) ** 2)
        )
        
        resonance = 1.0 / (1.0 + weighted_distance)
        
        harmonic_alignment = np.dot(observer_vector, target_vector) / (
            np.linalg.norm(observer_vector) * np.linalg.norm(target_vector) + 1e-8
        )
        
        depth = resonance * 0.7 + (harmonic_alignment + 1) / 2 * 0.3
        
        return float(depth)
    
    def analyze_emotional_resonance(
        self,
        source_state: any,
        context_states: List[any]
    ) -> ResonanceSignature:
        source_vector = torch.FloatTensor(source_state.to_vector())
        context_vectors = torch.FloatTensor(
            np.array([state.to_vector() for state in context_states])
        )
        
        with torch.no_grad():
            resonance_output = self.resonance_network(source_vector, context_vectors)
        
        harmonic_components = resonance_output['harmonic_components'].numpy()
        top_harmonics = np.argsort(harmonic_components)[-3:][::-1]
        
        primary_emotion = self._identify_primary_emotion(source_vector.numpy())
        
        intensity = float(source_state.arousal * 0.6 + source_state.valence * 0.4)
        
        authenticity_score = float(resonance_output['authenticity_score'].item())
        
        complexity_index = self._compute_emotional_complexity(
            harmonic_components, resonance_output['attention_weights'].numpy()
        )
        
        emotion_names = list(self.emotional_vocabulary.keys())
        harmonic_list = [
            (emotion_names[idx], float(harmonic_components[idx]))
            for idx in top_harmonics
        ]
        
        interference_patterns = self._detect_interference_patterns(
            source_vector.numpy(), context_vectors.numpy()
        )
        
        signature = ResonanceSignature(
            primary_emotion=primary_emotion,
            intensity=intensity,
            authenticity_score=authenticity_score,
            complexity_index=complexity_index,
            harmonic_components=harmonic_list,
            interference_patterns=interference_patterns
        )
        
        self.resonance_history.append(signature)
        
        return signature
    
    def _identify_primary_emotion(self, emotional_vector: np.ndarray) -> str:
        similarities = {}
        
        for emotion_name, emotion_vector in self.emotional_vocabulary.items():
            similarity = np.dot(emotional_vector, emotion_vector) / (
                np.linalg.norm(emotional_vector) * np.linalg.norm(emotion_vector) + 1e-8
            )
            similarities[emotion_name] = similarity
        
        return max(similarities, key=similarities.get)
    
    def _compute_emotional_complexity(
        self,
        harmonics: np.ndarray,
        attention_weights: np.ndarray
    ) -> float:
        harmonic_entropy = -np.sum(harmonics * np.log(harmonics + 1e-10))
        max_entropy = np.log(len(harmonics))
        normalized_entropy = harmonic_entropy / max_entropy
        
        attention_distribution = attention_weights / (attention_weights.sum() + 1e-10)
        attention_entropy = -np.sum(
            attention_distribution * np.log(attention_distribution + 1e-10)
        )
        max_attention_entropy = np.log(len(attention_distribution))
        normalized_attention = attention_entropy / max_attention_entropy
        
        top_k = 5
        top_harmonics = np.sort(harmonics)[-top_k:]
        diversity = len(np.unique(np.round(top_harmonics, 2))) / top_k
        
        complexity = (
            normalized_entropy * 0.4 +
            normalized_attention * 0.3 +
            diversity * 0.3
        )
        
        return float(complexity)
    
    def _detect_interference_patterns(
        self,
        source: np.ndarray,
        context: np.ndarray
    ) -> List[str]:
        patterns = []
        
        context_mean = np.mean(context, axis=0)
        
        if np.linalg.norm(source - context_mean) > 1.5:
            patterns.append('dissonance')
        
        context_variance = np.var(context, axis=0)
        if np.mean(context_variance) > 0.3:
            patterns.append('emotional_turbulence')
        
        valence_conflict = (source[0] > 0.6 and context_mean[0] < 0.4) or \
                          (source[0] < 0.4 and context_mean[0] > 0.6)
        if valence_conflict:
            patterns.append('valence_conflict')
        
        arousal_mismatch = abs(source[1] - context_mean[1]) > 0.5
        if arousal_mismatch:
            patterns.append('arousal_mismatch')
        
        if len(patterns) == 0:
            patterns.append('harmonic_alignment')
        
        return patterns
    
    def simulate_emotional_contagion(
        self,
        source_state: any,
        recipient_state: any,
        empathy_profile: EmpathyProfile,
        duration: float = 1.0
    ) -> any:
        from core_engine import EmotionalState
        
        source_vec = source_state.to_vector()
        recipient_vec = recipient_state.to_vector()
        
        contagion_strength = (
            empathy_profile.affective_empathy * 0.4 +
            empathy_profile.emotional_contagion_susceptibility * 0.35 +
            empathy_profile.resonance_depth * 0.25
        )
        
        intensity_multiplier = self.contagion_model['intensity_amplification']
        transmission_threshold = self.contagion_model['transmission_threshold']
        
        if contagion_strength < transmission_threshold:
            contagion_strength *= 0.5
        
        time_decay = np.exp(-self.contagion_model['decay_rate'] * duration)
        effective_strength = contagion_strength * time_decay * intensity_multiplier
        
        valence_transmission = 0.8
        arousal_transmission = 0.9
        dominance_transmission = 0.4
        
        transmission_weights = np.array([
            valence_transmission, arousal_transmission, dominance_transmission,
            0.6, 0.5, 0.7, 0.6, 0.5, 0.6, 0.5
        ])
        
        influence = (source_vec - recipient_vec) * transmission_weights * effective_strength
        
        new_vec = recipient_vec + influence
        new_vec = np.clip(new_vec, 0.0, 1.0)
        
        return EmotionalState(
            valence=float(new_vec[0]),
            arousal=float(new_vec[1]),
            dominance=float(new_vec[2]),
            intimacy=float(new_vec[3]),
            cognitive_load=float(new_vec[4]),
            temporal_urgency=float(new_vec[5]),
            social_orientation=float(new_vec[6]),
            existential_depth=float(new_vec[7]),
            aesthetic_resonance=float(new_vec[8]),
            semantic_coherence=float(new_vec[9])
        )
    
    def get_empathy_analytics(self) -> Dict[str, any]:
        if not self.empathy_profiles:
            return {}
        
        profiles = list(self.empathy_profiles.values())
        
        return {
            'average_cognitive_empathy': float(np.mean([p.cognitive_empathy for p in profiles])),
            'average_affective_empathy': float(np.mean([p.affective_empathy for p in profiles])),
            'average_compassion': float(np.mean([p.compassionate_response for p in profiles])),
            'average_resonance_depth': float(np.mean([p.resonance_depth for p in profiles])),
            'total_profiles': len(profiles),
            'high_empathy_interactions': sum(1 for p in profiles if p.cognitive_empathy > 0.7),
            'resonance_signatures': len(self.resonance_history),
            'complexity_distribution': self._analyze_complexity_distribution()
        }
    
    def _analyze_complexity_distribution(self) -> Dict[str, int]:
        if not self.resonance_history:
            return {}
        
        complexities = [sig.complexity_index for sig in self.resonance_history]
        
        return {
            'low': sum(1 for c in complexities if c < 0.33),
            'medium': sum(1 for c in complexities if 0.33 <= c < 0.67),
            'high': sum(1 for c in complexities if c >= 0.67)
        }
