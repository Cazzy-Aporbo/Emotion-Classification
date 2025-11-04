import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path
import pickle
from datetime import datetime, timedelta


@dataclass
class EmotionalState:
    valence: float
    arousal: float
    dominance: float
    intimacy: float
    cognitive_load: float
    temporal_urgency: float
    social_orientation: float
    existential_depth: float
    aesthetic_resonance: float
    semantic_coherence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.valence, self.arousal, self.dominance, self.intimacy,
            self.cognitive_load, self.temporal_urgency, self.social_orientation,
            self.existential_depth, self.aesthetic_resonance, self.semantic_coherence
        ])
    
    def distance_to(self, other: 'EmotionalState') -> float:
        v1, v2 = self.to_vector(), other.to_vector()
        weights = np.array([1.2, 1.0, 0.9, 1.1, 0.8, 0.7, 0.85, 0.95, 1.05, 0.9])
        return np.sqrt(np.sum(weights * (v1 - v2) ** 2))


@dataclass
class EmotionalMemory:
    content: str
    embedding: np.ndarray
    emotional_state: EmotionalState
    context_window: List[str]
    interaction_count: int = 0
    resonance_score: float = 0.0
    decay_factor: float = 1.0
    associations: List[str] = field(default_factory=list)
    emotional_trajectory: List[Tuple[datetime, EmotionalState]] = field(default_factory=list)


class DimensionalEmbedder(nn.Module):
    def __init__(self, input_dim: int, emotional_dim: int, hidden_layers: List[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.15)
            ])
            prev_dim = hidden_dim
        
        self.feature_network = nn.Sequential(*layers)
        self.emotional_projection = nn.Linear(prev_dim, emotional_dim)
        self.semantic_projection = nn.Linear(prev_dim, emotional_dim)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=emotional_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(emotional_dim * 2, emotional_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, emotional_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_network(x)
        emotional_emb = self.emotional_projection(features)
        semantic_emb = self.semantic_projection(features)
        
        emotional_emb_expanded = emotional_emb.unsqueeze(0)
        emotional_context_expanded = emotional_context.unsqueeze(0)
        
        attended_emotional, _ = self.cross_attention(
            emotional_emb_expanded,
            emotional_context_expanded,
            emotional_context_expanded
        )
        
        attended_emotional = attended_emotional.squeeze(0)
        
        fusion_input = torch.cat([emotional_emb, attended_emotional], dim=-1)
        gate = self.fusion_gate(fusion_input)
        
        fused_emotional = gate * emotional_emb + (1 - gate) * attended_emotional
        
        return fused_emotional, semantic_emb


class EmotionalRAGCore:
    def __init__(
        self,
        embedding_dim: int = 768,
        emotional_dim: int = 256,
        hidden_layers: List[int] = [512, 384, 256],
        memory_capacity: int = 10000,
        temporal_window: int = 100
    ):
        self.embedding_dim = embedding_dim
        self.emotional_dim = emotional_dim
        self.memory_capacity = memory_capacity
        self.temporal_window = temporal_window
        
        self.embedder = DimensionalEmbedder(
            input_dim=embedding_dim,
            emotional_dim=emotional_dim,
            hidden_layers=hidden_layers
        )
        
        self.memories: List[EmotionalMemory] = []
        self.emotional_history: List[EmotionalState] = []
        self.interaction_graph = defaultdict(list)
        self.resonance_cache = {}
        
        self.current_emotional_state = EmotionalState(
            valence=0.5, arousal=0.5, dominance=0.5, intimacy=0.3,
            cognitive_load=0.4, temporal_urgency=0.3, social_orientation=0.5,
            existential_depth=0.4, aesthetic_resonance=0.5, semantic_coherence=0.6
        )
        
    def encode_emotional_context(self, text: str, base_embedding: np.ndarray) -> Tuple[np.ndarray, EmotionalState]:
        emotional_markers = self._extract_emotional_markers(text)
        emotional_state = self._compute_emotional_state(text, emotional_markers)
        
        with torch.no_grad():
            x = torch.FloatTensor(base_embedding).unsqueeze(0)
            emotional_ctx = torch.FloatTensor(emotional_state.to_vector()).unsqueeze(0)
            emotional_emb, semantic_emb = self.embedder(x, emotional_ctx)
            
            combined = torch.cat([emotional_emb, semantic_emb], dim=-1)
            return combined.squeeze(0).numpy(), emotional_state
    
    def _extract_emotional_markers(self, text: str) -> Dict[str, float]:
        markers = {
            'intensity': len([w for w in text.split() if w.isupper()]) / max(len(text.split()), 1),
            'question_depth': text.count('?') * 0.1,
            'pause_density': text.count('...') * 0.15,
            'exclamation': text.count('!') * 0.12,
            'personal_pronouns': len([w for w in text.lower().split() if w in ['i', 'me', 'my', 'mine', 'we', 'us', 'our']]) / max(len(text.split()), 1),
            'negation': len([w for w in text.lower().split() if w in ['not', 'no', 'never', 'nothing', 'none']]) / max(len(text.split()), 1),
            'certainty': len([w for w in text.lower().split() if w in ['always', 'definitely', 'certainly', 'absolutely']]) / max(len(text.split()), 1),
            'hedging': len([w for w in text.lower().split() if w in ['maybe', 'perhaps', 'possibly', 'might', 'could']]) / max(len(text.split()), 1)
        }
        return markers
    
    def _compute_emotional_state(self, text: str, markers: Dict[str, float]) -> EmotionalState:
        text_lower = text.lower()
        
        positive_words = ['love', 'joy', 'happy', 'wonderful', 'amazing', 'great', 'brilliant', 'excited', 'beautiful']
        negative_words = ['hate', 'sad', 'terrible', 'awful', 'horrible', 'bad', 'angry', 'frustrated', 'disappointed']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        valence = 0.5 + (positive_count - negative_count) * 0.1
        valence = np.clip(valence, 0.0, 1.0)
        
        arousal = 0.5 + markers['intensity'] * 0.3 + markers['exclamation'] * 0.2
        arousal = np.clip(arousal, 0.0, 1.0)
        
        dominance = 0.5 + markers['certainty'] * 0.3 - markers['hedging'] * 0.2
        dominance = np.clip(dominance, 0.0, 1.0)
        
        intimacy = markers['personal_pronouns'] * 0.8 + 0.2
        intimacy = np.clip(intimacy, 0.0, 1.0)
        
        cognitive_load = len(text.split()) / 100.0 + markers['question_depth']
        cognitive_load = np.clip(cognitive_load, 0.0, 1.0)
        
        temporal_urgency = markers['exclamation'] * 0.4 + (1.0 - markers['pause_density'])
        temporal_urgency = np.clip(temporal_urgency, 0.0, 1.0)
        
        social_orientation = markers['personal_pronouns'] * 0.7
        social_orientation = np.clip(social_orientation, 0.0, 1.0)
        
        existential_keywords = ['why', 'meaning', 'purpose', 'existence', 'life', 'death', 'forever', 'never']
        existential_depth = sum(1 for word in existential_keywords if word in text_lower) * 0.15
        existential_depth = np.clip(existential_depth, 0.0, 1.0)
        
        aesthetic_keywords = ['beautiful', 'elegant', 'graceful', 'stunning', 'artistic', 'creative']
        aesthetic_resonance = sum(1 for word in aesthetic_keywords if word in text_lower) * 0.2
        aesthetic_resonance = np.clip(aesthetic_resonance, 0.0, 1.0)
        
        semantic_coherence = 1.0 - markers['negation'] * 0.3
        semantic_coherence = np.clip(semantic_coherence, 0.0, 1.0)
        
        return EmotionalState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            intimacy=intimacy,
            cognitive_load=cognitive_load,
            temporal_urgency=temporal_urgency,
            social_orientation=social_orientation,
            existential_depth=existential_depth,
            aesthetic_resonance=aesthetic_resonance,
            semantic_coherence=semantic_coherence
        )
    
    def store_memory(
        self,
        content: str,
        base_embedding: np.ndarray,
        context_window: List[str]
    ) -> EmotionalMemory:
        enhanced_embedding, emotional_state = self.encode_emotional_context(content, base_embedding)
        
        memory = EmotionalMemory(
            content=content,
            embedding=enhanced_embedding,
            emotional_state=emotional_state,
            context_window=context_window,
            emotional_trajectory=[(datetime.now(), emotional_state)]
        )
        
        if len(self.memories) >= self.memory_capacity:
            self._consolidate_memories()
        
        self.memories.append(memory)
        self.emotional_history.append(emotional_state)
        self._update_resonance_cache(memory)
        
        return memory
    
    def _consolidate_memories(self):
        scores = []
        for mem in self.memories:
            age_factor = (datetime.now() - mem.emotional_trajectory[0][0]).total_seconds() / (24 * 3600)
            score = (mem.resonance_score * mem.decay_factor * 
                    (1.0 / (1.0 + age_factor * 0.1)) * 
                    (1.0 + mem.interaction_count * 0.05))
            scores.append(score)
        
        sorted_indices = np.argsort(scores)[::-1]
        keep_count = int(self.memory_capacity * 0.85)
        
        self.memories = [self.memories[i] for i in sorted_indices[:keep_count]]
    
    def _update_resonance_cache(self, memory: EmotionalMemory):
        cache_key = hash(memory.content[:100])
        self.resonance_cache[cache_key] = {
            'emotional_state': memory.emotional_state,
            'timestamp': datetime.now(),
            'access_count': 1
        }
        
        if len(self.resonance_cache) > 1000:
            sorted_items = sorted(
                self.resonance_cache.items(),
                key=lambda x: (x[1]['access_count'], x[1]['timestamp']),
                reverse=True
            )
            self.resonance_cache = dict(sorted_items[:800])
    
    def retrieve_emotionally_relevant(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        emotional_weight: float = 0.6
    ) -> List[Tuple[EmotionalMemory, float]]:
        query_enhanced, query_emotional_state = self.encode_emotional_context(query, query_embedding)
        
        candidates = []
        for memory in self.memories:
            semantic_similarity = self._cosine_similarity(query_enhanced, memory.embedding)
            emotional_distance = query_emotional_state.distance_to(memory.emotional_state)
            emotional_similarity = 1.0 / (1.0 + emotional_distance)
            
            temporal_factor = self._compute_temporal_relevance(memory)
            resonance_factor = memory.resonance_score
            
            combined_score = (
                semantic_similarity * (1 - emotional_weight) +
                emotional_similarity * emotional_weight
            ) * temporal_factor * (1 + resonance_factor * 0.2)
            
            candidates.append((memory, combined_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        selected_emotions = []
        
        for memory, score in candidates:
            if len(selected) >= top_k:
                break
            
            too_similar = False
            for prev_state in selected_emotions:
                if memory.emotional_state.distance_to(prev_state) < 0.3:
                    too_similar = True
                    break
            
            if not too_similar:
                memory.interaction_count += 1
                memory.resonance_score = min(1.0, memory.resonance_score + 0.05)
                selected.append((memory, score))
                selected_emotions.append(memory.emotional_state)
        
        return selected
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    
    def _compute_temporal_relevance(self, memory: EmotionalMemory) -> float:
        if not memory.emotional_trajectory:
            return 0.5
        
        latest_timestamp = memory.emotional_trajectory[-1][0]
        time_delta = (datetime.now() - latest_timestamp).total_seconds() / 3600
        
        decay_rate = 0.05
        temporal_score = np.exp(-decay_rate * time_delta)
        
        recency_bonus = 1.2 if time_delta < 1 else 1.0
        
        return temporal_score * recency_bonus * memory.decay_factor
    
    def update_emotional_context(self, new_state: EmotionalState):
        self.current_emotional_state = new_state
        self.emotional_history.append(new_state)
        
        if len(self.emotional_history) > self.temporal_window:
            self.emotional_history = self.emotional_history[-self.temporal_window:]
    
    def get_emotional_trajectory_summary(self) -> Dict[str, Any]:
        if len(self.emotional_history) < 2:
            return {}
        
        recent_states = self.emotional_history[-20:]
        
        valence_trend = np.polyfit(
            range(len(recent_states)),
            [s.valence for s in recent_states],
            1
        )[0]
        
        arousal_variance = np.var([s.arousal for s in recent_states])
        
        avg_intimacy = np.mean([s.intimacy for s in recent_states])
        avg_cognitive_load = np.mean([s.cognitive_load for s in recent_states])
        
        return {
            'valence_trend': float(valence_trend),
            'emotional_volatility': float(arousal_variance),
            'intimacy_level': float(avg_intimacy),
            'cognitive_engagement': float(avg_cognitive_load),
            'trajectory_complexity': float(len(set(tuple(s.to_vector()) for s in recent_states)) / len(recent_states))
        }
    
    def save_state(self, filepath: Path):
        state = {
            'memories': self.memories,
            'emotional_history': self.emotional_history,
            'current_state': self.current_emotional_state,
            'resonance_cache': self.resonance_cache,
            'model_state': self.embedder.state_dict()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: Path):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.memories = state['memories']
        self.emotional_history = state['emotional_history']
        self.current_emotional_state = state['current_state']
        self.resonance_cache = state['resonance_cache']
        self.embedder.load_state_dict(state['model_state'])
