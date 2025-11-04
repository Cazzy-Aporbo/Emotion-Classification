import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque
from datetime import datetime, timedelta
import heapq
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


@dataclass
class RetrievalContext:
    query_intent: str
    emotional_requirements: Dict[str, float]
    temporal_constraints: Optional[Tuple[datetime, datetime]]
    diversity_requirement: float
    depth_requirement: int
    interaction_history: List[str]


@dataclass
class EmotionalCluster:
    centroid: np.ndarray
    members: List[int]
    emotional_signature: Dict[str, float]
    coherence_score: float
    temporal_span: Tuple[datetime, datetime]


class AffectiveRetriever:
    def __init__(
        self,
        rag_core,
        cluster_epsilon: float = 0.35,
        min_cluster_size: int = 3,
        max_hops: int = 4
    ):
        self.core = rag_core
        self.cluster_epsilon = cluster_epsilon
        self.min_cluster_size = min_cluster_size
        self.max_hops = max_hops
        
        self.query_history = deque(maxlen=50)
        self.interaction_patterns = defaultdict(int)
        self.emotional_clusters = []
        self.cluster_update_threshold = 10
        self.queries_since_cluster_update = 0
        
    def retrieve_with_context(
        self,
        query: str,
        query_embedding: np.ndarray,
        context: RetrievalContext,
        top_k: int = 8
    ) -> List[Tuple[any, float, Dict[str, any]]]:
        self.query_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'context': context
        })
        
        if self.queries_since_cluster_update >= self.cluster_update_threshold:
            self._update_emotional_clusters()
            self.queries_since_cluster_update = 0
        else:
            self.queries_since_cluster_update += 1
        
        retrieval_strategy = self._determine_strategy(query, context)
        
        if retrieval_strategy == 'multi_hop':
            results = self._multi_hop_retrieval(query, query_embedding, context, top_k)
        elif retrieval_strategy == 'cluster_based':
            results = self._cluster_based_retrieval(query, query_embedding, context, top_k)
        elif retrieval_strategy == 'temporal_aware':
            results = self._temporal_aware_retrieval(query, query_embedding, context, top_k)
        else:
            results = self._adaptive_hybrid_retrieval(query, query_embedding, context, top_k)
        
        enhanced_results = self._apply_emotional_filtering(results, context)
        diversified_results = self._ensure_diversity(enhanced_results, context.diversity_requirement)
        
        return diversified_results[:top_k]
    
    def _determine_strategy(self, query: str, context: RetrievalContext) -> str:
        query_lower = query.lower()
        
        causal_markers = ['why', 'how', 'because', 'reason', 'cause', 'explain']
        temporal_markers = ['when', 'before', 'after', 'during', 'timeline', 'history']
        comparative_markers = ['compare', 'contrast', 'difference', 'similar', 'versus']
        
        has_causal = any(marker in query_lower for marker in causal_markers)
        has_temporal = any(marker in query_lower for marker in temporal_markers)
        has_comparative = any(marker in query_lower for marker in comparative_markers)
        
        if has_causal or context.depth_requirement > 2:
            return 'multi_hop'
        elif context.temporal_constraints is not None or has_temporal:
            return 'temporal_aware'
        elif has_comparative or len(self.emotional_clusters) > 5:
            return 'cluster_based'
        else:
            return 'adaptive_hybrid'
    
    def _multi_hop_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        context: RetrievalContext,
        top_k: int
    ) -> List[Tuple[any, float, Dict[str, any]]]:
        visited = set()
        current_nodes = [(query, query_embedding, 1.0, [])]
        all_results = []
        
        for hop in range(self.max_hops):
            if not current_nodes:
                break
            
            next_nodes = []
            hop_weight = 1.0 / (hop + 1) ** 0.7
            
            for current_query, current_embedding, parent_score, path in current_nodes:
                candidates = self.core.retrieve_emotionally_relevant(
                    current_query,
                    current_embedding,
                    top_k=max(5, top_k // (hop + 1)),
                    emotional_weight=0.5 + hop * 0.1
                )
                
                for memory, base_score in candidates:
                    memory_id = id(memory)
                    if memory_id in visited:
                        continue
                    
                    visited.add(memory_id)
                    
                    path_penalty = 1.0 - len(path) * 0.1
                    emotional_continuity = self._compute_emotional_continuity(path, memory)
                    
                    final_score = (
                        base_score * hop_weight * path_penalty * 
                        emotional_continuity * parent_score
                    )
                    
                    metadata = {
                        'hop': hop,
                        'path': path + [current_query],
                        'emotional_continuity': emotional_continuity,
                        'reasoning_chain': self._extract_reasoning_chain(path, memory)
                    }
                    
                    all_results.append((memory, final_score, metadata))
                    
                    if hop < self.max_hops - 1 and len(memory.associations) > 0:
                        for assoc in memory.associations[:2]:
                            assoc_embedding = self._get_association_embedding(assoc)
                            if assoc_embedding is not None:
                                next_nodes.append((
                                    assoc,
                                    assoc_embedding,
                                    final_score * 0.8,
                                    path + [current_query]
                                ))
            
            current_nodes = sorted(next_nodes, key=lambda x: x[2], reverse=True)[:top_k]
        
        return sorted(all_results, key=lambda x: x[1], reverse=True)
    
    def _compute_emotional_continuity(self, path: List[str], memory: any) -> float:
        if not path or not hasattr(memory, 'emotional_state'):
            return 1.0
        
        if len(self.core.emotional_history) < 2:
            return 0.8
        
        recent_states = self.core.emotional_history[-len(path)-1:]
        memory_state = memory.emotional_state
        
        continuities = []
        for state in recent_states:
            distance = state.distance_to(memory_state)
            continuity = 1.0 / (1.0 + distance)
            continuities.append(continuity)
        
        return np.mean(continuities) if continuities else 0.8
    
    def _extract_reasoning_chain(self, path: List[str], memory: any) -> List[str]:
        chain = []
        for i, query in enumerate(path):
            chain.append(f"Step {i+1}: {query[:50]}...")
        chain.append(f"Result: {memory.content[:60]}...")
        return chain
    
    def _get_association_embedding(self, association: str) -> Optional[np.ndarray]:
        for memory in self.core.memories:
            if association in memory.content:
                return memory.embedding
        return None
    
    def _cluster_based_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        context: RetrievalContext,
        top_k: int
    ) -> List[Tuple[any, float, Dict[str, any]]]:
        if not self.emotional_clusters:
            self._update_emotional_clusters()
        
        query_enhanced, query_emotional_state = self.core.encode_emotional_context(
            query, query_embedding
        )
        
        cluster_scores = []
        for cluster in self.emotional_clusters:
            emotional_alignment = self._compute_cluster_emotional_alignment(
                query_emotional_state, cluster
            )
            semantic_similarity = np.dot(
                query_enhanced,
                cluster.centroid
            ) / (np.linalg.norm(query_enhanced) * np.linalg.norm(cluster.centroid) + 1e-8)
            
            cluster_score = emotional_alignment * 0.6 + semantic_similarity * 0.4
            cluster_scores.append((cluster, cluster_score))
        
        cluster_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_clusters = cluster_scores[:max(3, top_k // 3)]
        
        results = []
        for cluster, cluster_score in selected_clusters:
            for member_idx in cluster.members:
                if member_idx < len(self.core.memories):
                    memory = self.core.memories[member_idx]
                    
                    memory_score = self._compute_memory_score_in_cluster(
                        memory, query_enhanced, query_emotional_state, cluster
                    )
                    
                    final_score = memory_score * cluster_score * cluster.coherence_score
                    
                    metadata = {
                        'cluster_id': id(cluster),
                        'cluster_signature': cluster.emotional_signature,
                        'intra_cluster_position': self._compute_cluster_position(memory, cluster)
                    }
                    
                    results.append((memory, final_score, metadata))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _compute_cluster_emotional_alignment(
        self,
        query_state: any,
        cluster: EmotionalCluster
    ) -> float:
        alignments = []
        for key, value in cluster.emotional_signature.items():
            query_value = getattr(query_state, key, 0.5)
            alignment = 1.0 - abs(query_value - value)
            alignments.append(alignment)
        
        return np.mean(alignments) if alignments else 0.5
    
    def _compute_memory_score_in_cluster(
        self,
        memory: any,
        query_embedding: np.ndarray,
        query_emotional_state: any,
        cluster: EmotionalCluster
    ) -> float:
        semantic_sim = np.dot(
            query_embedding, memory.embedding
        ) / (np.linalg.norm(query_embedding) * np.linalg.norm(memory.embedding) + 1e-8)
        
        emotional_distance = query_emotional_state.distance_to(memory.emotional_state)
        emotional_sim = 1.0 / (1.0 + emotional_distance)
        
        return semantic_sim * 0.5 + emotional_sim * 0.5
    
    def _compute_cluster_position(self, memory: any, cluster: EmotionalCluster) -> str:
        distance_to_centroid = np.linalg.norm(memory.embedding - cluster.centroid)
        
        distances = []
        for member_idx in cluster.members:
            if member_idx < len(self.core.memories):
                member = self.core.memories[member_idx]
                dist = np.linalg.norm(member.embedding - cluster.centroid)
                distances.append(dist)
        
        if not distances:
            return "center"
        
        percentile = (np.searchsorted(sorted(distances), distance_to_centroid) / 
                     len(distances))
        
        if percentile < 0.33:
            return "core"
        elif percentile < 0.67:
            return "intermediate"
        else:
            return "peripheral"
    
    def _temporal_aware_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        context: RetrievalContext,
        top_k: int
    ) -> List[Tuple[any, float, Dict[str, any]]]:
        temporal_candidates = []
        
        for memory in self.core.memories:
            if not memory.emotional_trajectory:
                continue
            
            memory_timestamp = memory.emotional_trajectory[-1][0]
            
            if context.temporal_constraints:
                start_time, end_time = context.temporal_constraints
                if not (start_time <= memory_timestamp <= end_time):
                    continue
            
            base_candidates = self.core.retrieve_emotionally_relevant(
                query, query_embedding, top_k=top_k * 3
            )
            
            memory_scores = dict(base_candidates)
            if memory not in memory_scores:
                continue
            
            base_score = memory_scores[memory]
            
            temporal_relevance = self._compute_temporal_pattern_match(
                memory, context
            )
            
            emotional_evolution_score = self._analyze_emotional_evolution(memory)
            
            final_score = (
                base_score * 0.4 +
                temporal_relevance * 0.35 +
                emotional_evolution_score * 0.25
            )
            
            metadata = {
                'timestamp': memory_timestamp,
                'temporal_relevance': temporal_relevance,
                'emotional_evolution': emotional_evolution_score,
                'trajectory_length': len(memory.emotional_trajectory)
            }
            
            temporal_candidates.append((memory, final_score, metadata))
        
        return sorted(temporal_candidates, key=lambda x: x[1], reverse=True)
    
    def _compute_temporal_pattern_match(
        self,
        memory: any,
        context: RetrievalContext
    ) -> float:
        if not context.interaction_history or not memory.emotional_trajectory:
            return 0.5
        
        recent_interactions = context.interaction_history[-5:]
        
        pattern_scores = []
        for interaction in recent_interactions:
            for traj_time, traj_state in memory.emotional_trajectory:
                time_diff = abs((datetime.now() - traj_time).total_seconds())
                time_similarity = 1.0 / (1.0 + time_diff / 3600)
                pattern_scores.append(time_similarity)
        
        return np.mean(pattern_scores) if pattern_scores else 0.5
    
    def _analyze_emotional_evolution(self, memory: any) -> float:
        if len(memory.emotional_trajectory) < 2:
            return 0.5
        
        states = [state for _, state in memory.emotional_trajectory]
        
        valence_changes = [
            abs(states[i].valence - states[i-1].valence)
            for i in range(1, len(states))
        ]
        
        arousal_changes = [
            abs(states[i].arousal - states[i-1].arousal)
            for i in range(1, len(states))
        ]
        
        evolution_complexity = (
            np.mean(valence_changes) * 0.5 +
            np.mean(arousal_changes) * 0.3 +
            len(set(tuple(s.to_vector()) for s in states)) / len(states) * 0.2
        )
        
        return min(1.0, evolution_complexity)
    
    def _adaptive_hybrid_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        context: RetrievalContext,
        top_k: int
    ) -> List[Tuple[any, float, Dict[str, any]]]:
        base_results = self.core.retrieve_emotionally_relevant(
            query, query_embedding, top_k=top_k * 2
        )
        
        enhanced_results = []
        for memory, base_score in base_results:
            adaptivity_score = self._compute_adaptivity_score(memory, context)
            interaction_bonus = self._compute_interaction_bonus(memory)
            resonance_amplification = memory.resonance_score * 0.15
            
            final_score = (
                base_score * 0.6 +
                adaptivity_score * 0.25 +
                interaction_bonus * 0.1 +
                resonance_amplification * 0.05
            )
            
            metadata = {
                'adaptivity': adaptivity_score,
                'interaction_strength': interaction_bonus,
                'resonance': memory.resonance_score,
                'retrieval_method': 'adaptive_hybrid'
            }
            
            enhanced_results.append((memory, final_score, metadata))
        
        return sorted(enhanced_results, key=lambda x: x[1], reverse=True)
    
    def _compute_adaptivity_score(self, memory: any, context: RetrievalContext) -> float:
        if not context.emotional_requirements:
            return 0.7
        
        scores = []
        for dimension, required_value in context.emotional_requirements.items():
            memory_value = getattr(memory.emotional_state, dimension, 0.5)
            alignment = 1.0 - abs(memory_value - required_value)
            scores.append(alignment)
        
        return np.mean(scores) if scores else 0.7
    
    def _compute_interaction_bonus(self, memory: any) -> float:
        normalized_interactions = min(1.0, memory.interaction_count / 20.0)
        return normalized_interactions * 0.5 + 0.5
    
    def _apply_emotional_filtering(
        self,
        results: List[Tuple[any, float, Dict[str, any]]],
        context: RetrievalContext
    ) -> List[Tuple[any, float, Dict[str, any]]]:
        if not context.emotional_requirements:
            return results
        
        filtered = []
        for memory, score, metadata in results:
            passes_filter = True
            
            for dimension, required_value in context.emotional_requirements.items():
                memory_value = getattr(memory.emotional_state, dimension, 0.5)
                if abs(memory_value - required_value) > 0.4:
                    passes_filter = False
                    break
            
            if passes_filter:
                filtered.append((memory, score, metadata))
        
        return filtered if filtered else results
    
    def _ensure_diversity(
        self,
        results: List[Tuple[any, float, Dict[str, any]]],
        diversity_requirement: float
    ) -> List[Tuple[any, float, Dict[str, any]]]:
        if diversity_requirement < 0.3 or len(results) <= 1:
            return results
        
        diverse_results = [results[0]]
        
        for candidate_memory, candidate_score, candidate_metadata in results[1:]:
            min_distance = float('inf')
            
            for selected_memory, _, _ in diverse_results:
                emotional_distance = candidate_memory.emotional_state.distance_to(
                    selected_memory.emotional_state
                )
                
                semantic_similarity = np.dot(
                    candidate_memory.embedding, selected_memory.embedding
                ) / (
                    np.linalg.norm(candidate_memory.embedding) * 
                    np.linalg.norm(selected_memory.embedding) + 1e-8
                )
                
                combined_distance = emotional_distance * 0.6 + (1 - semantic_similarity) * 0.4
                min_distance = min(min_distance, combined_distance)
            
            diversity_threshold = 0.2 + diversity_requirement * 0.5
            
            if min_distance >= diversity_threshold:
                diverse_results.append((candidate_memory, candidate_score, candidate_metadata))
        
        return diverse_results
    
    def _update_emotional_clusters(self):
        if len(self.core.memories) < self.min_cluster_size:
            return
        
        embeddings = np.array([mem.embedding for mem in self.core.memories])
        
        clustering = DBSCAN(
            eps=self.cluster_epsilon,
            min_samples=self.min_cluster_size,
            metric='cosine'
        )
        labels = clustering.fit_predict(embeddings)
        
        self.emotional_clusters = []
        unique_labels = set(labels) - {-1}
        
        for label in unique_labels:
            member_indices = np.where(labels == label)[0]
            member_embeddings = embeddings[member_indices]
            centroid = np.mean(member_embeddings, axis=0)
            
            emotional_values = defaultdict(list)
            timestamps = []
            
            for idx in member_indices:
                memory = self.core.memories[idx]
                for attr in ['valence', 'arousal', 'dominance', 'intimacy',
                            'cognitive_load', 'temporal_urgency', 'social_orientation',
                            'existential_depth', 'aesthetic_resonance', 'semantic_coherence']:
                    emotional_values[attr].append(getattr(memory.emotional_state, attr))
                
                if memory.emotional_trajectory:
                    timestamps.append(memory.emotional_trajectory[-1][0])
            
            emotional_signature = {
                key: float(np.mean(values))
                for key, values in emotional_values.items()
            }
            
            coherence_score = self._compute_cluster_coherence(member_embeddings)
            
            temporal_span = (
                min(timestamps) if timestamps else datetime.now(),
                max(timestamps) if timestamps else datetime.now()
            )
            
            cluster = EmotionalCluster(
                centroid=centroid,
                members=member_indices.tolist(),
                emotional_signature=emotional_signature,
                coherence_score=coherence_score,
                temporal_span=temporal_span
            )
            
            self.emotional_clusters.append(cluster)
    
    def _compute_cluster_coherence(self, embeddings: np.ndarray) -> float:
        if len(embeddings) < 2:
            return 1.0
        
        centroid = np.mean(embeddings, axis=0)
        distances = cdist(embeddings, centroid.reshape(1, -1), metric='cosine')
        coherence = 1.0 - np.mean(distances)
        
        return max(0.0, min(1.0, coherence))
    
    def get_retrieval_analytics(self) -> Dict[str, any]:
        if not self.query_history:
            return {}
        
        recent_queries = list(self.query_history)[-20:]
        
        strategy_counts = defaultdict(int)
        avg_scores = []
        
        return {
            'total_queries': len(self.query_history),
            'active_clusters': len(self.emotional_clusters),
            'interaction_patterns': dict(self.interaction_patterns),
            'cluster_sizes': [len(c.members) for c in self.emotional_clusters],
            'average_cluster_coherence': np.mean([c.coherence_score for c in self.emotional_clusters]) if self.emotional_clusters else 0.0
        }
