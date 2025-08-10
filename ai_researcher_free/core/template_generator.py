"""
Template-based Hypothesis Generator
100% Free hypothesis generation using smart templates and rules
"""

import random
import re
from typing import List, Tuple, Dict, Any
import json
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class TemplateHypothesisGenerator:
    """
    Generate research hypotheses using sophisticated templates
    No API costs - completely free and works offline
    """
    
    def __init__(self):
        self.templates = self._load_hypothesis_templates()
        self.domain_keywords = self._load_domain_keywords()
        self.methodology_patterns = self._load_methodology_patterns()
        
    def _load_hypothesis_templates(self) -> Dict[str, List[str]]:
        """Load sophisticated hypothesis templates by domain"""
        return {
            'computer_vision': [
                "A novel {technique} approach incorporating {concept} could improve {task} performance by addressing {limitation}",
                "Integrating {concept} with {architecture} may enhance {metric} in {application} tasks",
                "A multi-scale {concept} framework could better capture {feature_type} for {task}",
                "Cross-modal attention between {concept} and {modality} features may improve {task}",
                "Self-supervised learning with {concept} could reduce labeled data requirements for {task}",
                "A hierarchical {concept} architecture might achieve better {metric} than current methods",
                "Combining {concept} with adversarial training could enhance model robustness in {scenario}",
                "Temporal modeling of {concept} could improve performance in video-based {task}",
                "A lightweight {concept} approach could maintain accuracy while reducing computational cost",
                "Few-shot learning enhanced with {concept} may enable better generalization to new {domain}"
            ],
            'natural_language_processing': [
                "A contextual {concept} model could improve {task} by better capturing {linguistic_feature}",
                "Multi-task learning incorporating {concept} may enhance performance across {task_family}",
                "A pre-trained {concept} approach could achieve better {metric} with less training data",
                "Cross-lingual {concept} could enable better transfer learning for low-resource languages",
                "Incorporating {concept} into transformer architectures may improve {task} efficiency",
                "A graph-based {concept} approach could better model relationships in {task}",
                "Adaptive {concept} mechanisms may enhance model performance on diverse {task} domains",
                "Meta-learning with {concept} could improve few-shot performance in {task}",
                "A unified {concept} framework could handle multiple {task_type} tasks simultaneously",
                "Contrastive learning enhanced with {concept} may improve representation quality"
            ],
            'attention_mechanisms': [
                "A sparse {concept} attention pattern could reduce computational complexity while maintaining performance",
                "Multi-head attention enhanced with {concept} may capture more diverse feature relationships",
                "Cross-attention between {concept} and {feature_type} could improve multimodal understanding",
                "Learnable {concept} attention weights may adapt better to different input distributions",
                "Hierarchical {concept} attention could capture both local and global dependencies",
                "A gated {concept} attention mechanism may selectively focus on relevant information",
                "Temporal {concept} attention could better model sequential dependencies in {task}",
                "Self-attention with {concept} regularization may prevent overfitting to training patterns",
                "A hybrid {concept} attention approach could combine benefits of different attention types",
                "Dynamic {concept} attention that adapts to input complexity may improve efficiency"
            ],
            'neural_networks': [
                "A novel {concept} architecture could achieve better {metric} than existing approaches",
                "Regularization techniques incorporating {concept} may reduce overfitting in {scenario}",
                "A residual {concept} design could improve gradient flow in deep networks",
                "Adaptive {concept} layers may automatically adjust to different input characteristics",
                "A modular {concept} architecture could enable better transfer learning",
                "Incorporating {concept} into the loss function may improve training stability",
                "A pruning method based on {concept} could reduce model size while maintaining accuracy",
                "Ensemble methods enhanced with {concept} may achieve better robustness",
                "A neuromorphic {concept} approach could improve energy efficiency",
                "Meta-learning with {concept} could enable faster adaptation to new tasks"
            ],
            'reinforcement_learning': [
                "A {concept}-based reward shaping approach could improve learning efficiency in {environment}",
                "Multi-agent {concept} could enhance coordination in collaborative tasks",
                "Hierarchical {concept} may enable better long-term planning in complex environments",
                "Incorporating {concept} into policy gradient methods may reduce variance",
                "A model-based {concept} approach could improve sample efficiency",
                "Curiosity-driven exploration with {concept} may discover better strategies",
                "A distributed {concept} framework could scale to larger state spaces",
                "Imitation learning enhanced with {concept} may require fewer expert demonstrations",
                "A safety-constrained {concept} approach could prevent harmful actions",
                "Meta-reinforcement learning with {concept} could enable faster adaptation"
            ],
            'general_ml': [
                "A {concept}-based approach could improve {task} performance by leveraging {property}",
                "Combining {concept} with {technique} may address current limitations in {domain}",
                "A novel {concept} framework could enable better handling of {challenge}",
                "Incorporating {concept} into existing methods may improve {metric}",
                "A data-efficient {concept} approach could reduce annotation requirements",
                "A robust {concept} method could maintain performance under {adversarial_condition}",
                "A scalable {concept} algorithm could handle larger datasets more efficiently",
                "An interpretable {concept} model could provide better insights into {phenomenon}",
                "A transfer learning approach with {concept} could improve performance on {target_domain}",
                "An online {concept} method could adapt to changing data distributions"
            ]
        }
    
    def _load_domain_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """Load domain-specific keywords for template filling"""
        return {
            'computer_vision': {
                'technique': ['deep learning', 'convolutional', 'transformer-based', 'graph neural network', 
                             'self-supervised', 'contrastive learning', 'adversarial', 'meta-learning'],
                'concept': ['attention mechanism', 'feature fusion', 'multi-scale processing', 'spatial reasoning',
                           'temporal modeling', 'cross-modal learning', 'representation learning'],
                'task': ['object detection', 'image classification', 'semantic segmentation', 'instance segmentation',
                        'image captioning', 'visual question answering', 'action recognition', 'pose estimation'],
                'architecture': ['ResNet', 'Vision Transformer', 'EfficientNet', 'YOLO', 'Mask R-CNN', 'U-Net'],
                'metric': ['accuracy', 'mAP', 'IoU', 'F1-score', 'BLEU score', 'inference speed'],
                'application': ['autonomous driving', 'medical imaging', 'surveillance', 'robotics', 'AR/VR'],
                'limitation': ['occlusion handling', 'scale variation', 'lighting conditions', 'computational cost'],
                'feature_type': ['local features', 'global context', 'temporal patterns', 'spatial relationships'],
                'modality': ['RGB', 'depth', 'thermal', 'LiDAR', 'audio'],
                'scenario': ['domain shift', 'adversarial attacks', 'limited data', 'real-time constraints']
            },
            'natural_language_processing': {
                'concept': ['attention mechanism', 'contextualized embeddings', 'cross-lingual representation',
                           'syntactic parsing', 'semantic role labeling', 'discourse modeling'],
                'task': ['sentiment analysis', 'named entity recognition', 'machine translation', 'question answering',
                        'text summarization', 'dialogue generation', 'reading comprehension'],
                'linguistic_feature': ['syntactic structure', 'semantic relationships', 'pragmatic context',
                                     'discourse coherence', 'world knowledge', 'temporal reasoning'],
                'task_family': ['classification tasks', 'generation tasks', 'structured prediction', 'sequence labeling'],
                'metric': ['BLEU score', 'ROUGE score', 'perplexity', 'F1-score', 'exact match', 'human evaluation'],
                'task_type': ['understanding', 'generation', 'reasoning'],
                'domain': ['biomedical', 'legal', 'scientific', 'conversational', 'social media']
            },
            'attention_mechanisms': {
                'concept': ['multi-head', 'cross-modal', 'temporal', 'spatial', 'hierarchical', 'sparse', 'adaptive'],
                'feature_type': ['visual features', 'textual features', 'temporal patterns', 'spatial patterns'],
                'task': ['machine translation', 'image captioning', 'video understanding', 'multimodal reasoning']
            },
            'neural_networks': {
                'concept': ['skip connections', 'attention modules', 'normalization layers', 'activation functions',
                           'regularization', 'pruning', 'quantization', 'knowledge distillation'],
                'metric': ['accuracy', 'convergence speed', 'memory usage', 'inference time', 'energy consumption'],
                'scenario': ['limited data', 'noisy labels', 'distribution shift', 'adversarial examples']
            },
            'reinforcement_learning': {
                'concept': ['policy gradient', 'value function', 'exploration strategy', 'reward shaping',
                           'hierarchical decomposition', 'multi-agent coordination'],
                'environment': ['continuous control', 'discrete action spaces', 'partial observability',
                              'multi-agent settings', 'sparse rewards'],
                'property': ['sample efficiency', 'exploration capability', 'generalization', 'robustness']
            },
            'general_ml': {
                'concept': ['ensemble methods', 'regularization', 'optimization', 'feature selection',
                           'dimensionality reduction', 'clustering', 'anomaly detection'],
                'task': ['classification', 'regression', 'clustering', 'dimensionality reduction',
                        'anomaly detection', 'recommendation'],
                'technique': ['supervised learning', 'unsupervised learning', 'semi-supervised learning',
                             'active learning', 'transfer learning', 'meta-learning'],
                'property': ['interpretability', 'scalability', 'robustness', 'efficiency'],
                'challenge': ['imbalanced data', 'missing values', 'high dimensionality', 'concept drift'],
                'metric': ['accuracy', 'precision', 'recall', 'AUC', 'silhouette score'],
                'adversarial_condition': ['noise', 'outliers', 'distributional shift', 'adversarial examples'],
                'target_domain': ['medical diagnosis', 'financial prediction', 'natural language', 'computer vision'],
                'phenomenon': ['feature importance', 'decision boundaries', 'cluster structure', 'anomaly patterns']
            }
        }
    
    def _load_methodology_patterns(self) -> Dict[str, List[str]]:
        """Load methodology patterns for different research types"""
        return {
            'novel_architecture': [
                "proposes a new {architecture_type} that {innovation}",
                "introduces a novel {component} design for {task}",
                "presents an innovative {method} approach to {problem}"
            ],
            'improvement': [
                "improves upon existing {baseline} by {enhancement}",
                "enhances {current_method} through {technique}",
                "achieves better {metric} by incorporating {concept}"
            ],
            'combination': [
                "combines {method1} with {method2} to {benefit}",
                "integrates {technique} into {framework} for {task}",
                "merges {approach1} and {approach2} to address {limitation}"
            ],
            'application': [
                "applies {method} to {new_domain} for the first time",
                "extends {technique} to handle {new_scenario}",
                "adapts {approach} for {specific_application}"
            ]
        }
    
    def generate_hypotheses(self, research_gaps: List[Dict[str, Any]], 
                          num_hypotheses: int = 5) -> List[Tuple[str, float]]:
        """
        Generate hypotheses for research gaps using templates
        
        Args:
            research_gaps: List of research gaps with metadata
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            List of (hypothesis, confidence_score) tuples
        """
        hypotheses = []
        
        for gap in research_gaps[:num_hypotheses]:
            # Determine domain and concept
            domain = gap.get('category', 'general_ml')
            concept = gap.get('concept', 'novel approach')
            
            # Generate hypothesis
            hypothesis = self._generate_single_hypothesis(concept, domain, gap)
            confidence = self._calculate_confidence(hypothesis, gap)
            
            hypotheses.append((hypothesis, confidence))
        
        # Sort by confidence and return top hypotheses
        hypotheses.sort(key=lambda x: x[1], reverse=True)
        return hypotheses[:num_hypotheses]
    
    def _generate_single_hypothesis(self, concept: str, domain: str, 
                                   gap_info: Dict[str, Any]) -> str:
        """Generate a single hypothesis for a concept and domain"""
        # Get templates for domain
        domain_templates = self.templates.get(domain, self.templates['general_ml'])
        template = random.choice(domain_templates)
        
        # Get keywords for domain
        domain_keywords = self.domain_keywords.get(domain, self.domain_keywords['general_ml'])
        
        # Fill template with appropriate keywords
        filled_template = self._fill_template(template, concept, domain_keywords)
        
        # Post-process to ensure quality
        hypothesis = self._post_process_hypothesis(filled_template, concept)
        
        return hypothesis
    
    def _fill_template(self, template: str, concept: str, 
                      keywords: Dict[str, List[str]]) -> str:
        """Fill template with appropriate keywords"""
        # Find all placeholders in template
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        # Fill each placeholder
        filled = template
        for placeholder in placeholders:
            if placeholder == 'concept':
                replacement = concept
            elif placeholder in keywords:
                replacement = random.choice(keywords[placeholder])
            else:
                # Fallback to generic terms
                replacement = self._get_generic_replacement(placeholder)
            
            filled = filled.replace(f'{{{placeholder}}}', replacement)
        
        return filled
    
    def _get_generic_replacement(self, placeholder: str) -> str:
        """Get generic replacement for unknown placeholders"""
        generic_replacements = {
            'task': 'machine learning task',
            'metric': 'performance',
            'technique': 'novel approach',
            'method': 'algorithm',
            'architecture': 'neural network',
            'framework': 'system',
            'approach': 'method',
            'limitation': 'current constraints',
            'challenge': 'existing problem',
            'domain': 'application area',
            'scenario': 'use case',
            'property': 'characteristic',
            'benefit': 'improved performance',
            'enhancement': 'optimization',
            'innovation': 'advances the field'
        }
        
        return generic_replacements.get(placeholder, 'novel contribution')
    
    def _post_process_hypothesis(self, hypothesis: str, concept: str) -> str:
        """Post-process hypothesis to ensure quality and readability"""
        # Capitalize first letter
        hypothesis = hypothesis[0].upper() + hypothesis[1:] if hypothesis else ""
        
        # Ensure it ends with proper punctuation
        if not hypothesis.endswith('.'):
            hypothesis += '.'
        
        # Remove redundant spaces
        hypothesis = re.sub(r'\s+', ' ', hypothesis)
        
        # Ensure concept is properly integrated
        if concept.lower() not in hypothesis.lower():
            # Try to naturally incorporate the concept
            if 'novel' in hypothesis.lower():
                hypothesis = hypothesis.replace('novel', f'novel {concept}-based')
            else:
                hypothesis = f"A {concept}-enhanced approach that " + hypothesis.lower()
        
        return hypothesis.strip()
    
    def _calculate_confidence(self, hypothesis: str, gap_info: Dict[str, Any]) -> float:
        """Calculate confidence score for generated hypothesis"""
        confidence = 0.5  # Base confidence
        
        # Bonus for specific terminology
        if any(term in hypothesis.lower() for term in ['novel', 'improve', 'enhance', 'better']):
            confidence += 0.1
        
        # Bonus for technical terms
        technical_terms = ['neural', 'attention', 'learning', 'algorithm', 'model', 'network']
        tech_count = sum(1 for term in technical_terms if term in hypothesis.lower())
        confidence += min(0.2, tech_count * 0.05)
        
        # Bonus for gap confidence
        gap_confidence = gap_info.get('confidence_score', 0.5)
        confidence += gap_confidence * 0.2
        
        # Penalty for overly long hypotheses
        word_count = len(hypothesis.split())
        if word_count > 25:
            confidence -= 0.1
        
        # Ensure confidence is in valid range
        return max(0.1, min(0.95, confidence))
    
    def generate_methodology_description(self, hypothesis: str) -> str:
        """Generate methodology description for a hypothesis"""
        # Analyze hypothesis to determine research type
        research_type = self._classify_research_type(hypothesis)
        
        # Get methodology pattern
        patterns = self.methodology_patterns.get(research_type, self.methodology_patterns['novel_architecture'])
        pattern = random.choice(patterns)
        
        # Extract key components from hypothesis
        components = self._extract_hypothesis_components(hypothesis)
        
        # Fill methodology pattern
        methodology = self._fill_methodology_pattern(pattern, components)
        
        return methodology
    
    def _classify_research_type(self, hypothesis: str) -> str:
        """Classify the type of research based on hypothesis"""
        hypothesis_lower = hypothesis.lower()
        
        if any(word in hypothesis_lower for word in ['novel', 'new', 'proposes', 'introduces']):
            return 'novel_architecture'
        elif any(word in hypothesis_lower for word in ['improves', 'enhances', 'better', 'outperform']):
            return 'improvement'
        elif any(word in hypothesis_lower for word in ['combines', 'integrates', 'merges']):
            return 'combination'
        elif any(word in hypothesis_lower for word in ['applies', 'extends', 'adapts']):
            return 'application'
        else:
            return 'novel_architecture'
    
    def _extract_hypothesis_components(self, hypothesis: str) -> Dict[str, str]:
        """Extract key components from hypothesis"""
        # Simple keyword extraction
        components = {
            'method': 'proposed approach',
            'task': 'target task',
            'benefit': 'improved performance',
            'technique': 'novel technique'
        }
        
        # Try to extract specific terms
        if 'attention' in hypothesis.lower():
            components['method'] = 'attention mechanism'
        if 'transformer' in hypothesis.lower():
            components['method'] = 'transformer architecture'
        if 'classification' in hypothesis.lower():
            components['task'] = 'classification'
        if 'detection' in hypothesis.lower():
            components['task'] = 'object detection'
        
        return components
    
    def _fill_methodology_pattern(self, pattern: str, components: Dict[str, str]) -> str:
        """Fill methodology pattern with components"""
        filled = pattern
        for key, value in components.items():
            filled = filled.replace(f'{{{key}}}', value)
        
        return filled
    
    def generate_research_questions(self, hypothesis: str) -> List[str]:
        """Generate research questions based on hypothesis"""
        questions = []
        
        # Question templates
        templates = [
            f"How does {self._extract_method(hypothesis)} compare to existing approaches?",
            f"What are the computational requirements of {self._extract_method(hypothesis)}?",
            f"In which scenarios does {self._extract_method(hypothesis)} perform best?",
            f"How robust is {self._extract_method(hypothesis)} to different input conditions?",
            f"What are the limitations of {self._extract_method(hypothesis)}?",
            f"How can {self._extract_method(hypothesis)} be extended to other domains?",
            f"What is the theoretical foundation for {self._extract_method(hypothesis)}?",
            f"How does {self._extract_method(hypothesis)} scale with dataset size?"
        ]
        
        # Select relevant questions
        questions = random.sample(templates, min(5, len(templates)))
        
        return questions
    
    def _extract_method(self, hypothesis: str) -> str:
        """Extract method name from hypothesis"""
        # Simple extraction - look for key method indicators
        if 'attention' in hypothesis.lower():
            return 'the proposed attention mechanism'
        elif 'transformer' in hypothesis.lower():
            return 'the transformer-based approach'
        elif 'neural' in hypothesis.lower():
            return 'the neural network method'
        else:
            return 'the proposed method'
    
    def export_hypotheses(self, hypotheses: List[Tuple[str, float]], 
                         filename: str = None) -> str:
        """Export hypotheses to JSON format"""
        if filename is None:
            filename = f"hypotheses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'total_hypotheses': len(hypotheses),
            'hypotheses': [
                {
                    'id': i + 1,
                    'hypothesis': hypothesis,
                    'confidence_score': confidence,
                    'methodology': self.generate_methodology_description(hypothesis),
                    'research_questions': self.generate_research_questions(hypothesis)
                }
                for i, (hypothesis, confidence) in enumerate(hypotheses)
            ]
        }
        
        filepath = f"results/{filename}"
        os.makedirs('results', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Hypotheses exported to {filepath}")
        return filepath
