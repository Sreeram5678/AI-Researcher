"""
Paper Analyzer
Analyzes research papers and extracts insights using free methods
"""

import re
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class PaperAnalyzer:
    """
    Analyze research papers to extract insights and trends
    Uses only free text processing and analysis methods
    """
    
    def __init__(self):
        self.methodology_patterns = self._load_methodology_patterns()
        self.result_patterns = self._load_result_patterns()
        self.domain_keywords = self._load_domain_keywords()
        
    def _load_methodology_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for identifying methodologies in papers"""
        return {
            'deep_learning': [
                r'deep\s+neural\s+network', r'convolutional\s+neural\s+network', 
                r'recurrent\s+neural\s+network', r'transformer', r'bert', r'gpt'
            ],
            'attention_mechanisms': [
                r'attention\s+mechanism', r'self-attention', r'multi-head\s+attention',
                r'cross-attention', r'scaled\s+dot-product\s+attention'
            ],
            'computer_vision': [
                r'object\s+detection', r'image\s+classification', r'semantic\s+segmentation',
                r'instance\s+segmentation', r'image\s+captioning', r'visual\s+question\s+answering'
            ],
            'natural_language_processing': [
                r'natural\s+language\s+processing', r'machine\s+translation', 
                r'sentiment\s+analysis', r'named\s+entity\s+recognition', r'text\s+summarization'
            ],
            'reinforcement_learning': [
                r'reinforcement\s+learning', r'policy\s+gradient', r'q-learning',
                r'actor-critic', r'deep\s+q-network'
            ]
        }
    
    def _load_result_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for extracting results from papers"""
        return {
            'performance_metrics': [
                r'accuracy\s*[:=]\s*(\d+\.?\d*)%?',
                r'f1[-\s]score\s*[:=]\s*(\d+\.?\d*)',
                r'map\s*[:=]\s*(\d+\.?\d*)',
                r'bleu\s*[:=]\s*(\d+\.?\d*)',
                r'rouge\s*[:=]\s*(\d+\.?\d*)',
                r'perplexity\s*[:=]\s*(\d+\.?\d*)'
            ],
            'improvements': [
                r'(\d+\.?\d*)%?\s*improvement',
                r'outperform[s]?\s+.*by\s+(\d+\.?\d*)%?',
                r'(\d+\.?\d*)%?\s*better\s+than',
                r'achieve[s]?\s+(\d+\.?\d*)%?\s*accuracy'
            ],
            'datasets': [
                r'cifar-?10', r'cifar-?100', r'imagenet', r'mnist', r'coco',
                r'glue', r'squad', r'wmt', r'penn\s+treebank'
            ]
        }
    
    def _load_domain_keywords(self) -> Dict[str, List[str]]:
        """Load domain-specific keywords"""
        return {
            'computer_vision': [
                'image', 'visual', 'object', 'detection', 'segmentation', 'classification',
                'convolutional', 'cnn', 'vision transformer', 'vit', 'resnet'
            ],
            'natural_language_processing': [
                'text', 'language', 'linguistic', 'semantic', 'syntactic', 'nlp',
                'transformer', 'bert', 'gpt', 'attention', 'embedding'
            ],
            'machine_learning': [
                'learning', 'training', 'optimization', 'gradient', 'neural',
                'network', 'algorithm', 'model', 'supervised', 'unsupervised'
            ],
            'reinforcement_learning': [
                'reinforcement', 'policy', 'reward', 'agent', 'environment',
                'exploration', 'exploitation', 'q-learning', 'actor-critic'
            ]
        }
    
    def analyze_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single research paper
        
        Args:
            paper: Dictionary containing paper data (title, abstract, etc.)
            
        Returns:
            Analysis results dictionary
        """
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        
        analysis = {
            'paper_id': paper.get('id', 'unknown'),
            'title': paper.get('title', ''),
            'methodologies': self._extract_methodologies(text),
            'domains': self._classify_domains(text),
            'performance_metrics': self._extract_performance_metrics(text),
            'datasets': self._extract_datasets(text),
            'key_concepts': self._extract_key_concepts(text),
            'novelty_indicators': self._identify_novelty_indicators(text),
            'technical_depth': self._assess_technical_depth(text),
            'research_type': self._classify_research_type(text)
        }
        
        return analysis
    
    def _extract_methodologies(self, text: str) -> Dict[str, List[str]]:
        """Extract methodologies mentioned in the paper"""
        methodologies = {}
        
        for category, patterns in self.methodology_patterns.items():
            found_methods = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    found_methods.extend(matches)
            
            if found_methods:
                methodologies[category] = list(set(found_methods))
        
        return methodologies
    
    def _classify_domains(self, text: str) -> Dict[str, float]:
        """Classify paper into research domains with confidence scores"""
        domain_scores = {}
        text_lower = text.lower()
        
        for domain, keywords in self.domain_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences and weight by keyword importance
                count = text_lower.count(keyword.lower())
                weight = 1.0 / len(keyword.split())  # Shorter terms get higher weight
                score += count * weight
            
            # Normalize by text length
            normalized_score = score / max(1, len(text.split()) / 100)
            domain_scores[domain] = round(normalized_score, 3)
        
        return domain_scores
    
    def _extract_performance_metrics(self, text: str) -> Dict[str, List[str]]:
        """Extract performance metrics from paper"""
        metrics = {}
        
        for metric_type, patterns in self.result_patterns['performance_metrics']:
            found_metrics = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found_metrics.extend(matches)
            
            if found_metrics:
                metrics[metric_type] = found_metrics
        
        return metrics
    
    def _extract_datasets(self, text: str) -> List[str]:
        """Extract datasets mentioned in the paper"""
        datasets = []
        patterns = self.result_patterns['datasets']
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            datasets.extend(matches)
        
        return list(set(datasets))
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key technical concepts"""
        # Extract capitalized phrases (likely technical terms)
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Extract acronyms
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        
        # Filter for ML/AI related terms
        ml_keywords = ['neural', 'network', 'learning', 'attention', 'transformer', 
                       'convolution', 'optimization', 'gradient', 'embedding']
        
        relevant_concepts = []
        for term in technical_terms + acronyms:
            if any(keyword in term.lower() for keyword in ml_keywords):
                relevant_concepts.append(term)
        
        return list(set(relevant_concepts))[:20]  # Limit to top 20
    
    def _identify_novelty_indicators(self, text: str) -> Dict[str, int]:
        """Identify indicators of novelty in the paper"""
        novelty_patterns = {
            'novel': r'\bnovel\b',
            'new': r'\bnew\s+(?:method|approach|technique|model|architecture)\b',
            'first': r'\bfirst\s+time\b|\bfirst\s+to\b',
            'propose': r'\bpropose[d]?\b',
            'introduce': r'\bintroduce[d]?\b',
            'present': r'\bpresent\s+a\b'
        }
        
        indicators = {}
        for indicator, pattern in novelty_patterns.items():
            count = len(re.findall(pattern, text, re.IGNORECASE))
            indicators[indicator] = count
        
        return indicators
    
    def _assess_technical_depth(self, text: str) -> Dict[str, Any]:
        """Assess the technical depth of the paper"""
        # Count mathematical expressions
        math_expressions = len(re.findall(r'\$.*?\$|\\[a-zA-Z]+', text))
        
        # Count algorithm descriptions
        algorithm_keywords = ['algorithm', 'procedure', 'step', 'iteration', 'convergence']
        algorithm_mentions = sum(text.lower().count(keyword) for keyword in algorithm_keywords)
        
        # Count experimental details
        experiment_keywords = ['experiment', 'evaluation', 'benchmark', 'baseline', 'dataset']
        experiment_mentions = sum(text.lower().count(keyword) for keyword in experiment_keywords)
        
        # Technical term density
        technical_terms = ['optimization', 'gradient', 'backpropagation', 'regularization',
                          'hyperparameter', 'architecture', 'embedding', 'attention']
        technical_density = sum(text.lower().count(term) for term in technical_terms)
        
        return {
            'math_expressions': math_expressions,
            'algorithm_mentions': algorithm_mentions,
            'experiment_mentions': experiment_mentions,
            'technical_density': technical_density,
            'depth_score': (math_expressions + algorithm_mentions + 
                          experiment_mentions + technical_density) / max(1, len(text.split()) / 100)
        }
    
    def _classify_research_type(self, text: str) -> str:
        """Classify the type of research contribution"""
        text_lower = text.lower()
        
        # Empirical research
        if any(word in text_lower for word in ['experiment', 'evaluation', 'benchmark', 'empirical']):
            return 'empirical'
        
        # Theoretical research
        elif any(word in text_lower for word in ['theoretical', 'analysis', 'proof', 'theorem']):
            return 'theoretical'
        
        # Methodological research
        elif any(word in text_lower for word in ['method', 'algorithm', 'approach', 'technique']):
            return 'methodological'
        
        # Survey/review
        elif any(word in text_lower for word in ['survey', 'review', 'overview', 'comprehensive']):
            return 'survey'
        
        # Application research
        elif any(word in text_lower for word in ['application', 'system', 'implementation']):
            return 'application'
        
        else:
            return 'unknown'
    
    def analyze_paper_collection(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a collection of papers to identify trends
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Collection analysis results
        """
        logger.info(f"Analyzing collection of {len(papers)} papers")
        
        # Analyze each paper
        paper_analyses = []
        for paper in papers:
            analysis = self.analyze_paper(paper)
            paper_analyses.append(analysis)
        
        # Aggregate analyses
        collection_analysis = {
            'total_papers': len(papers),
            'domain_distribution': self._aggregate_domains(paper_analyses),
            'methodology_trends': self._aggregate_methodologies(paper_analyses),
            'research_type_distribution': self._aggregate_research_types(paper_analyses),
            'technical_depth_stats': self._aggregate_technical_depth(paper_analyses),
            'common_datasets': self._aggregate_datasets(paper_analyses),
            'novelty_analysis': self._aggregate_novelty(paper_analyses),
            'temporal_trends': self._analyze_temporal_trends(papers),
            'emerging_concepts': self._identify_emerging_concepts(paper_analyses),
            'research_gaps': self._identify_potential_gaps(paper_analyses)
        }
        
        return collection_analysis
    
    def _aggregate_domains(self, analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate domain classifications across papers"""
        domain_totals = {}
        
        for analysis in analyses:
            for domain, score in analysis['domains'].items():
                domain_totals[domain] = domain_totals.get(domain, 0) + score
        
        # Normalize by number of papers
        return {domain: score / len(analyses) for domain, score in domain_totals.items()}
    
    def _aggregate_methodologies(self, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate methodology usage across papers"""
        methodology_counts = {}
        
        for analysis in analyses:
            for category, methods in analysis['methodologies'].items():
                methodology_counts[category] = methodology_counts.get(category, 0) + len(methods)
        
        return methodology_counts
    
    def _aggregate_research_types(self, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate research type distribution"""
        type_counts = Counter([analysis['research_type'] for analysis in analyses])
        return dict(type_counts)
    
    def _aggregate_technical_depth(self, analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate technical depth statistics"""
        depth_scores = [analysis['technical_depth']['depth_score'] for analysis in analyses]
        
        return {
            'mean_depth': np.mean(depth_scores),
            'median_depth': np.median(depth_scores),
            'std_depth': np.std(depth_scores),
            'min_depth': np.min(depth_scores),
            'max_depth': np.max(depth_scores)
        }
    
    def _aggregate_datasets(self, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate dataset usage across papers"""
        dataset_counts = Counter()
        
        for analysis in analyses:
            for dataset in analysis['datasets']:
                dataset_counts[dataset.lower()] += 1
        
        return dict(dataset_counts.most_common(20))  # Top 20 datasets
    
    def _aggregate_novelty(self, analyses: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate novelty indicators"""
        novelty_totals = {}
        
        for analysis in analyses:
            for indicator, count in analysis['novelty_indicators'].items():
                novelty_totals[indicator] = novelty_totals.get(indicator, 0) + count
        
        # Average per paper
        return {indicator: count / len(analyses) for indicator, count in novelty_totals.items()}
    
    def _analyze_temporal_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in the paper collection"""
        # Group papers by year
        year_groups = {}
        for paper in papers:
            try:
                if 'published' in paper:
                    year = datetime.fromisoformat(paper['published'].replace('Z', '+00:00')).year
                    year_groups[year] = year_groups.get(year, 0) + 1
            except:
                continue
        
        if not year_groups:
            return {'message': 'No temporal data available'}
        
        return {
            'papers_by_year': year_groups,
            'year_range': f"{min(year_groups.keys())}-{max(year_groups.keys())}",
            'peak_year': max(year_groups, key=year_groups.get),
            'recent_growth': self._calculate_recent_growth(year_groups)
        }
    
    def _calculate_recent_growth(self, year_groups: Dict[int, int]) -> float:
        """Calculate growth rate in recent years"""
        years = sorted(year_groups.keys())
        if len(years) < 2:
            return 0.0
        
        recent_years = years[-3:]  # Last 3 years
        if len(recent_years) < 2:
            return 0.0
        
        old_count = year_groups[recent_years[0]]
        new_count = year_groups[recent_years[-1]]
        
        if old_count == 0:
            return float('inf') if new_count > 0 else 0.0
        
        return (new_count - old_count) / old_count * 100
    
    def _identify_emerging_concepts(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify emerging concepts across papers"""
        concept_counts = Counter()
        
        for analysis in analyses:
            for concept in analysis['key_concepts']:
                concept_counts[concept] += 1
        
        # Consider concepts that appear in multiple papers but not too common
        emerging = []
        total_papers = len(analyses)
        
        for concept, count in concept_counts.items():
            frequency = count / total_papers
            if 0.1 <= frequency <= 0.3:  # Appears in 10-30% of papers
                emerging.append(concept)
        
        return emerging[:15]  # Top 15 emerging concepts
    
    def _identify_potential_gaps(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify potential research gaps"""
        # Look for combinations of domains and methodologies that are underexplored
        domain_method_pairs = []
        
        for analysis in analyses:
            primary_domain = max(analysis['domains'], key=analysis['domains'].get)
            for method_category in analysis['methodologies'].keys():
                domain_method_pairs.append(f"{primary_domain} + {method_category}")
        
        pair_counts = Counter(domain_method_pairs)
        
        # Generate potential combinations and find underexplored ones
        all_domains = ['computer_vision', 'natural_language_processing', 'reinforcement_learning']
        all_methods = ['deep_learning', 'attention_mechanisms', 'reinforcement_learning']
        
        gaps = []
        for domain in all_domains:
            for method in all_methods:
                if domain != method:  # Avoid redundant combinations
                    pair = f"{domain} + {method}"
                    if pair_counts.get(pair, 0) < 2:  # Appears in less than 2 papers
                        gaps.append(f"Limited exploration of {method} in {domain}")
        
        return gaps[:10]  # Top 10 potential gaps
    
    def generate_trend_report(self, collection_analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive trend report"""
        report = f"""
# Research Trend Analysis Report

## Overview
- Total Papers Analyzed: {collection_analysis['total_papers']}
- Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Domain Distribution
"""
        
        for domain, score in sorted(collection_analysis['domain_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
            report += f"- {domain.replace('_', ' ').title()}: {score:.2f}\n"
        
        report += "\n## Methodology Trends\n"
        for method, count in sorted(collection_analysis['methodology_trends'].items(),
                                   key=lambda x: x[1], reverse=True):
            report += f"- {method.replace('_', ' ').title()}: {count} papers\n"
        
        report += "\n## Research Types\n"
        for rtype, count in collection_analysis['research_type_distribution'].items():
            report += f"- {rtype.title()}: {count} papers\n"
        
        report += "\n## Popular Datasets\n"
        for dataset, count in list(collection_analysis['common_datasets'].items())[:10]:
            report += f"- {dataset}: {count} papers\n"
        
        report += "\n## Emerging Concepts\n"
        for concept in collection_analysis['emerging_concepts']:
            report += f"- {concept}\n"
        
        report += "\n## Potential Research Gaps\n"
        for gap in collection_analysis['research_gaps']:
            report += f"- {gap}\n"
        
        return report
    
    def create_visualizations(self, collection_analysis: Dict[str, Any], 
                            output_dir: str = "results") -> List[str]:
        """Create visualizations for the analysis"""
        os.makedirs(output_dir, exist_ok=True)
        saved_plots = []
        
        # Domain distribution pie chart
        if collection_analysis['domain_distribution']:
            plt.figure(figsize=(10, 6))
            domains = list(collection_analysis['domain_distribution'].keys())
            scores = list(collection_analysis['domain_distribution'].values())
            
            plt.pie(scores, labels=[d.replace('_', ' ').title() for d in domains], autopct='%1.1f%%')
            plt.title('Research Domain Distribution')
            
            plot_path = os.path.join(output_dir, 'domain_distribution.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_path)
        
        # Methodology trends bar chart
        if collection_analysis['methodology_trends']:
            plt.figure(figsize=(12, 6))
            methods = list(collection_analysis['methodology_trends'].keys())
            counts = list(collection_analysis['methodology_trends'].values())
            
            plt.bar([m.replace('_', ' ').title() for m in methods], counts)
            plt.title('Methodology Usage Trends')
            plt.xlabel('Methodology')
            plt.ylabel('Number of Papers')
            plt.xticks(rotation=45, ha='right')
            
            plot_path = os.path.join(output_dir, 'methodology_trends.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_path)
        
        # Temporal trends if available
        if 'temporal_trends' in collection_analysis and 'papers_by_year' in collection_analysis['temporal_trends']:
            plt.figure(figsize=(10, 6))
            years = sorted(collection_analysis['temporal_trends']['papers_by_year'].keys())
            counts = [collection_analysis['temporal_trends']['papers_by_year'][year] for year in years]
            
            plt.plot(years, counts, marker='o')
            plt.title('Publication Trends Over Time')
            plt.xlabel('Year')
            plt.ylabel('Number of Papers')
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_dir, 'temporal_trends.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            saved_plots.append(plot_path)
        
        logger.info(f"Created {len(saved_plots)} visualizations in {output_dir}")
        return saved_plots
    
    def export_analysis(self, collection_analysis: Dict[str, Any], 
                       filename: str = None) -> str:
        """Export analysis results to JSON"""
        if filename is None:
            filename = f"paper_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join("results", filename)
        os.makedirs("results", exist_ok=True)
        
        # Add metadata
        export_data = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'analyzer_version': '1.0.0',
                'total_papers': collection_analysis['total_papers']
            },
            'analysis_results': collection_analysis
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis exported to {filepath}")
        return filepath
