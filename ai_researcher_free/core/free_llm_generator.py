"""
Free LLM-based Hypothesis Generator
Uses free LLM services and local models when available
"""

import requests
import json
import time
import logging
from typing import List, Tuple, Dict, Any, Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class FreeLLMGenerator:
    """
    Generate hypotheses using free LLM services
    Supports multiple free providers with automatic fallback
    """
    
    def __init__(self, huggingface_token: str = None):
        self.huggingface_token = huggingface_token or os.getenv('HUGGINGFACE_API_KEY')
        self.providers = self._setup_providers()
        self.current_provider = 'template'  # Start with template fallback
        self.rate_limits = {
            'huggingface': {'calls_per_hour': 100, 'last_call': 0, 'call_count': 0},
            'ollama': {'calls_per_hour': 1000, 'last_call': 0, 'call_count': 0}
        }
        
    def _setup_providers(self) -> Dict[str, Dict[str, Any]]:
        """Setup free LLM providers"""
        providers = {
            'huggingface': {
                'available': bool(self.huggingface_token),
                'models': [
                    'microsoft/DialoGPT-large',
                    'google/flan-t5-large',
                    'facebook/blenderbot-400M-distill'
                ],
                'url_template': 'https://api-inference.huggingface.co/models/{model}',
                'free_limit': 30000  # requests per month
            },
            'ollama': {
                'available': self._check_ollama_available(),
                'models': ['llama2:7b', 'mistral:7b', 'codellama:7b'],
                'url': 'http://localhost:11434/api/generate',
                'free_limit': float('inf')  # Unlimited local usage
            },
            'template': {
                'available': True,  # Always available
                'description': 'Template-based generation (fallback)',
                'free_limit': float('inf')
            }
        }
        
        logger.info(f"Available providers: {[k for k, v in providers.items() if v['available']]}")
        return providers
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running locally"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def generate_hypotheses(self, research_gaps: List[Dict[str, Any]], 
                          context_papers: List[Dict[str, Any]] = None,
                          num_hypotheses: int = 5) -> List[Tuple[str, float]]:
        """
        Generate hypotheses using available free LLM services
        
        Args:
            research_gaps: List of identified research gaps
            context_papers: Related papers for context (optional)
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            List of (hypothesis, confidence_score) tuples
        """
        hypotheses = []
        
        # Try different providers in order of preference
        providers_to_try = ['ollama', 'huggingface', 'template']
        
        for gap in research_gaps[:num_hypotheses]:
            hypothesis = None
            confidence = 0.5
            
            for provider in providers_to_try:
                if not self.providers[provider]['available']:
                    continue
                
                if not self._check_rate_limit(provider):
                    logger.warning(f"Rate limit exceeded for {provider}")
                    continue
                
                try:
                    if provider == 'ollama':
                        hypothesis = self._generate_with_ollama(gap, context_papers)
                        confidence = 0.8
                    elif provider == 'huggingface':
                        hypothesis = self._generate_with_huggingface(gap, context_papers)
                        confidence = 0.7
                    elif provider == 'template':
                        from .template_generator import TemplateHypothesisGenerator
                        template_gen = TemplateHypothesisGenerator()
                        result = template_gen.generate_hypotheses([gap], 1)
                        if result:
                            hypothesis, confidence = result[0]
                    
                    if hypothesis:
                        self._update_rate_limit(provider)
                        break
                        
                except Exception as e:
                    logger.warning(f"Provider {provider} failed: {e}")
                    continue
            
            if hypothesis:
                hypotheses.append((hypothesis, confidence))
            else:
                # Ultimate fallback
                fallback_hypothesis = self._generate_fallback_hypothesis(gap)
                hypotheses.append((fallback_hypothesis, 0.4))
        
        return sorted(hypotheses, key=lambda x: x[1], reverse=True)
    
    def _generate_with_ollama(self, gap: Dict[str, Any], 
                             context_papers: List[Dict[str, Any]] = None) -> Optional[str]:
        """Generate hypothesis using local Ollama"""
        concept = gap.get('concept', 'unknown concept')
        category = gap.get('category', 'machine learning')
        
        # Prepare context
        context = self._prepare_context(context_papers) if context_papers else ""
        
        prompt = f"""You are a research scientist generating hypotheses for machine learning research.

Research Gap: {concept}
Domain: {category}
Context: {context[:300]}

Generate a specific, testable hypothesis that:
1. Addresses this research gap
2. Is technically feasible
3. Can be evaluated with clear metrics
4. Is novel and not extensively studied

Format: One clear hypothesis sentence.

Hypothesis:"""
        
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama2:7b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'max_tokens': 150
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                hypothesis = result.get('response', '').strip()
                return self._clean_hypothesis(hypothesis)
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
        
        return None
    
    def _generate_with_huggingface(self, gap: Dict[str, Any], 
                                  context_papers: List[Dict[str, Any]] = None) -> Optional[str]:
        """Generate hypothesis using Hugging Face Inference API"""
        if not self.huggingface_token:
            return None
        
        concept = gap.get('concept', 'unknown concept')
        category = gap.get('category', 'machine learning')
        
        # Use a free text generation model
        model = 'google/flan-t5-large'
        url = f'https://api-inference.huggingface.co/models/{model}'
        
        prompt = f"Generate a research hypothesis for {concept} in {category}: "
        
        headers = {'Authorization': f'Bearer {self.huggingface_token}'}
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json={
                    'inputs': prompt,
                    'parameters': {
                        'max_new_tokens': 100,
                        'temperature': 0.7,
                        'do_sample': True
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    hypothesis = result[0].get('generated_text', '')
                    return self._clean_hypothesis(hypothesis)
                elif isinstance(result, dict):
                    hypothesis = result.get('generated_text', '')
                    return self._clean_hypothesis(hypothesis)
            else:
                logger.warning(f"Hugging Face API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
        
        return None
    
    def _prepare_context(self, context_papers: List[Dict[str, Any]]) -> str:
        """Prepare context from related papers"""
        if not context_papers:
            return ""
        
        context_parts = []
        for paper in context_papers[:3]:  # Limit context length
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            context_parts.append(f"{title}: {abstract[:100]}...")
        
        return " ".join(context_parts)
    
    def _clean_hypothesis(self, hypothesis: str) -> str:
        """Clean and format generated hypothesis"""
        # Remove common prefixes
        prefixes_to_remove = [
            'hypothesis:', 'research hypothesis:', 'generated hypothesis:',
            'the hypothesis is:', 'a hypothesis could be:'
        ]
        
        hypothesis_lower = hypothesis.lower().strip()
        for prefix in prefixes_to_remove:
            if hypothesis_lower.startswith(prefix):
                hypothesis = hypothesis[len(prefix):].strip()
                break
        
        # Capitalize first letter
        if hypothesis:
            hypothesis = hypothesis[0].upper() + hypothesis[1:]
        
        # Ensure it ends with a period
        if hypothesis and not hypothesis.endswith('.'):
            hypothesis += '.'
        
        # Remove extra whitespace
        hypothesis = ' '.join(hypothesis.split())
        
        # Validate hypothesis quality
        if len(hypothesis.split()) < 5:
            return None  # Too short
        
        if len(hypothesis) > 500:
            hypothesis = hypothesis[:500] + '.'  # Too long
        
        return hypothesis
    
    def _generate_fallback_hypothesis(self, gap: Dict[str, Any]) -> str:
        """Generate fallback hypothesis when all LLM providers fail"""
        concept = gap.get('concept', 'novel approach')
        category = gap.get('category', 'machine learning')
        
        templates = {
            'computer_vision': f"A novel {concept} approach could improve visual recognition tasks by addressing current limitations in feature representation.",
            'natural_language_processing': f"Incorporating {concept} into language models may enhance text understanding and generation capabilities.",
            'attention_mechanisms': f"A {concept}-based attention mechanism could capture more relevant dependencies in sequential data.",
            'neural_networks': f"Integrating {concept} into neural architectures may improve model performance and efficiency.",
            'general_ml': f"A {concept}-based approach could advance machine learning by providing better solutions to existing challenges."
        }
        
        return templates.get(category, templates['general_ml'])
    
    def _check_rate_limit(self, provider: str) -> bool:
        """Check if provider is within rate limits"""
        if provider not in self.rate_limits:
            return True
        
        limit_info = self.rate_limits[provider]
        current_time = time.time()
        
        # Reset counter if hour has passed
        if current_time - limit_info['last_call'] > 3600:
            limit_info['call_count'] = 0
            limit_info['last_call'] = current_time
        
        return limit_info['call_count'] < limit_info['calls_per_hour']
    
    def _update_rate_limit(self, provider: str):
        """Update rate limit counter"""
        if provider in self.rate_limits:
            self.rate_limits[provider]['call_count'] += 1
            self.rate_limits[provider]['last_call'] = time.time()
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers"""
        status = {}
        
        for provider, config in self.providers.items():
            status[provider] = {
                'available': config['available'],
                'description': config.get('description', f'{provider} API'),
                'rate_limit_status': self.rate_limits.get(provider, {})
            }
            
            if provider == 'huggingface':
                status[provider]['has_token'] = bool(self.huggingface_token)
            elif provider == 'ollama':
                status[provider]['local_server'] = self._check_ollama_available()
        
        return status
    
    def test_providers(self) -> Dict[str, bool]:
        """Test all available providers"""
        test_gap = {
            'concept': 'attention mechanisms',
            'category': 'neural_networks',
            'description': 'Test gap for provider testing'
        }
        
        results = {}
        
        for provider in ['ollama', 'huggingface', 'template']:
            try:
                if provider == 'ollama' and self.providers['ollama']['available']:
                    result = self._generate_with_ollama(test_gap)
                    results[provider] = bool(result)
                elif provider == 'huggingface' and self.providers['huggingface']['available']:
                    result = self._generate_with_huggingface(test_gap)
                    results[provider] = bool(result)
                elif provider == 'template':
                    result = self._generate_fallback_hypothesis(test_gap)
                    results[provider] = bool(result)
                else:
                    results[provider] = False
            except Exception as e:
                logger.error(f"Provider {provider} test failed: {e}")
                results[provider] = False
        
        return results
    
    def generate_experiment_ideas(self, hypothesis: str) -> List[str]:
        """Generate experiment ideas for testing a hypothesis"""
        if self.providers['ollama']['available'] and self._check_rate_limit('ollama'):
            return self._generate_experiment_ideas_llm(hypothesis)
        else:
            return self._generate_experiment_ideas_template(hypothesis)
    
    def _generate_experiment_ideas_llm(self, hypothesis: str) -> List[str]:
        """Generate experiment ideas using LLM"""
        prompt = f"""Given this research hypothesis: "{hypothesis}"

Suggest 5 specific experiments to test this hypothesis. Each experiment should:
1. Be technically feasible
2. Have clear success metrics
3. Use standard datasets when possible
4. Be completable with reasonable resources

Format as a numbered list:

Experiments:"""
        
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama2:7b',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.5, 'max_tokens': 300}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '')
                
                # Extract numbered items
                import re
                experiments = re.findall(r'\d+\.\s*([^0-9]+?)(?=\d+\.|$)', text, re.DOTALL)
                return [exp.strip() for exp in experiments[:5]]
                
        except Exception as e:
            logger.error(f"Experiment idea generation failed: {e}")
        
        return self._generate_experiment_ideas_template(hypothesis)
    
    def _generate_experiment_ideas_template(self, hypothesis: str) -> List[str]:
        """Generate experiment ideas using templates"""
        experiments = [
            f"Implement the proposed method and compare against baseline approaches on standard benchmarks",
            f"Conduct ablation studies to understand the contribution of each component",
            f"Evaluate performance across different dataset sizes and complexity levels",
            f"Test robustness under various noise conditions and domain shifts",
            f"Analyze computational efficiency and memory requirements compared to existing methods"
        ]
        
        return experiments
