"""
AI Researcher Free - Core Module
100% Free AI Research Assistant

This module contains the core components for automated AI research
using only free resources and APIs.
"""

from .free_knowledge_base import FreeKnowledgeBase
from .template_generator import TemplateHypothesisGenerator
from .free_llm_generator import FreeLLMGenerator
from .experiment_runner import FreeExperimentRunner
from .paper_analyzer import PaperAnalyzer

__version__ = "1.0.0"
__author__ = "AI Researcher Free Team"

__all__ = [
    'FreeKnowledgeBase',
    'TemplateHypothesisGenerator', 
    'FreeLLMGenerator',
    'FreeExperimentRunner',
    'PaperAnalyzer'
]
