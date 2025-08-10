#!/usr/bin/env python3
"""
AI Researcher Free - Streamlit Mac Version
Streamlit web interface optimized for Mac
"""

import streamlit as st
import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure Streamlit page
st.set_page_config(
    page_title="AI Researcher Free - Mac Version",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🤖 AI Researcher Free - Mac Edition</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; margin-bottom: 2rem;">💰 100% Free AI Research Assistant | 🍎 Optimized for Mac</div>', unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.papers_processed = 0
    st.session_state.gaps_found = []
    st.session_state.hypotheses = []

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    research_query = st.text_input(
        "🔍 Research Query", 
        value="attention mechanisms transformer neural networks",
        help="Enter your research topic (e.g., 'computer vision', 'NLP transformers')"
    )
    
    max_papers = st.slider("📄 Max Papers", min_value=3, max_value=15, value=8, help="Number of papers to analyze")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Features")
    st.markdown("✅ Literature Analysis")
    st.markdown("✅ Research Gap Detection")
    st.markdown("✅ Hypothesis Generation")
    st.markdown("✅ Trend Visualization")
    st.markdown("✅ Result Export")
    
    st.markdown("---")
    
    st.markdown("### 💰 Cost")
    st.success("**$0.00** - Completely FREE!")
    
    st.markdown("### 🍎 Mac Optimized")
    st.info("Lightweight version without heavy ML dependencies")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📚 Literature Analysis", 
    "🔍 Research Gaps", 
    "💡 Hypotheses", 
    "📊 Analysis & Trends",
    "💾 Results & Export"
])

with tab1:
    st.header("📚 Literature Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Settings")
        st.write(f"**Query:** {research_query}")
        st.write(f"**Max Papers:** {max_papers}")
        st.write(f"**Source:** arXiv API")
    
    with col2:
        st.markdown("### Quick Stats")
        if st.session_state.papers_processed > 0:
            st.metric("Papers Processed", st.session_state.papers_processed)
            st.metric("Research Gaps", len(st.session_state.gaps_found))
            st.metric("Hypotheses", len(st.session_state.hypotheses))
        else:
            st.info("Click 'Start Analysis' to begin")
    
    st.markdown("---")
    
    if st.button("🚀 Start Literature Analysis", type="primary", use_container_width=True):
        with st.spinner("🔄 Fetching and analyzing papers from arXiv..."):
            # Import and run analysis
            try:
                from demo.mac_optimized_demo import MacOptimizedKnowledgeBase, MacOptimizedHypothesisGenerator
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Initialize knowledge base
                status_text.text("📚 Initializing knowledge base...")
                progress_bar.progress(20)
                kb = MacOptimizedKnowledgeBase()
                
                # Step 2: Fetch papers
                status_text.text("📥 Fetching papers from arXiv...")
                progress_bar.progress(40)
                result = kb.ingest_papers(research_query, max_papers=max_papers)
                
                # Step 3: Find gaps
                status_text.text("🔍 Identifying research gaps...")
                progress_bar.progress(60)
                gaps = kb.find_research_gaps()
                
                # Step 4: Generate hypotheses
                status_text.text("💡 Generating hypotheses...")
                progress_bar.progress(80)
                generator = MacOptimizedHypothesisGenerator()
                hypotheses = generator.generate_hypotheses(gaps[:5])
                
                # Step 5: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Update session state
                st.session_state.papers_processed = result['total_processed']
                st.session_state.gaps_found = gaps
                st.session_state.hypotheses = hypotheses
                st.session_state.initialized = True
                
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Successfully analyzed {result['total_processed']} papers!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 Try reducing the number of papers or check your internet connection")
    
    # Display recent results if available
    if st.session_state.initialized:
        st.markdown("### 📋 Recent Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Papers Analyzed", st.session_state.papers_processed)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Research Gaps", len(st.session_state.gaps_found))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Hypotheses Generated", len(st.session_state.hypotheses))
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("🔍 Research Gaps Identified")
    
    if st.session_state.gaps_found:
        st.markdown(f"### Found {len(st.session_state.gaps_found)} Potential Research Gaps")
        
        # Create DataFrame for gaps
        gaps_data = []
        for i, gap in enumerate(st.session_state.gaps_found[:10], 1):
            gaps_data.append({
                "Rank": i,
                "Concept": gap['concept'],
                "Category": gap['category'].replace('_', ' ').title(),
                "Confidence": f"{gap['confidence']:.2f}",
                "Frequency": gap['frequency'],
                "Description": gap['description'][:100] + "..." if len(gap['description']) > 100 else gap['description']
            })
        
        gaps_df = pd.DataFrame(gaps_data)
        
        # Display table
        st.dataframe(gaps_df, use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Gaps by Category")
            category_counts = pd.Series([gap['category'] for gap in st.session_state.gaps_found]).value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            category_counts.plot(kind='bar', ax=ax)
            ax.set_title('Research Gaps by Category')
            ax.set_xlabel('Category')
            ax.set_ylabel('Number of Gaps')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 🎯 Confidence Distribution")
            confidences = [gap['confidence'] for gap in st.session_state.gaps_found]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(confidences, bins=10, alpha=0.7, color='skyblue')
            ax.set_title('Research Gap Confidence Distribution')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Number of Gaps')
            st.pyplot(fig)
        
    else:
        st.info("🔍 Run the literature analysis first to identify research gaps")
        st.markdown("Navigate to the **Literature Analysis** tab and click **Start Analysis**")

with tab3:
    st.header("💡 Research Hypotheses")
    
    if st.session_state.hypotheses:
        st.markdown(f"### Generated {len(st.session_state.hypotheses)} Research Hypotheses")
        
        for i, (hypothesis, confidence) in enumerate(st.session_state.hypotheses, 1):
            with st.expander(f"💡 Hypothesis {i} (Confidence: {confidence:.2f})"):
                st.write(hypothesis)
                st.markdown(f"**Confidence Score:** {confidence:.2f}")
                
                # Add action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"👍 Like", key=f"like_{i}"):
                        st.success("Hypothesis marked as promising!")
                with col2:
                    if st.button(f"📚 Research More", key=f"research_{i}"):
                        st.info("This would open detailed research suggestions")
                with col3:
                    if st.button(f"🧪 Design Experiment", key=f"experiment_{i}"):
                        st.info("This would help design experiments to test this hypothesis")
        
        # Best hypothesis highlight
        if st.session_state.hypotheses:
            best_hypothesis, best_confidence = st.session_state.hypotheses[0]
            st.markdown("### 🌟 Most Promising Hypothesis")
            st.markdown('<div class="feature-box">', unsafe_allow_html=True)
            st.markdown(f"**{best_hypothesis}**")
            st.markdown(f"*Confidence: {best_confidence:.2f}*")
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("💡 Run the literature analysis first to generate hypotheses")

with tab4:
    st.header("📊 Analysis & Trends")
    
    if st.session_state.initialized:
        st.markdown("### 🔍 Research Landscape Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Concept Frequency")
            # Mock data for demonstration
            concepts = ['attention', 'transformer', 'neural networks', 'computer vision', 'deep learning']
            frequencies = [8, 6, 5, 3, 4]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(concepts, frequencies, color='lightcoral')
            ax.set_title('Most Frequent Research Concepts')
            ax.set_xlabel('Frequency')
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🎯 Research Maturity")
            maturity_data = {
                'Emerging': len([g for g in st.session_state.gaps_found if g['confidence'] > 0.7]),
                'Developing': len([g for g in st.session_state.gaps_found if 0.4 < g['confidence'] <= 0.7]),
                'Mature': len([g for g in st.session_state.gaps_found if g['confidence'] <= 0.4])
            }
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(maturity_data.values(), labels=maturity_data.keys(), autopct='%1.1f%%')
            ax.set_title('Research Area Maturity')
            st.pyplot(fig)
        
        # Research timeline (mock)
        st.markdown("### 📅 Research Timeline")
        timeline_data = pd.DataFrame({
            'Year': [2020, 2021, 2022, 2023, 2024],
            'Papers': [2, 3, 1, 1, 1],
            'Breakthrough': [1, 0, 1, 0, 0]
        })
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(timeline_data['Year'], timeline_data['Papers'], marker='o', label='Papers')
        ax.bar(timeline_data['Year'], timeline_data['Breakthrough'], alpha=0.5, label='Breakthroughs')
        ax.set_title('Research Activity Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)
        
    else:
        st.info("📊 Run the analysis first to see trends and patterns")

with tab5:
    st.header("💾 Results & Export")
    
    if st.session_state.initialized:
        st.markdown("### 📁 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export Hypotheses (JSON)", use_container_width=True):
                hypotheses_data = [
                    {"hypothesis": h, "confidence": c} 
                    for h, c in st.session_state.hypotheses
                ]
                st.download_button(
                    label="💾 Download Hypotheses",
                    data=json.dumps(hypotheses_data, indent=2),
                    file_name=f"hypotheses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export Research Gaps (CSV)", use_container_width=True):
                gaps_df = pd.DataFrame(st.session_state.gaps_found)
                csv = gaps_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Research Gaps",
                    data=csv,
                    file_name=f"research_gaps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.markdown("### 📋 Session Summary")
        
        summary = {
            "Analysis Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Research Query": research_query,
            "Papers Analyzed": st.session_state.papers_processed,
            "Research Gaps Found": len(st.session_state.gaps_found),
            "Hypotheses Generated": len(st.session_state.hypotheses),
            "Best Hypothesis Confidence": f"{st.session_state.hypotheses[0][1]:.2f}" if st.session_state.hypotheses else "N/A",
            "Total Cost": "$0.00 (FREE)"
        }
        
        for key, value in summary.items():
            st.write(f"**{key}:** {value}")
        
        # Download summary
        if st.button("📋 Export Summary (JSON)", use_container_width=True):
            st.download_button(
                label="💾 Download Summary",
                data=json.dumps(summary, indent=2),
                file_name=f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
    else:
        st.info("💾 Complete the analysis to export results")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem;'>
        🤖 <strong>AI Researcher Free</strong> - Mac Optimized Version<br>
        💰 100% Free • 🍎 Mac Compatible • 🚀 No API Keys Required<br>
        <em>Built with ❤️ for the research community</em>
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar footer
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🆘 Need Help?")
    st.markdown("🔧 Check SETUP_MAC.md")
    st.markdown("📚 Read USAGE.md")
    st.markdown("💡 Try minimal demo")
    
    if st.button("🔄 Reset Session"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
