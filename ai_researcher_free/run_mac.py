#!/usr/bin/env python3
"""
AI Researcher Free - Mac Launcher
Easy launcher for Mac users with dependency issues
"""

import sys
import os
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🤖 AI RESEARCHER FREE - MAC LAUNCHER")
    print("🍎 Optimized for Mac (no heavy ML dependencies)")
    print("💰 100% Free AI Research Assistant")
    print("=" * 60)
    print()

def check_basic_requirements():
    """Check basic requirements"""
    try:
        import requests
        import matplotlib
        import pandas
        print("✅ Basic requirements satisfied")
        return True
    except ImportError as e:
        print(f"❌ Missing basic requirement: {e}")
        print("💡 Run: pip install requests matplotlib pandas")
        return False

def main():
    """Main launcher"""
    print_banner()
    
    print("🔍 Available options:")
    print("1️⃣  Mac Optimized Demo (recommended for Mac)")
    print("2️⃣  Minimal Demo (basic functionality)")
    print("3️⃣  Check dependencies")
    print("4️⃣  Help & troubleshooting")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Launching Mac Optimized Demo...")
        if check_basic_requirements():
            demo_path = Path(__file__).parent / "demo" / "mac_optimized_demo.py"
            subprocess.run([sys.executable, str(demo_path)])
        else:
            print("❌ Please install missing requirements first")
    
    elif choice == "2":
        print("\n⚡ Launching Minimal Demo...")
        demo_path = Path(__file__).parent / "demo" / "minimal_demo.py"
        subprocess.run([sys.executable, str(demo_path)])
    
    elif choice == "3":
        print("\n🔍 Checking dependencies...")
        check_basic_requirements()
        
        print("\n📦 Checking optional dependencies:")
        optional_deps = [
            ("torch", "PyTorch for ML experiments"),
            ("transformers", "Hugging Face transformers"),
            ("sentence_transformers", "Sentence embeddings"),
            ("streamlit", "Web interface"),
            ("gradio", "Alternative web interface")
        ]
        
        for dep, description in optional_deps:
            try:
                __import__(dep)
                print(f"✅ {dep}: Available ({description})")
            except ImportError:
                print(f"❌ {dep}: Not available ({description})")
        
        print("\n💡 To install all dependencies:")
        print("pip install -r requirements.txt")
    
    elif choice == "4":
        print("\n🆘 Help & Troubleshooting:")
        print()
        print("🔧 Common Issues:")
        print("• ModuleNotFoundError: Install missing packages")
        print("• _lzma error: Run 'brew install xz'")
        print("• PyTorch issues: Use CPU version or skip ML features")
        print()
        print("🎯 Recommended workflow for Mac:")
        print("1. Start with Mac Optimized Demo (option 1)")
        print("2. This uses arXiv API + basic analysis")
        print("3. No heavy ML dependencies required")
        print("4. Works great for research exploration")
        print()
        print("📚 For full functionality:")
        print("• Use Google Colab for ML experiments")
        print("• Install Ollama for local LLM")
        print("• Try the web interface: streamlit run demo/streamlit_app.py")
        print()
        print("🔗 Resources:")
        print("• README.md: Complete documentation")
        print("• USAGE.md: Detailed usage guide")
        print("• demo/ folder: All available interfaces")
    
    else:
        print("❌ Invalid choice. Please select 1-4.")
    
    print("\n👋 Thanks for using AI Researcher Free!")

if __name__ == "__main__":
    main()
