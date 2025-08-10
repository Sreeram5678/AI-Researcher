#!/usr/bin/env python3
"""
AI Researcher by Sreeram - Demo Launcher
Easy way to launch different interfaces
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🤖 AI RESEARCHER BY SREERAM - DEMO LAUNCHER")
    print("💰 100% Free AI Research Assistant")
    print("=" * 60)
    print()

def check_requirements():
    """Check if requirements are installed"""
    try:
        import torch
        import streamlit
        import gradio
        import transformers
        print("✅ All requirements are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing requirements: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def launch_streamlit():
    """Launch Streamlit interface"""
    print("🌐 Launching Streamlit web interface...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    print()
    
    demo_path = Path(__file__).parent / "demo" / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(demo_path)])

def launch_gradio():
    """Launch Gradio interface"""
    print("🎨 Launching Gradio web interface...")
    print("📱 Open your browser to: http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop")
    print()
    
    demo_path = Path(__file__).parent / "demo" / "gradio_app.py"
    subprocess.run([sys.executable, str(demo_path)])

def launch_jupyter():
    """Launch Jupyter notebook"""
    print("📓 Launching Jupyter notebook...")
    print("📚 Opening demo/jupyter_demo.ipynb")
    print("⏹️  Press Ctrl+C to stop")
    print()
    
    demo_path = Path(__file__).parent / "demo" / "jupyter_demo.ipynb"
    subprocess.run([sys.executable, "-m", "jupyter", "notebook", str(demo_path)])

def run_quick_demo():
    """Run quick command-line demo"""
    print("⚡ Running quick command-line demo...")
    print("📊 This will demonstrate the complete workflow")
    print()
    
    demo_path = Path(__file__).parent / "demo" / "quick_start.py"
    subprocess.run([sys.executable, str(demo_path)])

def show_help():
    """Show available options"""
    print("🎯 Available Demo Options:")
    print()
    print("1️⃣  streamlit  - Web interface (recommended)")
    print("2️⃣  gradio     - Alternative web interface")  
    print("3️⃣  jupyter    - Interactive notebook")
    print("4️⃣  quick      - Command-line demo")
    print("5️⃣  help       - Show this help")
    print()
    print("🚀 Usage Examples:")
    print("  python run_demo.py streamlit")
    print("  python run_demo.py quick")
    print("  python run_demo.py --help")
    print()

def main():
    """Main demo launcher"""
    parser = argparse.ArgumentParser(
        description="AI Researcher by Sreeram - Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_demo.py streamlit    # Launch web interface
  python run_demo.py quick        # Run command-line demo
  python run_demo.py jupyter      # Open Jupyter notebook
  python run_demo.py gradio       # Alternative web interface
        """
    )
    
    parser.add_argument(
        "interface",
        nargs="?",
        default="help",
        choices=["streamlit", "gradio", "jupyter", "quick", "help"],
        help="Interface to launch"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if dependencies are installed"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check dependencies if requested
    if args.check_deps:
        check_requirements()
        return
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Launch appropriate interface
    if args.interface == "streamlit":
        if check_requirements():
            launch_streamlit()
    elif args.interface == "gradio":
        if check_requirements():
            launch_gradio()
    elif args.interface == "jupyter":
        if check_requirements():
            launch_jupyter()
    elif args.interface == "quick":
        run_quick_demo()
    elif args.interface == "help":
        show_help()
    
    print("\n👋 Thanks for trying AI Researcher by Sreeram!")
    print("⭐ Star us on GitHub if you found this useful!")

if __name__ == "__main__":
    main()
