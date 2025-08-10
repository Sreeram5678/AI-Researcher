# Contributing to AI Researcher

Thank you for your interest in contributing to AI Researcher! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### **Types of Contributions We Welcome**

- 🐛 **Bug Reports**: Help us identify and fix issues
- 💡 **Feature Requests**: Suggest new features or improvements
- 📚 **Documentation**: Improve docs, add examples, fix typos
- 🔧 **Code Contributions**: Submit pull requests with improvements
- 🧪 **Testing**: Test features and report issues
- 🌍 **Localization**: Help translate to other languages
- 📖 **Tutorials**: Create guides and examples

### **Getting Started**

1. **Fork the Repository**
   ```bash
   git clone https://github.com/Sreeram5678/AI-Researcher.git
   cd AI-Researcher
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make Your Changes**
   - Follow the coding style guidelines
   - Add tests if applicable
   - Update documentation

4. **Test Your Changes**
   ```bash
   # Run the demo
   python demo/quick_start.py
   
   # Run web interface
   streamlit run demo/streamlit_app.py
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Describe your changes clearly

## 📋 Pull Request Guidelines

### **Before Submitting**

- ✅ **Test thoroughly** - Ensure your changes work
- ✅ **Update documentation** - Keep docs in sync with code
- ✅ **Follow style guidelines** - Maintain code consistency
- ✅ **Add tests** - Include tests for new features
- ✅ **Check compatibility** - Ensure it works on different platforms

### **PR Description Template**

```markdown
## Description
Brief description of what this PR accomplishes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tested locally
- [ ] Added unit tests
- [ ] Updated documentation

## Screenshots (if applicable)
Add screenshots for UI changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🎯 Development Guidelines

### **Code Style**

- **Python**: Follow PEP 8 guidelines
- **Comments**: Use clear, descriptive comments
- **Naming**: Use descriptive variable and function names
- **Documentation**: Include docstrings for functions and classes

### **File Structure**

```
ai_researcher_free/
├── core/           # Core functionality
├── demo/           # Demo applications
├── data/           # Data files
├── models/         # Model files
├── results/        # Output results
└── tests/          # Test files
```

### **Testing**

- Write tests for new features
- Ensure existing tests pass
- Use pytest for testing framework
- Aim for good test coverage

## 🚀 Quick Development Setup

### **Local Development**

```bash
# Clone and setup
git clone https://github.com/Sreeram5678/AI-Researcher.git
cd AI-Researcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### **Running Tests**

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest

# Run with coverage
pytest --cov=ai_researcher_free
```

## 📞 Getting Help

### **Communication Channels**

- 💬 **GitHub Issues**: [Report bugs and request features](https://github.com/Sreeram5678/AI-Researcher/issues)
- 📧 **Email**: sreeram.lagisetty@gmail.com
- 🐛 **Bug Reports**: Use GitHub Issues with detailed information
- 💡 **Feature Requests**: Open a GitHub Issue with the "enhancement" label

### **Issue Templates**

When reporting issues, please include:
- **Environment**: OS, Python version, dependencies
- **Steps to reproduce**: Clear, step-by-step instructions
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Screenshots**: If applicable
- **Error messages**: Full error traceback

## 🏆 Recognition

### **Contributor Hall of Fame**

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation
- Community acknowledgments

### **Types of Recognition**

- 🌟 **Top Contributors**: Most active contributors
- 🚀 **Feature Creators**: Contributors of major features
- 🐛 **Bug Hunters**: Contributors who fix critical bugs
- 📚 **Documentation Heroes**: Contributors who improve docs

## 📄 License

By contributing to AI Researcher, you agree that your contributions will be subject to the project's licensing terms. For licensing information, please contact Sreeram at sreeram.lagisetty@gmail.com.

## 🙏 Thank You

Thank you for contributing to AI Researcher! Your contributions help make AI research more accessible to everyone.

---

**Happy Contributing! 🚀**

For more information, visit our [main repository](https://github.com/Sreeram5678/AI-Researcher).
