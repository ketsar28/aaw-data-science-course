# Contributing to Data Science Zero to Hero

Terima kasih atas ketertarikan Anda untuk berkontribusi ke **Data Science Zero to Hero**! 🎉

Kami menyambut kontribusi dari siapa saja - baik Anda pemula yang baru belajar atau expert yang ingin berbagi pengetahuan.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Contribution Guidelines](#contribution-guidelines)
- [Style Guide](#style-guide)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)

## 🤝 Code of Conduct

Dengan berpartisipasi dalam project ini, Anda setuju untuk menjaga lingkungan yang:
- **Respectful**: Hormati semua kontributor terlepas dari level pengalaman
- **Inclusive**: Sambut kontributor dari berbagai background
- **Collaborative**: Bekerja sama dengan semangat positif
- **Constructive**: Berikan feedback yang membangun

## 💡 How Can I Contribute?

### 🐛 Reporting Bugs

Jika Anda menemukan bug atau error:

1. **Check existing issues** - Pastikan bug belum dilaporkan
2. **Create detailed issue** dengan informasi:
   - Deskripsi jelas tentang bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots jika relevan
   - Environment (OS, Python version, dll.)

### 💡 Suggesting Enhancements

Punya ide untuk improvement?

1. **Check existing issues/discussions**
2. **Create new issue** dengan tag `enhancement`
3. **Describe**:
   - Current limitation
   - Proposed solution
   - Why it would be valuable
   - Alternative solutions considered

### 📝 Improving Documentation

Documentation improvements sangat diapresiasi!

- Fix typos
- Clarify confusing explanations
- Add more examples
- Improve visualizations
- Translate content

### 🔧 Contributing Code

Types of code contributions:
- **Bug fixes**
- **New examples/exercises**
- **New datasets**
- **Utility functions**
- **Improved implementations**
- **New modules** (discuss first!)

## 🚀 Getting Started

### 1. Fork & Clone

```bash
# Fork repository di GitHub
# Clone your fork
git clone https://github.com/YOUR-USERNAME/aaw-data-science-course.git
cd aaw-data-science-course
```

### 2. Setup Development Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ds-zero-hero

# Or use venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 3. Create Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📐 Contribution Guidelines

### For Jupyter Notebooks

1. **Clear all outputs** before committing (unless outputs are essential)
2. **Follow notebook structure**:
   ```
   # Header
   # Learning Objectives
   # Prerequisites
   # Imports
   # Content sections
   # Summary
   # References
   ```
3. **Add markdown cells** with explanations
4. **Use section headers** (#, ##, ###)
5. **Include visualizations** where helpful
6. **Test all cells** - ensure they run without errors

### For Python Scripts

1. **Follow PEP 8** style guide
2. **Add docstrings** (Google or NumPy style)
3. **Include type hints**
4. **Write tests** for new functions
5. **Add comments** for complex logic

Example:
```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics for regression.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing MAE, MSE, RMSE, R2
        
    Example:
        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> metrics = calculate_metrics(y_true, y_pred)
    """
    # Implementation
    pass
```

### For Documentation

1. **Use clear, simple language**
2. **Include examples**
3. **Add analogies** untuk konsep kompleks
4. **Provide context** (why, not just how)
5. **Check spelling & grammar**

### For Datasets

1. **Use publicly available data** atau synthetic data
2. **Include data description** (README.md in dataset folder)
3. **Specify license**
4. **Keep file size reasonable** (<10MB if possible)
5. **Provide data dictionary**

## 🎨 Style Guide

### Python Code Style

```python
# Use meaningful variable names
# Bad
x = df[df['a'] > 5]

# Good
high_value_customers = customers_df[customers_df['revenue'] > 5000]

# Add type hints
def preprocess_data(df: pd.DataFrame, target_col: str) -> tuple:
    """Preprocess dataframe for modeling."""
    pass

# Document parameters
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'random_forest',
    **kwargs
) -> object:
    """
    Train machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model ('random_forest', 'xgboost', etc.)
        **kwargs: Additional arguments for model
        
    Returns:
        Trained model object
    """
    pass
```

### Markdown Style

```markdown
# Main Header (H1)

## Section Header (H2)

### Subsection Header (H3)

**Bold for emphasis**
*Italic for technical terms*

- Bullet points for lists
- Keep items parallel

1. Numbered lists for steps
2. Sequential order matters

`inline code` for short snippets

```python
# Code blocks for longer snippets
def example():
    pass
```

> Blockquotes for important notes

| Column 1 | Column 2 |
|----------|----------|
| Data     | Data     |
```

## 📝 Commit Messages

Follow conventional commits:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(module-16): add Ridge regression example

docs(readme): update installation instructions

fix(notebook): correct formula in statistics module

style(code): format with black
```

## 🔄 Pull Request Process

### 1. Before Creating PR

- [ ] Code follows style guide
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Notebooks run without errors
- [ ] Commit messages are clear

### 2. Create Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**:
   - Clear title following commit conventions
   - Detailed description:
     - What changes were made
     - Why changes were needed
     - How to test
     - Screenshots if UI changes

3. **Link related issues**:
   ```
   Closes #123
   Related to #456
   ```

### 3. PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How to test these changes:
1. Step 1
2. Step 2

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added/updated
- [ ] All tests pass
```

### 4. Review Process

- Maintainer akan review PR Anda
- Mungkin ada request for changes
- Address feedback dengan commits baru
- Once approved, PR akan di-merge

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_preprocessing.py

# With coverage
pytest --cov=src tests/
```

## 📚 Additional Resources

- [Python Style Guide (PEP 8)](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [How to Write Good Commit Messages](https://chris.beams.io/posts/git-commit/)

## ❓ Questions?

- Check [FAQ](docs/faq.md)
- Search [GitHub Discussions](https://github.com/ketsar28/aaw-data-science-course/discussions)
- Create new discussion
- Reach out to maintainers

## 🙏 Recognition

All contributors will be recognized in:
- README.md contributors section
- Release notes
- Special mentions for significant contributions

---

**Thank you for contributing to Data Science Zero to Hero!** 🚀

Your contributions help thousands of aspiring data scientists learn and grow.
