# GitLab Dev - AI-Powered Development Assistant

GitLab Dev is a sophisticated AI-powered development assistant that integrates GitLab with multiple LLM providers through the Model Context Protocol (MCP). It provides an intelligent coding agent that can autonomously work on GitLab projects.

## Features

- **Multi-Provider LLM Integration**: Support for Gemini, OpenAI, and local Ollama models
- **GitLab Deep Integration**: Comprehensive API coverage for projects, issues, MRs, and pipelines
- **Automated Coding Agent**: End-to-end development workflow automation
- **Model Context Protocol**: Standards-based AI-tool communication
- **Web Interface**: Interactive chat interface with project management
- **Clean Architecture**: Domain-driven design with hexagonal architecture

## Quick Start

### Prerequisites

- Python 3.9+
- GitLab instance with API access
- LLM provider API keys (Gemini, OpenAI, or local Ollama)

### Installation

#### Using pip (recommended)
```bash
pip install -e .
```

#### Development setup
```bash
# Clone the repository
git clone <repository-url>
cd GitLabDev

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

Create a `.env` file with your configuration:

```env
# GitLab Configuration
GITLAB_URL=https://your-gitlab-instance.com
GITLAB_TOKEN=your-gitlab-token

# LLM Provider (choose one)
LLM_PROVIDER=gemini  # or openai, ollama
GEMINI_API_KEY=your-gemini-api-key

# Optional: OpenAI
OPENAI_API_KEY=your-openai-api-key

# Optional: Local Ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

### Running the Application

#### New way (using the package)
```bash
gitlab-dev
```

#### Legacy way (during transition)
```bash
python3 app/main.py
```

The application will be available at `http://localhost:8000`.

## Architecture

This project follows clean architecture principles with domain-driven design:

```
src/gitlab_dev/
├── core/              # Configuration and cross-cutting concerns
├── domain/            # Pure business logic
│   ├── entities/      # Domain entities
│   ├── services/      # Domain services
│   └── ports/         # Abstract interfaces
├── infrastructure/    # External system integrations
│   ├── gitlab/        # GitLab API integration
│   ├── llm/          # LLM provider adapters
│   └── mcp/          # MCP integration
├── application/       # Use cases and orchestration
└── interfaces/        # API and UI adapters
```

## Development

### Project Structure

The project has been restructured to follow modern Python packaging standards:

- **src/ layout**: Industry standard for Python packages
- **pyproject.toml**: Modern dependency management
- **Clean Architecture**: Separation of concerns with hexagonal architecture
- **Type Safety**: Full type annotations throughout
- **Modern Tooling**: Ruff, Black, MyPy for code quality

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m e2e
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type check
mypy src
```

## Migration Status

This project is currently undergoing a refactoring from a simple structure to a clean architecture. During the transition:

- ✅ New project structure created
- ✅ Modern packaging with pyproject.toml
- ✅ Domain entities and ports defined
- ✅ Infrastructure layer started
- 🔄 Maintaining backward compatibility
- ⏳ Gradual migration of components

The system continues to work exactly as before during this transition.

## Contributing

1. Follow the clean architecture principles
2. Maintain type safety with full annotations
3. Add tests for new functionality
4. Use the established code style (Black + Ruff)
5. Update documentation as needed

## License

MIT License - see LICENSE file for details.