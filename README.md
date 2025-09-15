# TOEFL/IELTS AI Teaching Platform

A multimodal LLM platform for teaching TOEFL/IELTS skills including grammar, reading, writing, and speaking.

## Project Structure

```
toefl-ielts-ai/
├── data/                   # Training data and datasets
│   ├── grammar/           # Grammar exercises and corrections
│   ├── reading/           # Reading comprehension passages
│   ├── writing/           # Writing prompts and samples
│   └── speaking/          # Speaking practice materials
├── models/                # Fine-tuned models
│   ├── grammar_model/     # Grammar correction model
│   ├── reading_model/     # Reading comprehension model
│   └── writing_model/     # Writing assessment model
├── backend/               # FastAPI backend
│   ├── api/              # API endpoints
│   ├── services/         # Business logic
│   └── utils/            # Utility functions
├── frontend/              # Next.js frontend
│   ├── components/       # React components
│   ├── pages/            # Next.js pages
│   └── styles/           # CSS and styling
├── scripts/               # Data processing scripts
└── docs/                  # Documentation
```

## Technology Stack

- **Backend**: FastAPI, Python, PostgreSQL
- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **AI/ML**: LLaMA Factory, Hugging Face Transformers
- **Audio**: ElevenLabs TTS, Whisper ASR
- **Deployment**: Docker, Nginx, DigitalOcean

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Run data collection scripts
5. Fine-tune models using LLaMA Factory
6. Start development server

## Project Timeline

- **Week 1-2**: Environment setup and data collection
- **Week 3-4**: Text model development
- **Week 5-6**: Web interface development
- **Week 7-8**: Audio integration
- **Week 9-10**: Deployment and launch