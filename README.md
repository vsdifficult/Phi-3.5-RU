# Phi-3.5-RU
Russian model PHI

## Model Presentation Features

The Phi-3.5-RU model includes built-in presentation capabilities to help users understand and visualize its performance.

### Key Features:
1. **Model Information**
   - Get detailed architecture and configuration
   - View total number of parameters
   - Check device placement (CPU/GPU)

2. **Visualization**
   - Generate heatmaps of text embeddings
   - Visualize token-level representations

### Usage Example

```python
from src.model import Phi35MoERUModel

# Initialize the model
model = Phi35MoERUModel()

# Generate model presentation
presentation = model.generate_presentation()
print(presentation)

# Visualize embeddings
model.visualize_embeddings("Пример текста для визуализации")
```

### Requirements
- matplotlib
- numpy
- torch
- transformers
