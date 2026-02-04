# AI-Powered Continuous Learning Compressor

A file compressor that uses a PyTorch-based neural network to predict byte distributions, combined with Arithmetic Coding for efficient compression. The model learns continuously from the data it compresses, becoming specialized to your specific data patterns over time.

## Project Structure
- `backend/`: AI model, compression engine, and training logic.
- `frontend/`: PyQt6-based desktop user interface.
- `models/`: Stores model versions ("Brains").
- `data/training_data/`: Compressed storage for learning data.

## Features
- Continuous learning: Model improves as it sees more data.
- Unified decompression: Automatically uses the correct model version for each file.
- Professional UI: Easy-to-use desktop interface.
