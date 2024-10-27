# FaceSwap Studio Pro

A high-performance real-time face swapping application with an intuitive GUI. Built with Python and CUDA acceleration, this application provides professional-grade face swapping capabilities with high FPS rates.

![Preview](preview.gif)

## Features

- Real-time face swapping using webcam
- High-performance CUDA-accelerated processing
- Modern, user-friendly interface
- Multiple camera support
- Adjustable performance settings
- Advanced configuration options
- Virtual camera output

## Requirements

- Windows 10/11 (64-bit)
- NVIDIA GPU with CUDA support
- Python 3.8 or higher
- Webcam
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Xx59poAXx/FaceSwapper.git
cd faceswap-studio-pro
```

2. Install dependencies:
```bash
pip install opencv-python numpy insightface customtkinter Pillow pyvirtualcam pygrabber CTkMessagebox onnxruntime-gpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Download the model file:
- Download `inswapper_128_fp16.onnx` from the releases page
- Place it in the same directory as `FaceSwapper.py`

## Usage

1. Run the application:
```bash
python FaceSwapper.py
```

2. Select your source image (the face you want to swap with)
3. Choose your webcam from the dropdown menu
4. Click "Start Processing" to begin the face swap
5. Use "Advanced Settings" to fine-tune performance

## Advanced Settings

- **Model Selection**: Choose between different ONNX models for quality/performance balance
- **Camera Resolution**: Adjust input resolution (higher = better quality, lower FPS)
- **Target FPS**: Set desired frames per second
- **Processing Threads**: Number of parallel processing threads (higher = more GPU memory usage)
- **Frame Buffer**: Adjust smoothness vs. latency
- **Face Detection Size**: Balance between detection accuracy and performance
- **Skip Frames**: Skip frames for performance optimization

## Performance Tips

1. Adjust the face detection size based on your GPU capabilities
2. Use frame skipping on lower-end systems
3. Lower the resolution for higher FPS
4. Reduce processing threads if experiencing GPU memory issues
5. Use smaller models for better performance on less powerful GPUs

## Known Issues

- Virtual camera may require admin privileges
- Some webcams might need specific resolution settings
- High GPU memory usage with multiple processing threads

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- InsightFace for the face analysis and swapping models
- CustomTkinter for the modern UI components
- PyTorch team for CUDA acceleration support

## Disclaimer

This software is for educational purposes only. Users are responsible for complying with local laws and regulations regarding face swapping and digital content manipulation.

## Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/Xx59poAXx/FaceSwapper/issues) page
2. Create a new issue with detailed information about your problem
3. Include system specifications and error messages

## Roadmap

- [ ] Multi-face swapping support
- [ ] Custom model training integration
- [ ] Advanced face detection settings
- [ ] Performance optimization profiles
- [ ] Multiple output formats

## System Requirements

Minimum:
- NVIDIA GPU with 4GB VRAM
- Intel i5/AMD Ryzen 5 or better
- 8GB RAM
- Windows 10 64-bit

Recommended:
- NVIDIA GPU with 6GB+ VRAM
- Intel i7/AMD Ryzen 7 or better
- 16GB RAM
- Windows 10/11 64-bit
- SSD Storage