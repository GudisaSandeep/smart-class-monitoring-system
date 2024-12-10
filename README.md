# Smart India Hackathon Project - AI-Powered Training Analytics

## Overview
An advanced AI-powered training analytics platform that provides real-time monitoring and analysis of skill training sessions. The system uses computer vision and artificial intelligence to assess student engagement, attention levels, and training effectiveness.

## Features
- Real-time video analytics
- AI-powered engagement tracking
- Facial emotion detection
- Training quality assessment
- Infrastructure compliance monitoring
- Automated report generation
- Multi-language support
- Role-based access control

## Technical Stack
- Python/Flask backend
- OpenCV for computer vision
- Google's Gemini AI for analysis
- DeepFace for emotion detection
- PyTorch & TensorFlow for AI processing
- Bootstrap for frontend
- IP Webcam integration

## Setup Instructions
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in .env file:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```
4. Run the application:
   ```bash
   python app.py
   ```
5. Access the application at `http://localhost:5001`

## Usage
1. Register an account with your email
2. Log in to access the dashboard
3. Connect your IP Webcam using the IP address
4. Start monitoring and analyzing training sessions
5. Generate detailed reports with AI insights

## Contributing
Feel free to submit issues and enhancement requests.

## License
[MIT License](LICENSE)
