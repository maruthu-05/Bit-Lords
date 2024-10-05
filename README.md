# Driver Behavior Analysis

This project aims to monitor and analyze driver behavior in real time using computer vision and machine learning techniques. The system focuses on detecting distractions, such as drowsiness and other activities unrelated to driving. It uses pre-trained models from the Intel OpenVINO Model Zoo and is integrated with oneAPI for high performance.

## Key Features:

- Real-time face and eye detection using a camera focused on the driver.
- Partially interactive AI that act as the partner for driver in drowsiness.
- Scoring system that evaluates driver behavior and provides alerts based on specific distractions.
- SOS alert system using a simple gesture.
- Over view of driver's behaviour for their organisation.

## Prerequisites

- OpenVINO Toolkit: Pre-trained models from Intel's OpenVINO Model Zoo.
- oneAPI Toolkit: For hardware acceleration.
- Firebase: Used for data storage and management.
- HTML: Frontend interface to capture and store inputs into Firebase.
- Python 3.8+: Core language for the project.

## Setup

1. Clone the Repository

   ```bash
   git clone https://github.com/Narain3108/Bit-Lords.git
   cd Bit-Lords
2. Install Dependencies Make sure you have the required packages installed. You can install the dependencies using the following command:

   ```bash
   pip install -r requirements.txt
3. Install OpenVINO Follow the instructions from Intel's official OpenVINO documentation to install OpenVINO and configure the environment.
4. ### Set up Firebase
- Create a Firebase project and configure your Firestore database.
- Use the Firebase SDK to connect your project to Firebase. You will need the firebaseConfig object from the Firebase console.
5. Configure the Camera Ensure the system has access to a camera that can capture the driver's face. This will be used to detect driver behavior.

## USAGE
1. Run the Driver Detection System You can start the driver behavior detection system by running:
   ```bash
   prg1.py
2. ### Real-time Alerts
   The system will analyze the driver’s actions and display alerts if any dangerous or distracted behavior is detected.
3. ### Scoring System
   The scoring system evaluates the driver’s behavior based on detected distractions. The score can be used to generate a report or trigger alerts through a responsible AI voice assistant integrated with the system.

## Integration with Firebase
- Data from each detection (distractions and score) is stored in Firebase for further analysis.
- HTML forms capture additional data and store it in the Firestore database, enabling a seamless interaction between the web interface and the backend detection system.
  
## Features

- Driver registration and login system
- Real-time driver behavior monitoring using pre-trained models from OpenVINO
- Detection of distractions like mobile phone usage
- Driver scoring based on detected behaviors
- Firebase integration for storing driver data

## Project Structure

### 1. Live Detecting Image Page
This page shows the live detecting of driver behavior while the monitoring system is running.

![WhatsApp Image 2024-10-05 at 04 32 17_92bb1f23](https://github.com/user-attachments/assets/b21cc0d2-9c01-4da2-acf6-af6393dab581)
![WhatsApp Image 2024-10-05 at 09 46 35_8a894359](https://github.com/user-attachments/assets/b36e6029-e38c-4d58-91b2-ff866ac36c1c)

- **Function**: Displays real-time visual feedback of the driver behavior detection system.
- **Description**: Shows live analysis of driver behavior, highlighting distractions such as phone usage.

### 2. Driver Login Page
This is the initial login page where the driver inputs their email to start the monitoring process.

![Driver Login](https://github.com/user-attachments/assets/e8289ff6-fe94-4096-9d37-befdaeb88c09)



- **Function**: Allows the driver to start or stop the monitoring session.
- **Description**: A simple interface for driver login with options to start or stop behavior monitoring.

### 3. Registration Page
This page is used to create a new driver account, with fields for basic details such as name, email, password, and driving experience.

![Driver Registration](https://github.com/user-attachments/assets/ba2abaaa-02eb-42b7-ba21-b4c323e74d0c)


- **Function**: Registers new users into the system by capturing details like username, email, password, and driving experience.
- **Description**: The driver provides personal information needed for the scoring and monitoring process.

### 4. Admin Login Page
This page allows an admin to log in to manage driver data and behavior reports.

![Admin Login](https://github.com/user-attachments/assets/f3c4d631-ed69-4b2e-8118-ccc4b17c615a)


- **Function**: Grants the admin access to view and manage driver data, including scores and performance.
- **Description**: Admin can access and review the performance of drivers and system activity.

### 5. Driver Profile Page
This page displays the driver’s details, including their current score and experience.

![Driver Profile](https://github.com/user-attachments/assets/49105bc1-296e-4724-b3ba-c098c1d06b0f)


- **Function**: Displays driver details and the behavior analysis score based on detected distractions.
- **Description**: A summary of driver performance and data fetched from the Firebase database.

## Future Enhancements
- Integration of more behavior detection models (e.g., emotion detection, seatbelt detection).
- Real-time alerts and notifications to the driver based on detected behaviors.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
