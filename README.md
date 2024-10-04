# Driver Behavior Analysis

This project focuses on detecting driver distractions and behaviors using a camera to monitor the driver’s face. The system can detect activities such as mobile phone usage (watching, speaking) and other non-driving behaviors. The output is used to score the driver’s performance and alert them in real-time if necessary.

## Features

- Driver registration and login system
- Real-time driver behavior monitoring using pre-trained models from OpenVINO
- Detection of distractions like mobile phone usage
- Driver scoring based on detected behaviors
- Firebase integration for storing driver data

## Project Structure

### 1. Driver Login Page
This is the initial login page where the driver inputs their email to start the monitoring process.

![Driver Login]![driver login](https://github.com/user-attachments/assets/e8289ff6-fe94-4096-9d37-befdaeb88c09)



- **Function**: Allows the driver to start or stop the monitoring session.
- **Description**: A simple interface for driver login with options to start or stop behavior monitoring.

### 2. Registration Page
This page is used to create a new driver account, with fields for basic details such as name, email, password, and driving experience.

![Driver Registration]![register](https://github.com/user-attachments/assets/ba2abaaa-02eb-42b7-ba21-b4c323e74d0c)


- **Function**: Registers new users into the system by capturing details like username, email, password, and driving experience.
- **Description**: The driver provides personal information needed for the scoring and monitoring process.

### 3. Driver Profile Page
This page displays the driver’s details, including their current score and experience.

![Driver Profile]![profilepage](https://github.com/user-attachments/assets/49105bc1-296e-4724-b3ba-c098c1d06b0f)


- **Function**: Displays driver details and the behavior analysis score based on detected distractions.
- **Description**: A summary of driver performance and data fetched from the Firebase database.

### 4. Admin Login Page
This page allows an admin to log in to manage driver data and behavior reports.

![Admin Login]![admin login](https://github.com/user-attachments/assets/f3c4d631-ed69-4b2e-8118-ccc4b17c615a)


- **Function**: Grants the admin access to view and manage driver data, including scores and performance.
- **Description**: Admin can access and review the performance of drivers and system activity.

### 5. Live Detecting Image Page
This page shows the live detecting of driver behavior while the monitoring system is running.

![Live Detecting]![WhatsApp Image 2024-10-05 at 02 41 00_67aa15a1](https://github.com/user-attachments/assets/5d5336ce-f1ba-42c1-a6c2-c50f6697d3b8)

- **Function**: Displays real-time visual feedback of the driver behavior detection system.
- **Description**: Shows live analysis of driver behavior, highlighting distractions such as phone usage.

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone <repository-url>
2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
3.Configure Firebase credentials in the project for database integration.
4.Run the driver behavior analysis model with:
   python driver_monitor.py
5.Open the web interface and log in as a driver or admin to start the session.
##Firebase Integration
The driver data (e.g., name, score, and behavior analysis) is stored in Firebase. You will need to configure Firebase by adding your Firebase project credentials.

##Technologies Used
-OpenVINO Toolkit: For running pre-trained models for driver monitoring.
-Firebase: As the database to store driver records.
-HTML/CSS/JavaScript: For front-end web development.
-Python: For the backend logic and behavior analysis model.

##Future Enhancements
-Integration of more behavior detection models (e.g., drowsiness, seatbelt detection).
-Real-time alerts and notifications to the driver based on detected behaviors.

##License
This project is licensed under the MIT License - see the LICENSE.md file for details.
