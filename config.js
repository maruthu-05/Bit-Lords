const firebaseConfig = {
  apiKey: "AIzaSyDzll5isKjWnT9J4KcOy0rknz9Blr9_EAM",
  authDomain: "dms-hackthon.firebaseapp.com",
  databaseURL: "https://dms-hackthon-default-rtdb.firebaseio.com",
  projectId: "dms-hackthon",
  storageBucket: "dms-hackthon.appspot.com",
  messagingSenderId: "334443472890",
  appId: "1:334443472890:web:13cb601358267324b06643",
  measurementId: "G-7BYNSGS0Z2"
};

// Initialize Firebase
if (!firebase.apps.length) {
  firebase.initializeApp(firebaseConfig);
}

// Reference your database
const registerformDB = firebase.database().ref("registerform");

document.getElementById("registerform").addEventListener("submit", submitForm);

function submitForm(e) {
  e.preventDefault();
  // Get form values
  const Username = getElementVal("username");
  const Email = getElementVal("email");
  const Password = getElementVal("password");
  const Experience = getElementVal("experience");
  const Age = getElementVal("age");
  const Native = getElementVal("native");
  const Phone = getElementVal("phone");
  const Address = getElementVal("address");
  const rating=1000;

  // Save form data to Firebase Realtime Database
  saveMessages(Username, Email, Password, Experience, Age, Native, Phone, Address,rating);

  // Show success alert
  document.querySelector(".alert").style.display = "block";

  // Remove the alert after 3 seconds
  setTimeout(() => {
      document.querySelector(".alert").style.display = "none";
  }, 3000);

  // Reset the form
  document.getElementById("registerform").reset();
}

// Function to save form values in Firebase
const saveMessages = (Username, Email, Password, Experience, Age, Native, Phone, Address,rating) => {
  const newRegisterForm = registerformDB.push();
  newRegisterForm.set({
    Username: Username,
    Email: Email,
    Password: Password, // Save the password
    Experience: Experience, // Save experience
    Age: Age,             // Save age
    Native: Native,       // Save native place
    Phone: Phone,         // Save phone number
    Address: Address,
    rating: rating
  }, function(error) {
    if (error) {
      console.error('Data could not be saved:', error);
    } else {
      console.log('Data saved successfully.');
    }
  });
};

// Function to get form values
const getElementVal = (id) => {
  return document.getElementById(id).value;
};
