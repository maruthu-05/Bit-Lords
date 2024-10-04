import firebase_admin
from firebase_admin import credentials, db

# Global variable to store the rating value
user_rating = None

# Initialize the Firebase app with service account credentials
cred = credentials.Certificate('C:/Users/narai/OneDrive/Desktop/cour/dms-hackthon-firebase-adminsdk-508l6-991963bae2.json')
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://dms-hackthon-default-rtdb.firebaseio.com/"  # Use the correct database URL without specific user reference
})

def get_user_rating(email):
    global user_rating
    # Reference to the "registerform" collection
    ref = db.reference('registerform')

    # Query the database for the user with the matching email
    users = ref.order_by_child('Email').equal_to(email).get()

    if users:
        # Assuming there is only one user with that email
        for user_id, user_info in users.items():
            user_rating = user_info.get('rating', None)  # Get the rating, default to None if not found
            print(f"User rating for {email}: {user_rating}")
            return user_id
    else:
        print(f"No user found with the email: {email}")
        return None

def update_user_field(email, field_name, new_value):
    # Get the user_id by querying with the email
    user_id = get_user_rating(email)

    if user_id:
        # Reference to the specific user in "registerform"
        try:
            ref = db.reference(f'registerform/{user_id}')
            ref.update({field_name: new_value})
            print(f"Updated {field_name} for {email} to {new_value}.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    else:
        print(f"Failed to update: No user found with the email {email}.")

# Example usage
if __name__ == '__main__':
    email_to_update = "dhoni@134gmail.com"  # Email to check
    field_to_update = 'rating'  # Field to update
    new_value = user_rating+score

    # Get user rating and update the field
    update_user_field(email_to_update, field_to_update, new_value)
  