import sys
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from sqlalchemy.orm import sessionmaker
from functools import wraps
import datetime

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from trash_detect.database import engine, User, IssueReport, DisposalRecord, create_db_and_tables

app = Flask(__name__)
app.secret_key = 'supersecretkey' # Replace with a strong secret key in production

# Create database tables if they don't exist
create_db_and_tables()

# SessionLocal for database interactions
Session = sessionmaker(bind=engine)

# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        aadhar_id = request.form['aadhar_id']
        # For simplicity, password is not implemented. Aadhar ID acts as a login identifier.
        # In a real system, you'd have proper password hashing and verification.

        db_session = Session()
        user = db_session.query(User).filter_by(aadhar_id=aadhar_id).first()
        db_session.close()

        if user:
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check Aadhar ID.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        surname = request.form['surname']
        aadhar_id = request.form['aadhar_id']
        
        # Simulate face_id for now
        face_id = f"face_{aadhar_id}"

        db_session = Session()
        try:
            # Check if Aadhar ID or Face ID already exists
            existing_user_aadhar = db_session.query(User).filter_by(aadhar_id=aadhar_id).first()
            if existing_user_aadhar:
                flash('A user with this Aadhar ID already exists.', 'danger')
                return redirect(url_for('register'))

            new_user = User(name=name, surname=surname, aadhar_id=aadhar_id, face_id=face_id)
            db_session.add(new_user)
            db_session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db_session.rollback()
            flash(f'An error occurred during registration: {e}', 'danger')
        finally:
            db_session.close()

    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    db_session = Session()
    user = db_session.query(User).filter_by(id=session['user_id']).first()
    disposal_footages = db_session.query(DisposalRecord).filter_by(user_id=session['user_id']).order_by(DisposalRecord.timestamp.desc()).limit(5).all()
    db_session.close()

    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('logout')) # Log out if user somehow not found

    return render_template('dashboard.html', user=user, disposal_footages=disposal_footages)

@app.route('/report_issue', methods=['GET', 'POST'])
@login_required
def report_issue():
    if request.method == 'POST':
        issue_type = request.form['issue_type']
        description = request.form['description']

        db_session = Session()
        try:
            new_issue = IssueReport(user_id=session['user_id'], issue_type=issue_type, description=description)
            db_session.add(new_issue)
            db_session.commit()
            flash('Issue reported successfully!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            db_session.rollback()
            flash(f'An error occurred while reporting issue: {e}', 'danger')
        finally:
            db_session.close()
    return render_template('report_issue.html')

@app.route('/redeem_points')
@login_required
def redeem_points():
    return render_template('redeem_points.html')

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    db_session = Session()
    user = db_session.query(User).filter_by(id=session['user_id']).first()

    if request.method == 'POST':
        user.name = request.form['name']
        user.surname = request.form['surname']
        # Aadhar ID and Face ID are typically not changeable by user
        try:
            db_session.commit()
            flash('Your details have been updated successfully!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            db_session.rollback()
            flash(f'An error occurred while updating details: {e}', 'danger')
        finally:
            db_session.close()
    
    return render_template('settings.html', user=user)

@app.route('/faq')
def faq():
    return render_template('faq.html')


@app.route('/upload_footage', methods=['POST'])
@login_required
def upload_footage():
    if 'footage' not in request.files:
        flash('No footage file found', 'danger')
        return redirect(url_for('dashboard'))

    footage = request.files['footage']
    if footage.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard'))

    if footage:
        # In a real scenario, you would save the file to a secure location
        # and then call the prediction API.
        # For now, we will just simulate the prediction.

        # Simulate a call to the prediction API
        # In a real implementation, you would use requests library to call the api.py
        # e.g., requests.post('http://localhost:5002/predict', files={'file': footage})
        dummy_prediction = {
            'person_bbox': [[10, 10, 50, 50]],
            'person_class': 1,
            'face_embedding': [0.1, 0.2, 0.3],
            'trash_class': 3, # Let's say 3 is plastic
            'disposal_class': 1 # Let's say 1 is proper disposal
        }

        # Award points based on the prediction
        points_to_award = 0
        if dummy_prediction['disposal_class'] == 1:
            points_to_award = 10 # Award 10 points for proper disposal
        else:
            points_to_award = -5 # Deduct 5 points for improper disposal

        db_session = Session()
        try:
            user = db_session.query(User).filter_by(id=session['user_id']).first()
            user.points += points_to_award

            # Create a new disposal record
            new_disposal_record = DisposalRecord(
                user_id=session['user_id'],
                cctv_location='CCTV_1', # Placeholder
                trash_type='plastic', # Placeholder
                disposed_properly=True if dummy_prediction['disposal_class'] == 1 else False,
                points_awarded=points_to_award,
                footage_url='some_url' # Placeholder
            )
            db_session.add(new_disposal_record)
            db_session.commit()
            flash(f'{points_to_award} points awarded for your disposal!', 'success')
        except Exception as e:
            db_session.rollback()
            flash(f'An error occurred: {e}', 'danger')
        finally:
            db_session.close()

    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)