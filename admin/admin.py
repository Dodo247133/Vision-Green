import sys
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from sqlalchemy.orm import sessionmaker
from functools import wraps
import datetime

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from trash_detect.database import engine, User, IssueReport, create_db_and_tables

admin_app = Flask(__name__)
admin_app.secret_key = 'adminsecretkey' # Replace with a strong secret key in production

# Create database tables if they don't exist
create_db_and_tables()

# SessionLocal for database interactions
Session = sessionmaker(bind=engine)

# --- Admin Authentication Decorator ---
def admin_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'admin_logged_in' not in session:
            flash('Please log in as an admin to access this page.', 'warning')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

@admin_app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Hardcoded admin credentials for prototype. In production, use a database.
        if username == 'admin' and password == 'adminpass':
            session['admin_logged_in'] = True
            flash('Logged in as admin successfully!', 'success')
            return redirect(url_for('admin_index'))
        else:
            flash('Invalid credentials.', 'danger')
    return render_template('admin_login.html')

@admin_app.route('/admin/logout')
@admin_login_required
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('You have been logged out as admin.', 'info')
    return redirect(url_for('admin_login'))

@admin_app.route('/admin')
@admin_login_required
def admin_index():
    db_session = Session()
    user_count = db_session.query(User).count()
    open_issues_count = db_session.query(IssueReport).filter_by(is_resolved=False).count()
    recent_users = db_session.query(User).order_by(User.id.desc()).limit(5).all()
    recent_issues = db_session.query(IssueReport).order_by(IssueReport.id.desc()).limit(5).all()
    db_session.close()
    return render_template('admin_index.html', user_count=user_count, open_issues_count=open_issues_count, recent_users=recent_users, recent_issues=recent_issues)

@admin_app.route('/admin/users')
@admin_login_required
def admin_users():
    db_session = Session()
    users = db_session.query(User).all()
    db_session.close()
    return render_template('users.html', users=users)

@admin_app.route('/admin/user/<int:user_id>')
@admin_login_required
def admin_user_detail(user_id):
    db_session = Session()
    user = db_session.query(User).filter_by(id=user_id).first()
    issues = db_session.query(IssueReport).filter_by(user_id=user_id).all()
    db_session.close()
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('admin_users'))
    return render_template('user_detail.html', user=user, issues=issues)

@admin_app.route('/admin/user/<int:user_id>/ban', methods=['POST'])
@admin_login_required
def admin_ban_user(user_id):
    db_session = Session()
    user = db_session.query(User).filter_by(id=user_id).first()
    if user:
        user.is_active = False
        db_session.commit()
        flash(f'User {user.name} {user.surname} has been banned.', 'success')
    else:
        flash('User not found.', 'danger')
    db_session.close()
    return redirect(url_for('admin_users'))

@admin_app.route('/admin/user/<int:user_id>/unban', methods=['POST'])
@admin_login_required
def admin_unban_user(user_id):
    db_session = Session()
    user = db_session.query(User).filter_by(id=user_id).first()
    if user:
        user.is_active = True
        db_session.commit()
        flash(f'User {user.name} {user.surname} has been unbanned.', 'success')
    else:
        flash('User not found.', 'danger')
    db_session.close()
    return redirect(url_for('admin_users'))

@admin_app.route('/admin/model_status')
@admin_login_required
def admin_model_status():
    # Placeholder for ML model status
    model_status = {
        'online': True,
        'videos_processed': 12345,
        'current_processing_rate': '10 videos/min',
        'good_ratings': 8000,
        'bad_ratings': 2000
    }
    return render_template('model_status.html', model_status=model_status)

@admin_app.route('/admin/issues')
@admin_login_required
def admin_issues():
    db_session = Session()
    issues = db_session.query(IssueReport).all()
    db_session.close()
    return render_template('issues.html', issues=issues)

@admin_app.route('/admin/issue/<int:issue_id>/resolve', methods=['POST'])
@admin_login_required
def admin_resolve_issue(issue_id):
    db_session = Session()
    issue = db_session.query(IssueReport).filter_by(id=issue_id).first()
    if issue:
        issue.is_resolved = True
        issue.resolved_by = "Admin User" # Placeholder
        issue.resolved_at = datetime.datetime.now()
        db_session.commit()
        flash(f'Issue {issue_id} resolved.', 'success')
    else:
        flash('Issue not found.', 'danger')
    db_session.close()
    return redirect(url_for('admin_issues'))

if __name__ == '__main__':
    admin_app.run(debug=True, port=5001)