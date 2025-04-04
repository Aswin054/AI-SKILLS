from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
from functools import wraps
import uuid
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///adaptive_learning.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(70), unique=True)
    password = db.Column(db.String(80))
    education_level = db.Column(db.String(100))
    current_job = db.Column(db.String(100))
    experience_years = db.Column(db.String(50))
    career_goal = db.Column(db.String(100))
    daily_goal = db.Column(db.String(50))
    learning_style = db.Column(db.String(100))

class UserSkill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    skill_name = db.Column(db.String(100))
    skill_level = db.Column(db.Integer))  # 0-100

class Assessment(db.Model):
    id = db.Column(db.Integer, primary000_key=True)
    name = db.Column(db.String(100))
    description = db.Column(db.String(300))
    skill_category = db.Column(db.String(100))

class UserAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    assessment_id = db.Column(db.Integer, db.ForeignKey('assessment.id'))
    score = db.Column(db.Integer))
    completion_date = db.Column(db.DateTime))
    details = db.Column(db.JSON))

class LearningModule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    description = db.Column(db.String(300))
    skill_category = db.Column(db.String(100))
    difficulty_level = db.Column(db.Integer))  # 1-5
    content_url = db.Column(db.String(200))
    estimated_duration = db.Column(db.Integer))  # in minutes

class UserLearningPath(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    module_id = db.Column(db.Integer, db.ForeignKey('learning_module.id'))
    status = db.Column(db.String(50))  # 'pending', 'in_progress', 'completed'
    start_date = db.Column(db.DateTime))
    completion_date = db.Column(db.DateTime))
    progress = db.Column(db.Integer))  # 0-100

class JobPosting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    company = db.Column(db.String(100))
    location = db.Column(db.String(100))
    salary_range = db.Column(db.String(100))
    description = db.Column(db.Text))
    required_skills = db.Column(db.JSON))  # List of skills
    demand_level = db.Column(db.String(50))  # 'low', 'medium', 'high', 'emerging'

# AI Models and Utilities
skill_model = SentenceTransformer('all-MiniLM-L6-v2')

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
        
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# Routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    hashed_password = generate_password_hash(data['password'], method='sha256')
    
    new_user = User(
        public_id=str(uuid.uuid4()),
        name=data['name'],
        email=data['email'],
        password=hashed_password,
        education_level=data.get('education_level', ''),
        current_job=data.get('current_job', ''),
        experience_years=data.get('experience_years', ''),
        career_goal=data.get('career_goal', ''),
        daily_goal=data.get('daily_goal', '1 hour'),
        learning_style=data.get('learning_style', 'Interactive (Hands-on exercises)')
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    # Initialize default skills
    default_skills = [
        ('Python', 0),
        ('Data Analysis', 0),
        ('Machine Learning', 0),
        ('SQL', 0),
        ('Data Visualization', 0),
        ('Statistics', 0)
    ]
    
    for skill_name, level in default_skills:
        new_skill = UserSkill(
            user_id=new_user.id,
            skill_name=skill_name,
            skill_level=level
        )
        db.session.add(new_skill)
    
    db.session.commit()
    
    return jsonify({'message': 'User created successfully!'}), 201

@app.route('/api/login', methods=['POST'])
def login():
    auth = request.authorization
    
    if not auth or not auth.username or not auth.password:
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})
    
    user = User.query.filter_by(email=auth.username).first()
    
    if not user:
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})
    
    if check_password_hash(user.password, auth.password):
        token = jwt.encode({
            'public_id': user.public_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        
        return jsonify({'token': token, 'user_id': user.public_id})
    
    return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})

@app.route('/api/user', methods=['GET'])
@token_required
def get_user(current_user):
    user_data = {
        'public_id': current_user.public_id,
        'name': current_user.name,
        'email': current_user.email,
        'education_level': current_user.education_level,
        'current_job': current_user.current_job,
        'experience_years': current_user.experience_years,
        'career_goal': current_user.career_goal,
        'daily_goal': current_user.daily_goal,
        'learning_style': current_user.learning_style
    }
    
    return jsonify({'user': user_data})

@app.route('/api/user/skills', methods=['GET'])
@token_required
def get_user_skills(current_user):
    skills = UserSkill.query.filter_by(user_id=current_user.id).all()
    
    skills_data = []
    for skill in skills:
        skills_data.append({
            'skill_name': skill.skill_name,
            'skill_level': skill.skill_level
        })
    
    return jsonify({'skills': skills_data})

@app.route('/api/user/skills', methods=['PUT'])
@token_required
def update_user_skills(current_user):
    data = request.get_json()
    
    # Delete all existing skills
    UserSkill.query.filter_by(user_id=current_user.id).delete()
    
    # Add updated skills
    for skill in data['skills']:
        new_skill = UserSkill(
            user_id=current_user.id,
            skill_name=skill['skill_name'],
            skill_level=skill['skill_level']
        )
        db.session.add(new_skill)
    
    db.session.commit()
    
    return jsonify({'message': 'Skills updated successfully!'})

@app.route('/api/user/assessments', methods=['GET'])
@token_required
def get_user_assess  ments(current_user):
    # Get available assessments
    available_assessments = Assessment.query.all()
    available_data = []
    for assessment in available_assessments:
        # Check if user has already taken this assessment
        user_assessment = UserAssessment.query.filter_by(
            user_id=current_user.id,
            assessment_id=assessment.id
        ).first()
        
        if not user_assessment:
            available_data.append({
                'id': assessment.id,
                'name': assessment.name,
                'description': assessment.description,
                'skill_category': assessment.skill_category,
                'completed': False
            })
    
    # Get completed assessments
    completed_assessments = UserAssessment.query.filter_by(user_id=current_user.id).all()
    completed_data = []
    for ua in completed_assessments:
        assessment = Assessment.query.get(ua.assessment_id)
        completed_data.append({
            'id': ua.id,
            'name': assessment.name,
            'score': ua.score,
            'completion_date': ua.completion_date.strftime('%Y-%m-%d'),
            'details': ua.details
        })
    
    return jsonify({
        'available_assessments': available_data,
        'completed_assessments': completed_data
    })

@app.route('/api/assessments/start/<int:assessment_id>', methods=['GET'])
@token_required
def start_assessment(current_user, assessment_id):
    assessment = Assessment.query.get(assessment_id)
    
    if not assessment:
        return jsonify({'message': 'Assessment not found!'}), 404
    
    # For this example, we'll return a static set of questions
    # In a real app, you'd generate these dynamically based on the assessment
    if assessment.skill_category == 'Python':
        questions = [
            {
                'id': 1,
                'text': "What is the output of the following Python code?\n\nprint([x**2 for x in range(5)])",
                'options': [
                    {'text': "[0, 1, 4, 9, 16]", 'correct': True},
                    {'text': "[1, 4, 9, 16, 25]", 'correct': False},
                    {'text': "[0, 1, 4, 9, 16, 25]", 'correct': False},
                    {'text': "Error", 'correct': False}
                ]
            },
            {
                'id': 2,
                'text': "Which of the following is NOT a valid Python data type?",
                'options': [
                    {'text': "list", 'correct': False},
                    {'text': "tuple", 'correct': False},
                    {'text': "array", 'correct': True},
                    {'text': "dict", 'correct': False}
                ]
            },
            {
                'id': 3,
                'text': "What does the zip() function do in Python?",
                'options': [
                    {'text': "Compresses files to save disk space", 'correct': False},
                    {'text': "Combines multiple iterables into tuples", 'correct': True},
                    {'text': "Encrypts data for security", 'correct': False},
                    {'text': "Converts data to ZIP format", 'correct': False}
                ]
            }
        ]
    else:
        questions = []  # Default empty if no matching category
    
    return jsonify({
        'assessment_id': assessment.id,
        'assessment_name': assessment.name,
        'questions': questions
    })

@app.route('/api/assessments/submit/<int:assessment_id>', methods=['POST'])
@token_required
def submit_assessment(current_user, assessment_id):
    data = request.get_json()
    answers = data.get('answers', [])
    
    # Calculate score
    total_questions = len(answers)
    correct_answers = sum(1 for answer in answers if answer.get('correct', False))
    score = int((correct_answers / total_questions) * 100) if total_questions > 0 else 0
    
    # Determine skill level
    if score >= 80:
        level = "Proficient"
    elif score >= 60:
        level = "Intermediate"
    else:
        level = "Beginner"
    
    # Get assessment to determine skill category
    assessment = Assessment.query.get(assessment_id)
    if not assessment:
        return jsonify({'message': 'Assessment not found!'}), 404
    
    # Save assessment results
    user_assessment = UserAssessment(
        user_id=current_user.id,
        assessment_id=assessment.id,
        score=score,
        completion_date=datetime.datetime.utcnow(),
        details={
            'level': level,
            'strengths': get_random_strengths(assessment.skill_category),
            'weaknesses': get_random_weaknesses(assessment.skill_category)
        }
    )
    db.session.add(user_assessment)
    
    # Update user's skill level for this category
    user_skill = UserSkill.query.filter_by(
        user_id=current_user.id,
        skill_name=assessment.skill_category
    ).first()
    
    if user_skill:
        # Only update if the new score is higher than the current
        if score > user_skill.skill_level:
            user_skill.skill_level = score
    else:
        # Create new skill if it doesn't exist
        new_skill = UserSkill(
            user_id=current_user.id,
            skill_name=assessment.skill_category,
            skill_level=score
        )
        db.session.add(new_skill)
    
    db.session.commit()
    
    # Generate learning recommendations
    recommendations = generate_learning_recommendations(current_user.id, assessment.skill_category, score)
    
    return jsonify({
        'message': 'Assessment submitted successfully!',
        'score': score,
        'level': level,
        'recommendations': recommendations
    })

@app.route('/api/learning-path', methods=['GET'])
@token_required
def get_learning_path(current_user):
    # Get user's current path items
    path_items = UserLearningPath.query.filter_by(user_id=current_user.id).all()
    
    if not path_items:
        # If no path exists, generate a new one
        return generate_initial_learning_path(current_user.id)
    
    path_data = []
    for item in path_items:
        module = LearningModule.query.get(item.module_id)
        path_data.append({
            'id': item.id,
            'module_id': item.module_id,
            'title': module.title,
            'description': module.description,
            'status': item.status,
            'progress': item.progress,
            'start_date': item.start_date.strftime('%Y-%m-%d') if item.start_date else None,
            'completion_date': item.completion_date.strftime('%Y-%m-%d') if item.completion_date else None,
            'content_url': module.content_url,
            'estimated_duration': module.estimated_duration,
            'difficulty_level': module.difficulty_level
        })
    
    return jsonify({'learning_path': path_data})

@app.route('/api/learning-path/progress/<int:path_item_id>', methods=['PUT'])
@token_required
def update_learning_progress(current_user, path_item_id):
    data = request.get_json()
    progress = data.get('progress', 0)
    
    path_item = UserLearningPath.query.get(path_item_id)
    if not path_item or path_item.user_id != current_user.id:
        return jsonify({'message': 'Learning path item not found!'}), 404
    
    path_item.progress = progress
    
    # Update status if progress is 100%
    if progress >= 100:
        path_item.status = 'completed'
        path_item.completion_date = datetime.datetime.utcnow()
        
        # Update user's skill level
        module = LearningModule.query.get(path_item.module_id)
        if module:
            user_skill = UserSkill.query.filter_by(
                user_id=current_user.id,
                skill_name=module.skill_category
            ).first()
            
            if user_skill:
                # Increase skill level by 10-20 points for completing a module
                increase = min(20, 100 - user_skill.skill_level)
                user_skill.skill_level += increase
    
    db.session.commit()
    
    return jsonify({'message': 'Progress updated successfully!'})

@app.route('/api/job-market', methods=['GET'])
@token_required
def get_job_market(current_user):
    # Get all job postings
    jobs = JobPosting.query.all()
    
    # Get user skills
    user_skills = UserSkill.query.filter_by(user_id=current_user.id).all()
    user_skill_names = [skill.skill_name for skill in user_skills if skill.skill_level > 40]  # Only consider skills with level > 40
    
    job_data = []
    for job in jobs:
        # Calculate match score based on required skills
        required_skills = job.required_skills or []
        matched_skills = [skill for skill in required_skills if skill in user_skill_names]
        match_score = int((len(matched_skills) / len(required_skills)) * 100) if required_skills else 0
        
        job_data.append({
            'id': job.id,
            'title': job.title,
            'company': job.company,
            'location': job.location,
            'salary_range': job.salary_range,
            'demand_level': job.demand_level,
            'match_score': match_score,
            'required_skills': required_skills,
            'matched_skills': matched_skills
        })
    
    # Sort by match score (descending)
    job_data.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Categorize into best matches, high demand, and emerging roles
    best_matches = [job for job in job_data if job['match_score'] >= 70]
    high_demand = [job for job in job_data if job['demand_level'] == 'high' and job not in best_matches]
    emerging_roles = [job for job in job_data if job['demand_level'] == 'emerging' and job not in best_matches]
    
    return jsonify({
        'best_matches': best_matches[:3],  # Return top 3 for each category
        'high_demand': high_demand[:3],
        'emerging_roles': emerging_roles[:3]
    })

@app.route('/api/career-support/resume', methods=['GET'])
@token_required
def get_resume_analysis(current_user):
    # Get user data
    user_skills = UserSkill.query.filter_by(user_id=current_user.id).all()
    user_assessments = UserAssessment.query.filter_by(user_id=current_user.id).all()
    learning_path = UserLearningPath.query.filter_by(user_id=current_user.id).all()
    
    # Generate resume analysis
    skills_data = [{'name': skill.skill_name, 'level': skill.skill_level} for skill in user_skills if skill.skill_level > 0]
    
    # Group by skill level
    advanced_skills = [skill for skill in skills_data if skill['level'] >= 80]
    intermediate_skills = [skill for skill in skills_data if 60 <= skill['level'] < 80]
    beginner_skills = [skill for skill in skills_data if skill['level'] < 60]
    
    # Get completed modules
    completed_modules = []
    for item in learning_path:
        if item.status == 'completed':
            module = LearningModule.query.get(item.module_id)
            if module:
                completed_modules.append(module.title)
    
    # Generate suggestions
    suggestions = []
    if len(advanced_skills) < 3:
        suggestions.append("Consider developing more advanced skills in your areas of expertise.")
    if not any(skill for skill in skills_data if skill['name'] == 'Communication'):
        suggestions.append("Add communication skills to your profile as they're valuable in any role.")
    if len(completed_modules) < 3:
        suggestions.append("Complete more learning modules to strengthen your profile.")
    
    return jsonify({
        'skills': skills_data,
        'strengths': {
            'advanced_skills': advanced_skills,
            'certifications': completed_modules
        },
        'suggestions': suggestions
    })

# AI and Utility Functions
def generate_initial_learning_path(user_id):
    # Get user's career goal and current skills
    user = User.query.get(user_id)
    user_skills = UserSkill.query.filter_by(user_id=user_id).all()
    
    # For this example, we'll use a predefined path based on career goal
    # In a real app, you'd use AI to generate a personalized path
    if user.career_goal == 'Data Scientist':
        modules = [
            ('Python Fundamentals', 'Learn Python basics', 'Python', 1, 180),
            ('Data Analysis with Pandas', 'Master data manipulation', 'Data Analysis', 2, 240),
            ('Machine Learning Basics', 'Introduction to ML algorithms', 'Machine Learning', 3, 300),
            ('Data Visualization', 'Create effective visualizations', 'Data Visualization', 2, 180),
            ('SQL for Data Science', 'Query databases efficiently', 'SQL', 2, 240)
        ]
    else:  # Default to Data Analyst path
        modules = [
            ('Python Fundamentals', 'Learn Python basics', 'Python', 1, 180),
            ('Data Analysis with Pandas', 'Master data manipulation', 'Data Analysis', 2, 240),
            ('Data Visualization', 'Create effective visualizations', 'Data Visualization', 2, 180),
            ('SQL Fundamentals', 'Query databases', 'SQL', 2, 240),
            ('Excel for Analysts', 'Advanced Excel techniques', 'Excel', 1, 120)
        ]
    
    # Create learning path items
    path_items = []
    for i, (title, desc, category, difficulty, duration) in enumerate(modules):
        # Check if module exists
        module = LearningModule.query.filter_by(title=title).first()
        if not module:
            module = LearningModule(
                title=title,
                description=desc,
                skill_category=category,
                difficulty_level=difficulty,
                content_url=f"/content/{title.lower().replace(' ', '-')}",
                estimated_duration=duration
            )
            db.session.add(module)
            db.session.commit()
        
        # Create path item
        status = 'pending'
        if i == 0:
            status = 'in_progress'
        
        path_item = UserLearningPath(
            user_id=user_id,
            module_id=module.id,
            status=status,
            start_date=datetime.datetime.utcnow() if status == 'in_progress' else None,
            progress=0
        )
        db.session.add(path_item)
        path_items.append({
            'id': path_item.id,
            'module_id': module.id,
            'title': module.title,
            'description': module.description,
            'status': path_item.status,
            'progress': path_item.progress,
            'content_url': module.content_url,
            'estimated_duration': module.estimated_duration,
            'difficulty_level': module.difficulty_level
        })
    
    db.session.commit()
    
    return jsonify({'learning_path': path_items})

def generate_learning_recommendations(user_id, skill_category, assessment_score):
    # Get user's current skills and learning path
    user_skills = UserSkill.query.filter_by(user_id=user_id).all()
    current_path = UserLearningPath.query.filter_by(user_id=user_id).all()
    
    # Get all modules in this skill category, ordered by difficulty
    modules = LearningModule.query.filter_by(skill_category=skill_category)\
        .order_by(LearningModule.difficulty_level.asc()).all()
    
    # Filter out modules the user is already taking or has completed
    current_module_ids = [item.module_id for item in current_path]
    available_modules = [m for m in modules if m.id not in current_module_ids]
    
    # Recommend modules based on assessment score
    if assessment_score < 60:
        # Beginner level - recommend basic modules
        recommended_modules = [m for m in available_modules if m.difficulty_level <= 2][:2]
    elif assessment_score < 80:
        # Intermediate level - recommend intermediate modules
        recommended_modules = [m for m in available_modules if 2 <= m.difficulty_level <= 3][:2]
    else:
        # Advanced level - recommend advanced modules
        recommended_modules = [m for m in available_modules if m.difficulty_level >= 3][:2]
    
    # Format recommendations
    recommendations = []
    for module in recommended_modules:
        recommendations.append({
            'module_id': module.id,
            'title': module.title,
            'description': module.description,
            'difficulty_level': module.difficulty_level,
            'reason': f"Based on your {skill_category} skill level ({assessment_score}%)"
        })
    
    return recommendations

def get_random_strengths(skill_category):
    strengths_map = {
        'Python': ['Syntax knowledge', 'Basic programming concepts', 'Problem-solving skills'],
        'Data Analysis': ['Data cleaning', 'Basic aggregations', 'Attention to detail'],
        'Machine Learning': ['Algorithm understanding', 'Model evaluation basics', 'Feature engineering concepts']
    }
    return strengths_map.get(skill_category, ['Problem-solving', 'Logical thinking'])

def get_random_weaknesses(skill_category):
    weaknesses_map = {
        'Python': ['Advanced concepts', 'Performance optimization', 'Debugging skills'],
        'Data Analysis': ['Advanced transformations', 'Time series analysis', 'Big data handling'],
        'Machine Learning': ['Hyperparameter tuning', 'Neural networks', 'Model deployment']
    }
    return weaknesses_map.get(skill_category, ['Advanced techniques', 'Real-world application'])

def calculate_skill_similarity(user_skills, job_skills):
    """Calculate similarity between user skills and job required skills using embeddings"""
    if not user_skills or not job_skills:
        return 0
    
    # Encode skills
    user_skills_text = ' '.join(user_skills)
    job_skills_text = ' '.join(job_skills)
    
    # Get embeddings
    user_embedding = skill_model.encode(user_skills_text)
    job_embedding = skill_model.encode(job_skills_text)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(
        user_embedding.reshape(1, -1),
        job_embedding.reshape(1, -1)
    )[0][0]
    
    # Scale to 0-100
    return int((similarity + 1) * 50)  # Cosine similarity ranges from -1 to 1

# Database Initialization
def initialize_database():
    with app.app_context():
        db.create_all()
        
        # Add sample assessments if none exist
        if Assessment.query.count() == 0:
            assessments = [
                ('Python Programming', 'Test your Python knowledge', 'Python'),
                ('Data Analysis', 'Assess your data analysis skills', 'Data Analysis'),
                ('Machine Learning Concepts', 'Evaluate your ML understanding', 'Machine Learning')
            ]
            
            for name, desc, category in assessments:
                assessment = Assessment(
                    name=name,
                    description=desc,
                    skill_category=category
                )
                db.session.add(assessment)
            
            db.session.commit()
        
        # Add sample job postings if none exist
        if JobPosting.query.count() == 0:
            jobs = [
                {
                    'title': 'Data Analyst',
                    'company': 'TechCorp Inc.',
                    'location': 'New York, NY',
                    'salary_range': '$75,000 - $95,000',
                    'description': 'Analyze data and create reports to drive business decisions.',
                    'required_skills': ['Python', 'SQL', 'Excel', 'Tableau'],
                    'demand_level': 'high'
                },
                {
                    'title': 'Machine Learning Engineer',
                    'company': 'AI Solutions',
                    'location': 'San Francisco, CA',
                    'salary_range': '$120,000 - $160,000',
                    'description': 'Develop and deploy machine learning models.',
                    'required_skills': ['Python', 'Machine Learning', 'TensorFlow', 'PyTorch'],
                    'demand_level': 'high'
                },
                {
                    'title': 'Data Scientist',
                    'company': 'DataWorks LLC',
                    'location': 'Remote',
                    'salary_range': '$90,000 - $130,000',
                    'description': 'Extract insights from complex data sets.',
                    'required_skills': ['Python', 'Machine Learning', 'Statistics', 'SQL'],
                    'demand_level': 'medium'
                }
            ]
            
            for job in jobs:
                posting = JobPosting(
                    title=job['title'],
                    company=job['company'],
                    location=job['location'],
                    salary_range=job['salary_range'],
                    description=job['description'],
                    required_skills=job['required_skills'],
                    demand_level=job['demand_level']
                )
                db.session.add(posting)
            
            db.session.commit()

if __name__ == '__main__':
    initialize_database()
    app.run(debug=True)
