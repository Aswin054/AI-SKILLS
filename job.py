import random

# Mock job database (simulating an API response)
JOB_DATABASE = [
    {"title": "Security Manager", "skills": ["Risk Management", "Operational Planning"]},
    {"title": "IT Support Specialist", "skills": ["Network Administration", "Hardware Troubleshooting"]},
    {"title": "EMT/Paramedic", "skills": ["CPR Certified", "Trauma Care"]},
    {"title": "Data Analyst", "skills": ["Data Analysis", "Threat Assessment"]},
]

def recommend_jobs(military_skills):
    """Match military skills to civilian jobs."""
    matched_jobs = []
    for job in JOB_DATABASE:
        common_skills = set(job["skills"]) & set(military_skills)
        if common_skills:
            job["match_score"] = len(common_skills)
            matched_jobs.append(job)
    return sorted(matched_jobs, key=lambda x: x["match_score"], reverse=True)

# Example Usage
if __name__ == "__main__":
    # Assume skills from Skill Translator
    military_skills = ["Team Leadership", "Risk Management", "Operational Planning"]
    recommended_jobs = recommend_jobs(military_skills)
    print("Recommended Jobs:")
    for job in recommended_jobs:
        print(f"- {job['title']} (Match Score: {job['match_score']})")
