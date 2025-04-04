def build_resume(name, military_role, translated_skills, civilian_job_target):
    """Generate a resume snippet for civilian applications."""
    resume = f"""
    RESUME FOR: {name}
    ----------------------------------------
    Objective: Transitioning {military_role} seeking a role as {civilian_job_target}.

    Relevant Skills:
    - {', '.join(translated_skills['skills'])}

    Military Experience:
    - Served as {military_role}, specializing in {', '.join(translated_skills['skills'])}.
    - Led teams in high-pressure environments with a focus on {translated_skills['skills'][0]}.
    """
    return resume

# Example Usage
if __name__ == "__main__":
    name = input("Your Name: ")
    military_role = "11B Infantry Team Leader"
    translated_skills = {"skills": ["Team Leadership", "Risk Management"]}
    civilian_job_target = "Security Manager"
    print(build_resume(name, military_role, translated_skills, civilian_job_target))
