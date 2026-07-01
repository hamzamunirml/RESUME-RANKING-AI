"""
generate_dataset.py
--------------------
Generates a synthetic dataset of resumes (PDF) and job descriptions (TXT)
for the AI Resume Screening & Candidate Ranking System project.

Run:
    python src/generate_dataset.py

Output:
    data/resumes/*.pdf              -> 40 synthetic candidate resumes
    data/job_descriptions/*.txt     -> 5 job descriptions (one per role)
"""

import os
import random
from fpdf import FPDF

random.seed(42)

# ---------------------------------------------------------------------
# 1. Base data pools
# ---------------------------------------------------------------------

FIRST_NAMES = [
    "Ali",
    "Ahmed",
    "Sara",
    "Hamza",
    "Ayesha",
    "Bilal",
    "Fatima",
    "Usman",
    "Zainab",
    "Hassan",
    "Mariam",
    "Omar",
    "Hina",
    "Kashif",
    "Sana",
    "Imran",
    "Nida",
    "Faisal",
    "Rabia",
    "Tariq",
    "Amna",
    "Danish",
    "Sadia",
    "Waqas",
    "Iqra",
    "Salman",
    "Mahnoor",
    "Adeel",
    "Farah",
    "Junaid",
    "Noor",
    "Rizwan",
    "Anum",
    "Shahid",
    "Laiba",
    "Asad",
    "Hira",
    "Nabeel",
    "Komal",
    "Zeeshan",
]
LAST_NAMES = [
    "Khan",
    "Ahmed",
    "Malik",
    "Butt",
    "Raza",
    "Iqbal",
    "Sheikh",
    "Chaudhry",
    "Farooq",
    "Abbasi",
    "Qureshi",
    "Javed",
    "Hussain",
    "Siddiqui",
    "Awan",
]

ROLES = {
    "MERN Developer": {
        "skills": [
            "MongoDB",
            "Express.js",
            "React.js",
            "Node.js",
            "JavaScript",
            "REST API",
            "Redux",
            "JWT Authentication",
            "HTML",
            "CSS",
            "Git",
            "Tailwind CSS",
            "Mongoose",
            "Socket.io",
        ],
        "titles": [
            "MERN Stack Developer",
            "Full Stack JavaScript Developer",
            "Web Application Developer",
        ],
        "degrees": [
            "BS Computer Science",
            "BS Software Engineering",
            "BS Information Technology",
        ],
    },
    "Frontend Developer": {
        "skills": [
            "React.js",
            "JavaScript",
            "TypeScript",
            "HTML5",
            "CSS3",
            "Tailwind CSS",
            "Bootstrap",
            "Redux",
            "Next.js",
            "Figma to Code",
            "Responsive Design",
            "Webpack",
            "Git",
        ],
        "titles": ["Frontend Developer", "UI Developer", "React Developer"],
        "degrees": [
            "BS Computer Science",
            "BS Software Engineering",
            "BS Information Technology",
        ],
    },
    "Backend Developer": {
        "skills": [
            "Node.js",
            "Express.js",
            "Python",
            "Django",
            "Flask",
            "REST API",
            "SQL",
            "PostgreSQL",
            "MongoDB",
            "Docker",
            "AWS",
            "Microservices",
            "Git",
            "Redis",
        ],
        "titles": ["Backend Developer", "API Developer", "Server-Side Engineer"],
        "degrees": [
            "BS Computer Science",
            "BS Software Engineering",
            "MS Computer Science",
        ],
    },
    "Data Scientist": {
        "skills": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "TensorFlow",
            "Machine Learning",
            "Deep Learning",
            "SQL",
            "Data Visualization",
            "Matplotlib",
            "Seaborn",
            "Statistics",
            "NLP",
            "Power BI",
        ],
        "titles": ["Data Scientist", "Machine Learning Engineer", "Data Analyst"],
        "degrees": [
            "BS Artificial Intelligence",
            "BS Data Science",
            "MS Computer Science",
        ],
    },
    "UI/UX Designer": {
        "skills": [
            "Figma",
            "Adobe XD",
            "Wireframing",
            "Prototyping",
            "User Research",
            "Design Systems",
            "Sketch",
            "Photoshop",
            "Illustrator",
            "Usability Testing",
            "Interaction Design",
            "Typography",
        ],
        "titles": ["UI/UX Designer", "Product Designer", "Interaction Designer"],
        "degrees": ["BS Software Engineering", "BFA Design", "BS Computer Science"],
    },
}

CITIES = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Rawalpindi",
    "Multan",
    "Faisalabad",
    "Peshawar",
]
UNIVERSITIES = [
    "KFUEIT",
    "UET Lahore",
    "FAST-NUCES",
    "COMSATS",
    "Punjab University",
    "NED University",
    "GIKI",
    "Air University",
]
COMPANIES = [
    "TechNova",
    "CodeCrafters",
    "PixelWorks",
    "DataForge",
    "ByteBridge",
    "NextGen Systems",
    "Innoverse",
    "SoftCr8ors",
    "Webify Solutions",
    "CloudNine Labs",
]

PROJECT_TEMPLATES = {
    "MERN Developer": [
        "E-commerce Platform with MERN Stack",
        "Real-time Chat Application",
        "Task Management Dashboard",
        "Blogging Platform with Auth",
    ],
    "Frontend Developer": [
        "Portfolio Website with React",
        "Movie Search App using TMDB API",
        "Admin Dashboard UI",
        "Weather Forecast App",
    ],
    "Backend Developer": [
        "RESTful API for Inventory System",
        "Authentication Microservice",
        "Payment Gateway Integration",
        "Job Queue System with Redis",
    ],
    "Data Scientist": [
        "Customer Churn Prediction Model",
        "Sentiment Analysis on Tweets",
        "Sales Forecasting Dashboard",
        "Image Classification with CNN",
    ],
    "UI/UX Designer": [
        "Mobile Banking App Redesign",
        "Food Delivery App Wireframes",
        "SaaS Dashboard UX Case Study",
        "Design System for Startup",
    ],
}

CERTS = [
    "Coursera - Machine Learning Specialization",
    "freeCodeCamp - Responsive Web Design",
    "Google - UX Design Certificate",
    "AWS Certified Cloud Practitioner",
    "Meta Front-End Developer Certificate",
    "Kaggle - Intro to Machine Learning",
    None,
    None,
]


# ---------------------------------------------------------------------
# 2. Resume text builder
# ---------------------------------------------------------------------


def build_resume_text(role):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"
    info = ROLES[role]

    n_skills = random.randint(6, len(info["skills"]))
    skills = random.sample(info["skills"], n_skills)

    exp_years = random.randint(0, 5)
    title = random.choice(info["titles"])
    degree = random.choice(info["degrees"])
    university = random.choice(UNIVERSITIES)
    city = random.choice(CITIES)
    company = random.choice(COMPANIES)
    projects = random.sample(
        PROJECT_TEMPLATES[role], k=min(2, len(PROJECT_TEMPLATES[role]))
    )
    cert = random.choice(CERTS)

    lines = []
    lines.append(name)
    lines.append(f"{title} | {city}, Pakistan")
    lines.append(
        f"Email: {first.lower()}.{last.lower()}@example.com | Phone: 03{random.randint(00,99):02d}-{random.randint(1000000,9999999)}"
    )
    lines.append("")
    lines.append("SUMMARY")
    lines.append(
        f"{title} with {exp_years} year(s) of experience in {role.lower()} "
        f"roles. Skilled in {', '.join(skills[:3])} and passionate about building "
        f"scalable, user-focused applications."
    )
    lines.append("")
    lines.append("SKILLS")
    lines.append(", ".join(skills))
    lines.append("")
    lines.append("EXPERIENCE")
    if exp_years > 0:
        lines.append(f"{title} - {company} ({city})")
        lines.append(f"Duration: {exp_years} year(s)")
        lines.append(f"- Worked on {projects[0]} using {skills[0]} and {skills[1]}.")
        lines.append(
            f"- Collaborated with cross-functional teams to deliver features on schedule."
        )
        lines.append(
            f"- Improved system performance and code quality through best practices."
        )
    else:
        lines.append("Fresh graduate - no formal industry experience yet.")
    lines.append("")
    lines.append("PROJECTS")
    for p in projects:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("EDUCATION")
    lines.append(
        f"{degree} - {university} ({random.randint(2018,2024)} - {random.randint(2022,2026)})"
    )
    lines.append("")
    if cert:
        lines.append("CERTIFICATIONS")
        lines.append(f"- {cert}")

    return "\n".join(lines), name


# ---------------------------------------------------------------------
# 3. PDF writer
# ---------------------------------------------------------------------


def write_resume_pdf(text, out_path):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)
    for line in text.split("\n"):
        if line.strip() == "":
            pdf.ln(4)
            continue
        if line.isupper():
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        else:
            pdf.multi_cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
    pdf.output(out_path)


# ---------------------------------------------------------------------
# 4. Job description builder
# ---------------------------------------------------------------------


def build_job_description(role):
    info = ROLES[role]
    n_skills = random.randint(8, len(info["skills"]))
    required_skills = random.sample(info["skills"], n_skills)

    text = f"""Job Title: {role}
Location: Remote / Pakistan
Experience Required: 1-3 years

About the Role:
We are looking for a talented {role} to join our team. The ideal candidate is
proficient in {', '.join(required_skills[:4])} and has a strong understanding
of modern development/design practices.

Responsibilities:
- Design, develop, and maintain high-quality solutions in {role.lower()} domain.
- Collaborate with cross-functional teams including design, product, and QA.
- Write clean, maintainable, and well-documented code/designs.
- Participate in code/design reviews and continuous improvement processes.

Required Skills:
{', '.join(required_skills)}

Qualifications:
- Bachelor's degree in Computer Science, Software Engineering, or related field.
- Strong problem-solving and communication skills.
- Prior project or internship experience preferred.
"""
    return text


# ---------------------------------------------------------------------
# 5. Main generation routine
# ---------------------------------------------------------------------


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resumes_dir = os.path.join(base_dir, "data", "resumes")
    jd_dir = os.path.join(base_dir, "data", "job_descriptions")
    os.makedirs(resumes_dir, exist_ok=True)
    os.makedirs(jd_dir, exist_ok=True)

    roles = list(ROLES.keys())
    total_resumes = 40
    resumes_per_role = total_resumes // len(roles)

    count = 0
    for role in roles:
        for i in range(resumes_per_role):
            text, name = build_resume_text(role)
            safe_name = name.replace(" ", "_")
            filename = (
                f"{safe_name}_{role.replace(' ', '_').replace('/', '')}_{i+1}.pdf"
            )
            write_resume_pdf(text, os.path.join(resumes_dir, filename))
            count += 1

    print(f"Generated {count} resumes in {resumes_dir}")

    for role in roles:
        jd_text = build_job_description(role)
        filename = f"{role.replace(' ', '_').replace('/', '')}_JD.txt"
        with open(os.path.join(jd_dir, filename), "w", encoding="utf-8") as f:
            f.write(jd_text)

    print(f"Generated {len(roles)} job descriptions in {jd_dir}")


if __name__ == "__main__":
    main()
