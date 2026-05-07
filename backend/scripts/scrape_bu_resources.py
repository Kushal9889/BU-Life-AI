"""
Scrape BU public resource pages and save to data/bu_resources.json.
Run from the backend/ directory: python scripts/scrape_bu_resources.py
"""
import httpx
from bs4 import BeautifulSoup
import json
import os

BU_PAGES = [
    # Career
    {"url": "https://www.bu.edu/careers/", "category": "career", "title": "BU Career Development"},
    {"url": "https://www.bu.edu/careers/find-a-job/internships/", "category": "career", "title": "BU Internship Resources"},
    {"url": "https://www.bu.edu/careers/explore-careers/", "category": "career", "title": "BU Career Exploration"},
    {"url": "https://www.bu.edu/careers/find-a-job/", "category": "career", "title": "BU Job Search Resources"},
    {"url": "https://www.bu.edu/careers/build-skills/resume-cover-letter/", "category": "career", "title": "BU Resume & Cover Letter Help"},

    # International students
    {"url": "https://www.bu.edu/isso/", "category": "international", "title": "International Students & Scholars (ISSO)"},
    {"url": "https://www.bu.edu/isso/employment/opt/", "category": "international", "title": "OPT - Optional Practical Training"},
    {"url": "https://www.bu.edu/isso/employment/cpt/", "category": "international", "title": "CPT - Curricular Practical Training"},
    {"url": "https://www.bu.edu/isso/travel/", "category": "international", "title": "ISSO Travel & Visa Signatures"},
    {"url": "https://www.bu.edu/isso/immigration-basics/f-1-status/", "category": "international", "title": "F-1 Student Status Basics"},

    # Academic advising
    {"url": "https://www.bu.edu/cas/academics/advising/", "category": "advising", "title": "CAS Academic Advising"},
    {"url": "https://www.bu.edu/reg/", "category": "registrar", "title": "University Registrar"},
    {"url": "https://www.bu.edu/reg/registration/", "category": "registrar", "title": "BU Course Registration"},
    {"url": "https://www.bu.edu/reg/academics/calendar/", "category": "registrar", "title": "BU Academic Calendar"},
    {"url": "https://www.bu.edu/reg/grades/", "category": "registrar", "title": "BU Grades & Transcripts"},
    {"url": "https://www.bu.edu/reg/graduation/", "category": "registrar", "title": "BU Graduation & Diplomas"},

    # Tutoring & academic support
    {"url": "https://www.bu.edu/tutoring/", "category": "tutoring", "title": "Educational Resource Center (ERC)"},
    {"url": "https://www.bu.edu/tutoring/services/tutoring/", "category": "tutoring", "title": "ERC Tutoring Services"},
    {"url": "https://www.bu.edu/tutoring/services/writing/", "category": "tutoring", "title": "ERC Writing Assistance"},
    {"url": "https://www.bu.edu/tutoring/services/study-skills/", "category": "tutoring", "title": "ERC Study Skills"},

    # Health & wellness
    {"url": "https://www.bu.edu/shs/", "category": "health", "title": "Student Health Services"},
    {"url": "https://www.bu.edu/shs/medical/", "category": "health", "title": "BU Medical Services"},
    {"url": "https://www.bu.edu/shs/behavioral-medicine/", "category": "health", "title": "BU Behavioral Medicine & Counseling"},
    {"url": "https://www.bu.edu/wellness/", "category": "wellness", "title": "Student Wellness & Prevention Services"},
    {"url": "https://www.bu.edu/shs/immunizations/", "category": "health", "title": "BU Health Immunization Requirements"},

    # Library
    {"url": "https://www.bu.edu/library/", "category": "library", "title": "BU Libraries"},
    {"url": "https://www.bu.edu/library/mugar-memorial/", "category": "library", "title": "Mugar Memorial Library"},
    {"url": "https://www.bu.edu/library/research/", "category": "library", "title": "BU Library Research Help"},
    {"url": "https://www.bu.edu/library/services/", "category": "library", "title": "BU Library Services"},
    {"url": "https://www.bu.edu/library/about/hours/", "category": "library", "title": "BU Library Hours"},

    # Financial aid
    {"url": "https://www.bu.edu/financialaid/", "category": "financial_aid", "title": "Financial Aid & Scholarships"},
    {"url": "https://www.bu.edu/financialaid/applying/", "category": "financial_aid", "title": "How to Apply for Financial Aid at BU"},
    {"url": "https://www.bu.edu/financialaid/types-of-aid/", "category": "financial_aid", "title": "Types of Financial Aid at BU"},
    {"url": "https://www.bu.edu/financialaid/faq/", "category": "financial_aid", "title": "Financial Aid FAQ"},

    # Housing
    {"url": "https://www.bu.edu/housing/", "category": "housing", "title": "BU Housing"},
    {"url": "https://www.bu.edu/housing/undergrad/", "category": "housing", "title": "BU Undergraduate Housing"},
    {"url": "https://www.bu.edu/housing/dining/", "category": "housing", "title": "BU Housing & Dining Plans"},

    # Accessibility
    {"url": "https://www.bu.edu/disability/", "category": "accessibility", "title": "Disability & Access Services"},
    {"url": "https://www.bu.edu/disability/accommodations/", "category": "accessibility", "title": "BU Academic Accommodations"},

    # Dining
    {"url": "https://www.bu.edu/dining/", "category": "dining", "title": "BU Dining"},
    {"url": "https://www.bu.edu/dining/meal-plans/", "category": "dining", "title": "BU Meal Plans"},

    # IT & tech support
    {"url": "https://www.bu.edu/rit/", "category": "it", "title": "Research & IT Help"},
    {"url": "https://www.bu.edu/tech/", "category": "it", "title": "BU Information Services & Technology"},
    {"url": "https://www.bu.edu/tech/support/", "category": "it", "title": "BU Tech Support & Help Desk"},

    # Study abroad & fellowships
    {"url": "https://www.bu.edu/abroad/", "category": "study_abroad", "title": "Study Abroad"},
    {"url": "https://www.bu.edu/bufellow/", "category": "fellowships", "title": "BU Office of Fellowships"},

    # Student life & activities
    {"url": "https://www.bu.edu/studentservices/", "category": "student_life", "title": "Student Services at BU"},
    {"url": "https://www.bu.edu/activities/", "category": "student_life", "title": "BU Student Activities"},
    {"url": "https://www.bu.edu/caso/", "category": "student_life", "title": "Center for Student Affairs & Opportunities"},
    {"url": "https://www.bu.edu/safety/", "category": "safety", "title": "BU Campus Safety & Security"},
    {"url": "https://www.bu.edu/transportation/", "category": "transportation", "title": "BU Transportation & Parking"},
]


def scrape_page(url: str) -> str:
    try:
        r = httpx.get(url, timeout=15, follow_redirects=True)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:8000]
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return ""


results = []
for page in BU_PAGES:
    print(f"Scraping: {page['url']}")
    content = scrape_page(page["url"])
    if content:
        results.append({**page, "content": content})
        print(f"  OK ({len(content)} chars)")

out_path = os.path.join(os.path.dirname(__file__), "../../data/bu_resources.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone. Scraped {len(results)}/{len(BU_PAGES)} pages → data/bu_resources.json")
