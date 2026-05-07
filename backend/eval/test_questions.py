# 20 BU-specific test questions for RAG evaluation
# Each has a question and the expected source category

TEST_QUESTIONS = [
    # International students
    {
        "question": "How do I apply for OPT as an F-1 student?",
        "ground_truth": "F-1 students at BU apply for OPT through the International Students and Scholars Office (ISSO). You must apply up to 90 days before your program end date and no later than 60 days after. The ISSO provides guidance and required forms.",
        "expected_category": "international"
    },
    {
        "question": "What is CPT and how is it different from OPT?",
        "ground_truth": "CPT (Curricular Practical Training) is work authorization for internships that are part of your curriculum, while OPT (Optional Practical Training) is work authorization for after graduation. Both are for F-1 students and handled through the ISSO at BU.",
        "expected_category": "international"
    },
    {
        "question": "Where is the ISSO office at BU?",
        "ground_truth": "The International Students and Scholars Office (ISSO) is located at 888 Commonwealth Avenue, Boston. They handle visa advising, OPT/CPT applications, and support for all international students.",
        "expected_category": "international"
    },
    {
        "question": "What documents do I need for my F-1 visa travel signature?",
        "ground_truth": "To get a travel signature on your I-20, you need to contact the ISSO at BU. They will verify your enrollment and provide the signature required for re-entry into the US.",
        "expected_category": "international"
    },

    # Career
    {
        "question": "How do I get resume help at BU?",
        "ground_truth": "BU Career Development offers resume reviews through drop-in hours (Mon-Fri 10am-4pm, no appointment needed) and scheduled appointments. They are located in the GSU.",
        "expected_category": "career"
    },
    {
        "question": "Does BU have a career fair?",
        "ground_truth": "Yes, BU holds career fairs each semester including the Spring Career Fair at the GSU Ballroom. The BU Career Development office organizes these events.",
        "expected_category": "career"
    },
    {
        "question": "Where is the BU Career Development office?",
        "ground_truth": "BU Career Development is located in the George Sherman Union (GSU). They offer career advising, resume reviews, interview prep, and job search resources.",
        "expected_category": "career"
    },

    # Academic advising
    {
        "question": "How do I drop a class at BU?",
        "ground_truth": "To drop a class at BU, you use the Student Link portal before the drop deadline. After the deadline you need dean approval. The University Registrar handles official enrollment changes.",
        "expected_category": "advising"
    },
    {
        "question": "Where can I get academic advising in CAS?",
        "ground_truth": "CAS Academic Advising is available in the College of Arts and Sciences. Advisors help with course selection, major requirements, and academic planning.",
        "expected_category": "advising"
    },
    {
        "question": "What is the add/drop deadline at BU?",
        "ground_truth": "The add/drop deadline at BU is typically in the second week of the semester. Check the University Registrar's academic calendar for exact dates each semester.",
        "expected_category": "registrar"
    },

    # Tutoring
    {
        "question": "Where can I get free tutoring at BU?",
        "ground_truth": "The Educational Resource Center (ERC) at BU offers free drop-in and appointment tutoring for most undergraduate subjects, especially STEM. Located in the GSU.",
        "expected_category": "tutoring"
    },
    {
        "question": "What subjects does the Educational Resource Center cover?",
        "ground_truth": "The ERC (Educational Resource Center) at BU covers most undergraduate subjects with a focus on STEM courses. They offer both drop-in tutoring and scheduled appointments.",
        "expected_category": "tutoring"
    },

    # Health
    {
        "question": "How do I make an appointment at Student Health Services?",
        "ground_truth": "BU Student Health Services appointments can be made online through the Patient Connect portal or by calling the SHS office. They provide primary care, mental health services, and urgent care.",
        "expected_category": "health"
    },
    {
        "question": "Does BU offer mental health counseling?",
        "ground_truth": "Yes, BU offers mental health counseling through Student Health Services and the Student Wellness & Prevention Services office. Services include individual counseling, group therapy, and crisis support.",
        "expected_category": "wellness"
    },

    # Library
    {
        "question": "What are Mugar Library hours?",
        "ground_truth": "Mugar Memorial Library is open Mon-Thu 8am-midnight, Fri 8am-10pm, Sat 10am-8pm, Sun 10am-midnight. Extended hours are available during finals.",
        "expected_category": "library"
    },
    {
        "question": "Can I book a study room at the BU library?",
        "ground_truth": "Yes, BU Libraries offer group study room reservations. Rooms can be booked through the BU Libraries website. Mugar Memorial Library has group study rooms on the 2nd floor.",
        "expected_category": "library"
    },

    # Financial aid
    {
        "question": "How do I apply for financial aid at BU?",
        "ground_truth": "To apply for financial aid at BU, submit the FAFSA and the CSS Profile by BU's priority deadline. The BU Financial Aid office reviews applications and awards grants, loans, and work-study.",
        "expected_category": "financial_aid"
    },

    # Housing
    {
        "question": "How does BU housing lottery work?",
        "ground_truth": "BU Housing assigns rooms through a lottery system for returning students. Students receive a lottery number and select rooms during their assigned time slot through the housing portal.",
        "expected_category": "housing"
    },

    # Accessibility
    {
        "question": "How do I register for disability accommodations at BU?",
        "ground_truth": "To register for disability accommodations at BU, contact Disability & Access Services and submit documentation from a qualified professional. They will work with you and your professors to arrange appropriate accommodations.",
        "expected_category": "accessibility"
    },

    # Study abroad
    {
        "question": "How do I apply for study abroad at BU?",
        "ground_truth": "To apply for study abroad at BU, contact the BU Study Abroad office. They offer programs in over 30 countries. You apply through their office and work with your academic advisor to ensure credits transfer.",
        "expected_category": "study_abroad"
    },
]
