from fpdf import FPDF

def create_sample_resume():
    pdf = FPDF()
    pdf.add_page()
    
    # Set font
    pdf.set_font("Arial", "B", 16)
    
    # Name
    pdf.cell(0, 10, "John Smith", ln=True, align='C')
    
    # Contact Information
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Email: john.smith@email.com", ln=True, align='C')
    pdf.cell(0, 10, "Phone: (555) 123-4567", ln=True, align='C')
    
    # Summary
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Professional Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, "Experienced software developer with expertise in Python, JavaScript, and web development.")
    
    # Skills
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Technical Skills", ln=True)
    pdf.set_font("Arial", "", 12)
    skills = [
        "Python", "JavaScript", "React", "Node.js", "SQL",
        "Machine Learning", "Git", "Docker", "AWS", "MongoDB"
    ]
    skill_text = ", ".join(skills)
    pdf.multi_cell(0, 10, skill_text)
    
    # Experience
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Work Experience", ln=True)
    
    # Job 1
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Senior Software Developer - Tech Corp", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "2020 - Present", ln=True)
    pdf.multi_cell(0, 10, "- Developed and maintained web applications using React and Node.js\n- Implemented machine learning models for data analysis\n- Led a team of 5 developers")
    
    # Save the PDF
    pdf.output("sample_resume.pdf")

if __name__ == "__main__":
    create_sample_resume()
