from flask import Flask, request, render_template_string, flash, redirect, url_for
import os
import re
import pdfplumber
import docx
from werkzeug.utils import secure_filename
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'super secret key'

# Load the NLP model and setup stop words
nlp = spacy.load("en_core_web_lg")
stop_words = set(stopwords.words('english')) | STOP_WORDS

# HTML template for the upload page and results display
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Resume & Job Description Checker</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f4f4f9;
            color: #333;
        }
        h2, h3 {
            color: #333;
        }
        form {
            margin-bottom: 20px;
        }
        input[type="file"], input[type="submit"] {
            margin-top: 5px;
            margin-right: 10px;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            text-decoration: none;
            margin: 4px 2px;
            cursor: pointer;
        }
        pre {
            background-color: #eee;
            padding: 10px;
            overflow: auto;
            border: 1px solid #ccc;
            max-height: 200px;  /* Limit height */
        }
        details summary {
            font-weight: bold;
            cursor: pointer;
            margin-bottom: 5px;
        }
    </style>
</head>
<body>
    <h2>Upload Resume and Job Description</h2>
    <form method="post" action="/" enctype="multipart/form-data">
        <label>Resume:</label>
        <input type="file" name="resume"><br>
        <label>Job Description:</label>
        <input type="file" name="job_description"><br><br>
        <input type="submit" value="Upload">
    </form>

    {% if score %}
        <h3>Match Score: {{ score }}%</h3>
        <h3>Category: {{ category }}</h3>
        <h3>Extracted Emails: {{ emails }}</h3>
        <details>
            <summary>Cleaned Resume:</summary>
            <pre>{{ cleaned_resume }}</pre>
        </details>
        <details>
            <summary>Cleaned Job Description:</summary>
            <pre>{{ cleaned_jd }}</pre>
        </details>
        <h3>Years of Experience Difference: {{ years_diff }}</h3>
        <h3>Resume Years of Experience: {{ resume_years }}</h3>
        <h3>Job Description Years of Experience: {{ jd_years }}</h3>
        {% if years_diff > 0 %}
            <p>The job description requires {{ years_diff }} more year(s) of experience than your resume shows.</p>
        {% elif years_diff < 0 %}
            <p>Your resume exceeds the job requirements by {{ -years_diff }} year(s).</p>
        {% else %}
            <p>Your experience perfectly matches the job requirements.</p>
        {% endif %}
        {% if total_experience_months %}
            <h3>Total Experience: {{ total_experience_months }} months</h3>
        {% endif %}
    {% endif %}
</body>
</html>
'''

# ------------------ Utility Functions ------------------

def remove_education_section(text):
    """
    Removes the text between 'EDUCATION' and the next section header (e.g., 'EXPERIENCE', 'PROJECTS').
    """
    cleaned_text = re.sub(r'(?s)EDUCATION.*?(?=\n[A-Z]+\n)', '', text, flags=re.MULTILINE)
    return cleaned_text

def extract_experience(text):
    """
    Extracts all unique years from the text—including year ranges like "2021–2022"—and calculates the total experience.
    The experience is computed as the difference between the highest and lowest year found multiplied by 12 (months).
    """
    # Find year ranges (e.g., "2021–2022" or "2021-2022")
    year_ranges = re.findall(r'\b(20\d{2})[–-](20\d{2})\b', text)
    # Find individual year mentions
    individual_years = re.findall(r'\b(20\d{2})\b', text)

    all_years = set()
    for start, end in year_ranges:
        for year in range(int(start), int(end) + 1):
            all_years.add(year)
    all_years.update(map(int, individual_years))

    if all_years:
        sorted_years = sorted(all_years)
        total_experience = sorted_years[-1] - sorted_years[0]
        return total_experience * 12  # Return experience in months
    return 0

def extract_text(file_path):
    """
    Extracts text from a file based on its extension.
    Supported file types are PDF, TXT, and DOCX.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    try:
        if file_extension == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or '' for page in pdf.pages)
            return text
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
        else:
            raise ValueError("Unsupported file format: " + file_extension)
    except Exception as e:
        return f"Error extracting text from {file_path}: {str(e)}"

def extract_unique_words_jd(text):
    """
    Extracts unique words from the job description text, preserving the order in which they first appear.
    The unique words are returned as a newline-separated string.
    """
    words = re.findall(r'\b\w+\b', text)
    unique_words = []
    seen = set()
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    return "\n".join(unique_words)

def clean_text(text):
    """
    Cleans extracted text by converting to lowercase, filtering out stop words, punctuation,
    spaces, and specific parts of speech, then removing duplicate tokens.
    """
    doc = nlp(text.lower())
    filtered_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.pos_ not in ['ADP', 'DET', 'PRON']:
            filtered_tokens.append(token.lemma_ if token.pos_ == 'VERB' else token.text)
    cleaned_tokens = remove_duplicates(filtered_tokens)
    return " ".join(cleaned_tokens)

def remove_duplicates(tokens):
    """
    Removes duplicate tokens while preserving their original order.
    """
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    return unique_tokens

def process_text(text):
    """
    Tokenizes text and removes stopwords.
    """
    doc = nlp(text)
    return [token.text for token in doc if not token.is_stop]

def extract_emails(text):
    """
    Extracts email addresses using a regular expression.
    """
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.findall(email_pattern, text)

def calculate_similarity(resume_tokens, jd_tokens, resume_years, jd_years):
    """
    Calculates the similarity score based on text content and years of experience.
    The score is a weighted average: 70% textual similarity and 30% experience match.
    """
    resume_doc = nlp(" ".join(resume_tokens))
    jd_doc = nlp(" ".join(jd_tokens))
    textual_similarity = resume_doc.similarity(jd_doc) if resume_doc and jd_doc else 0

    # Experience similarity is the ratio of the resume's years to the required years
    experience_similarity = (resume_years / jd_years) if jd_years > 0 else 1.0

    weighted_similarity = (textual_similarity * 0.7) + (experience_similarity * 0.3)
    return weighted_similarity * 100  # Convert to percentage

def classify_similarity(score):
    """
    Classifies the similarity score into categories.
    """
    if score < 20:
        return "Poor"
    elif score < 50:
        return "Fair"
    elif score < 70:
        return "Good"
    elif score < 90:
        return "Very Good"
    else:
        return "Excellent"

def process_files(resume_path, jd_path):
    """
    Processes the resume and job description files and returns:
      - Match percentage and category
      - Extracted emails
      - Cleaned resume and job description text
      - Years of experience difference and totals
    """
    resume_text = extract_text(resume_path)
    jd_text = extract_text(jd_path)

    if 'Error' in resume_text or 'Error' in jd_text:
        return None, "Error extracting text", [], "", "", 0, 0, 0, 0

    # Remove education section from resume for experience extraction
    resume_text_for_experience = remove_education_section(resume_text)
    total_experience_months = extract_experience(resume_text_for_experience)

    # Clean texts for display and processing
    cleaned_resume = clean_text(resume_text)
    cleaned_jd = extract_unique_words_jd(jd_text)
    emails = extract_emails(resume_text)

    # Tokenize texts for similarity calculation
    resume_tokens = process_text(cleaned_resume)
    jd_tokens = process_text(jd_text)

    resume_years = total_experience_months // 12
    jd_years = 5  # Assumed required years from the job description

    match_percentage = calculate_similarity(resume_tokens, jd_tokens, resume_years, jd_years)
    category = classify_similarity(match_percentage)
    years_diff = jd_years - resume_years

    return (match_percentage, category, emails, cleaned_resume, cleaned_jd,
            years_diff, resume_years, jd_years, total_experience_months)

# ------------------ Flask Routes ------------------

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        resume_file = request.files.get('resume')
        jd_file = request.files.get('job_description')

        if resume_file and jd_file:
            resume_filename = secure_filename(resume_file.filename)
            jd_filename = secure_filename(jd_file.filename)
            resume_file.save(resume_filename)
            jd_file.save(jd_filename)

            result = process_files(resume_filename, jd_filename)

            # Clean up uploaded files
            os.remove(resume_filename)
            os.remove(jd_filename)

            if result[0] is None:
                flash('Failed to process files.')
                return redirect(url_for('upload_file'))

            (score, category, emails, cleaned_resume, cleaned_jd,
             years_diff, resume_years, jd_years, total_experience_months) = result

            return render_template_string(HTML_TEMPLATE,
                                          score=f"{score:.2f}",
                                          category=category,
                                          emails=", ".join(emails),
                                          cleaned_resume=cleaned_resume,
                                          cleaned_jd=cleaned_jd,
                                          years_diff=years_diff,
                                          resume_years=resume_years,
                                          jd_years=jd_years,
                                          total_experience_months=total_experience_months)
    return render_template_string(HTML_TEMPLATE, score=None)

if __name__ == '__main__':
    app.run(debug=True)
