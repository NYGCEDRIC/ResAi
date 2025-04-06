from dotenv import load_dotenv
load_dotenv()
import base64
import streamlit as st
import os
import io
from PIL import Image
import pdf2image
import google.generativeai as genai
import time
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import re
import requests

# Load environment variables
load_dotenv()

# Configure the API
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Perplexity API setup
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
perplexity_headers = {
    "Authorization": f"Bearer {perplexity_api_key}",
    "Content-Type": "application/json"
}

# Function to get response from Gemini (using only gemini-1.5-flash)
def get_gemini_response(input_prompt, pdf_content, job_desc_input):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_prompt, pdf_content[0], job_desc_input])
        return response.text
    except Exception as e:
        error_msg = str(e)
        print(f"Error with gemini-1.5-flash: {error_msg}")
        st.error(f"Error generating response: {error_msg}")
        return None

# Function to process PDF
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read())
        first_page = images[0]
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        pdf_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        return pdf_parts, images
    else:
        raise FileNotFoundError("No file uploaded")

# Function to extract text from resume
def extract_text_from_pdf(images):
    text = ""
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        model = genai.GenerativeModel('gemini-1.5-flash')
        image_parts = [
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }
        ]
        ocr_prompt = "Extract all text from this image, preserve formatting as much as possible."
        response = model.generate_content([ocr_prompt, image_parts[0]])
        text += response.text + "\n"
    return text

# Function to parse the percentage match from the response
def parse_percentage(response_text):
    match = re.search(r'(\d+)%', response_text)
    if match:
        return int(match.group(1))
    return 0

# Function to generate suggestions for improvement
def generate_suggestions(pdf_content, job_desc):
    prompt = """
    You are a professional career coach with expertise in resume optimization.
    Based on the resume and job description provided, offer 5 specific, actionable suggestions to improve the resume.
    Format your response as a bulleted list. Be concise but specific.
    Focus on content, structure, keywords, and presentation improvements.
    """
    return get_gemini_response(prompt, pdf_content, job_desc)

# Function to highlight keywords in the resume text
def highlight_keywords(resume_text, job_desc):
    prompt = f"""
    Extract the top 15 most important keywords from this job description, and return them as a comma-separated list:
    
    {job_desc}
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    keywords = [kw.strip() for kw in response.text.split(',')]
    
    highlighted_text = resume_text
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(f"**{keyword}**", highlighted_text)
    
    return highlighted_text, keywords


# Function to search using Perplexity API with improved robustness and flexibility
def tavily_job_search(resume_text, job_desc_input, count=5):
    """
    Perform a job search using Tavily Search API
    
    Args:
        resume_text (str): Extracted text from the user's resume
        job_desc_input (str): Job description or target role
        count (int): Number of search results to retrieve
    
    Returns:
        str: Markdown-formatted job search results
    """
    # Retrieve Tavily API key from environment variables
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not tavily_api_key:
        return "❌ Error: Tavily API key is missing"
    
    # Use Gemini to extract key context for search
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    try:
        # Extract skills from resume
        skills_response = model.generate_content(f"""
        Extract the top 10 most relevant professional skills from this resume:
        {resume_text}
        Return as a comma-separated list of skills.
        """)
        resume_skills = skills_response.text.strip()
        
        # Extract job title and key requirements
        title_response = model.generate_content(f"""
        Extract the exact job title and 3-5 most critical requirements from this job description:
        {job_desc_input}
        
        Format your response as:
        Job Title: [Exact Job Title]
        Key Requirements: 
        1. [Requirement 1]
        2. [Requirement 2]
        3. [Requirement 3]
        """)
        job_context = title_response.text.strip()
        
        # Extract job title
        job_title = job_context.split('Job Title:')[1].split('\n')[0].strip()
        
        # Construct search query
        search_query = f'"{job_title}" jobs {" ".join(resume_skills.split(",")[:3])} hiring now'
        
        # Prepare Tavily API request
        tavily_payload = {
            "api_key": tavily_api_key,
            "query": search_query,
            "search_depth": "advanced",
            "include_domains": [
                "linkedin.com",
                "indeed.com", 
                "glassdoor.com", 
                "monster.com"
            ],
            "max_results": count,
            "include_raw_content": True
        }
        
        # Perform the search
        response = requests.post(
            "https://api.tavily.com/search",
            json=tavily_payload
        )
        
        # Check response
        if response.status_code != 200:
            return f"❌ Tavily Search API error: {response.status_code}\n{response.text}"
        
        # Parse search results
        search_results = response.json()
        
        # Generate markdown results
        markdown_results = "## 🔍 Personalized Job Search Results\n\n"
        
        # Check if results exist
        if not search_results.get('results', []):
            return "❌ No job results found"
        
        # Process each search result
        for idx, result in enumerate(search_results['results'], 1):
            # Extract job details
            title = result.get('title', 'Untitled Job')
            link = result.get('url', '#')
            snippet = result.get('raw_content', 'No description available')
            
            # Analyze job relevance
            relevance_prompt = f"""
            Analyze the relevance of this job to the candidate's profile:
            
            Candidate Skills: {resume_skills}
            Job Title: {title}
            Job Description: {snippet}
            
            Provide:
            1. Relevance Score (0-100%)
            2. Key Matching Skills
            3. Potential Fit Commentary
            """
            
            try:
                relevance_response = model.generate_content(relevance_prompt)
                relevance_analysis = relevance_response.text
            except Exception as e:
                relevance_analysis = f"Relevance analysis failed: {str(e)}"
            
            # Format markdown entry
            markdown_results += f"### {idx}. {title}\n\n"
            markdown_results += f"**Link:** [{link}]({link})\n\n"
            markdown_results += f"**Description:** {snippet}\n\n"
            markdown_results += f"**Relevance Analysis:**\n{relevance_analysis}\n\n"
            markdown_results += "---\n\n"
        
        return markdown_results
    
    except Exception as e:
        return f"❌ Comprehensive search error: {str(e)}"
# Streamlit App
st.set_page_config(page_title="ResAi", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 1rem;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .percentage-container {
        display: flex;
        align-items: center;
        margin: 20px 0;
    }
    .percentage {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .match-text {
        font-size: 1.2rem;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>ResAi</h1>", unsafe_allow_html=True)

# Create sidebar menu
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Resume Analysis", "Resume Optimizer", "Cover Letter Generator", "Interview Prep", "Market Position", "Skill Development", "Search"],
        icons=["file-earmark-text", "magic", "envelope", "chat-dots", "graph-up", "book", "search"],
        menu_icon="cast",
        default_index=0,
    )
    
    # Add contact info and credits
    st.markdown("---")
    st.markdown("### ResAi Team")
    st.markdown("Team: Cedric, " \
    " Ahmed," \
    " Aiden," \
    "Anthony")
    st.markdown("Contact: tandiaa@whitman.edu")

# Common inputs
job_desc_input = st.text_area("Job Description:", key="job_desc", height=200)
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=["pdf"])

# Initialize variables
pdf_content = None
resume_text = ""
images = []

# Check if file is uploaded
if uploaded_file is not None:
    try:
        with st.spinner("Processing your resume..."):
            pdf_content, images = input_pdf_setup(uploaded_file)
            resume_text = extract_text_from_pdf(images)
            st.success("Resume processed successfully!")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")

# Resume Analysis
if selected == "Resume Analysis" and pdf_content is not None and job_desc_input.strip() != "":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h2 class='sub-header'>Resume Analysis</h2>", unsafe_allow_html=True)
        analysis_button = st.button("Analyze Resume", type="primary")
        
        if analysis_button:
            with st.spinner("Analyzing your resume..."):
                input_prompt1 = """
                You are an experienced Technical Human Resource Manager with 15+ years of experience in talent acquisition.
                Your task is to review the provided resume against the job description.
                
                Please provide a detailed professional evaluation with these sections:
                1. OVERVIEW: A brief summary of the candidate's profile
                2. STRENGTHS: Key qualifications that align well with the role (be specific)
                3. GAPS: Areas where the candidate could improve or lacks required qualifications
                4. MATCH PERCENTAGE: Exact percentage of how well the resume matches the job requirements
                5. RECOMMENDATION: Whether to proceed with the candidate, and why
                
                Use a professional tone and provide actionable insights.
                Start with the match percentage on its own line, formatted as "XX%"
                """
                
                response = get_gemini_response(input_prompt1, pdf_content, job_desc_input)
                
                if response:
                    # Extract percentage
                    match_percentage = parse_percentage(response)
                    
                    # Display percentage with gauge
                    st.markdown("<div class='percentage-container'>", unsafe_allow_html=True)
                    st.markdown(f"<span class='percentage'>{match_percentage}%</span>", unsafe_allow_html=True)
                    st.markdown(f"<span class='match-text'>Match with Job Description</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Create a donut chart
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie([match_percentage, 100-match_percentage], 
                           colors=['#1E88E5', '#ECEFF1'], 
                           startangle=90, 
                           wedgeprops=dict(width=0.3))
                    ax.axis('equal')
                    st.pyplot(fig)
                    
                    # Display full analysis
                    st.markdown(f"<div class='highlight'>{response}</div>", unsafe_allow_html=True)

# Resume Optimizer
elif selected == "Resume Optimizer" and pdf_content is not None and job_desc_input.strip() != "":
    st.markdown("<h2 class='sub-header'>Resume Optimizer</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Resume Content")
        if resume_text:
            highlighted_text, keywords = highlight_keywords(resume_text, job_desc_input)
            st.markdown(highlighted_text)
    
    with col2:
        st.markdown("### Improvement Suggestions")
        optimize_button = st.button("Generate Optimization Suggestions", type="primary")
        
        if optimize_button:
            with st.spinner("Generating suggestions..."):
                suggestions = generate_suggestions(pdf_content, job_desc_input)
                if suggestions:
                    st.markdown(f"<div class='highlight'>{suggestions}</div>", unsafe_allow_html=True)

# Cover Letter Generator
elif selected == "Cover Letter Generator" and pdf_content is not None and job_desc_input.strip() != "":
    st.markdown("<h2 class='sub-header'>Cover Letter Generator</h2>", unsafe_allow_html=True)
    
    company_name = st.text_input("Company Name:")
    hiring_manager = st.text_input("Hiring Manager Name (leave blank if unknown):")
    customize_options = st.multiselect("Customize your cover letter focus:", 
                                      ["Technical Skills", "Leadership Experience", "Project Highlights", 
                                       "Cultural Fit", "Problem-Solving Abilities", "Industry Knowledge"])
    
    generate_button = st.button("Generate Cover Letter", type="primary")
    
    if generate_button:
        with st.spinner("Generating your personalized cover letter..."):
            prompt = f"""
            You are a professional resume writer with expertise in the tech industry.
            
            Create a personalized, compelling cover letter based on the resume and job description provided.
            
            Use these details:
            - Company: {company_name if company_name else "[Company Name]"}
            - Hiring Manager: {hiring_manager if hiring_manager else "Hiring Manager"}
            
            Focus areas: {", ".join(customize_options) if customize_options else "balanced approach"}
            
            The cover letter should:
            1. Be approximately 300-400 words
            2. Follow professional business letter format
            3. Have a compelling introduction, meaningful body paragraphs, and a call-to-action conclusion
            4. Highlight the candidate's most relevant skills and experiences
            5. Address how the candidate meets the specific job requirements
            6. Show enthusiasm for the role and company
            7. Avoid generic language and be tailored to this specific opportunity
            
            Do not use placeholder text - create a complete, ready-to-use cover letter.
            """
            
            cover_letter = get_gemini_response(prompt, pdf_content, job_desc_input)
            
            if cover_letter:
                # Display in a nice format
                st.markdown(f"<div class='highlight'>{cover_letter}</div>", unsafe_allow_html=True)
                
                # Add a download button
                st.download_button(
                    label="Download Cover Letter",
                    data=cover_letter,
                    file_name="cover_letter.txt",
                    mime="text/plain"
                )

# Interview Prep
elif selected == "Interview Prep" and pdf_content is not None and job_desc_input.strip() != "":
    st.markdown("<h2 class='sub-header'>Interview Preparation Guide</h2>", unsafe_allow_html=True)
    
    prep_button = st.button("Generate Interview Prep Guide", type="primary")
    
    if prep_button:
        with st.spinner("Creating your interview preparation guide..."):
            prompt = """
            You are an expert hiring manager. Based on the resume and job description provided, create:
            1. 5 technical questions likely to be asked in the interview
            2. 3 behavioral questions specific to this role
            3. 2 questions about gaps or potential weaknesses in the candidate's profile
            
            For each question, provide a sample answer strategy (not a complete answer).
            Format your response clearly with sections and numbered questions.
            """
            response = get_gemini_response(prompt, pdf_content, job_desc_input)
            
            if response:
                st.markdown(f"<div class='highlight'>{response}</div>", unsafe_allow_html=True)

# Market Position
elif selected == "Market Position" and pdf_content is not None and job_desc_input.strip() != "":
    st.markdown("<h2 class='sub-header'>Market Position Analysis</h2>", unsafe_allow_html=True)
    
    position_button = st.button("Analyze Market Position", type="primary")
    
    if position_button:
        with st.spinner("Analyzing your market position..."):
            prompt = """
            You are an experienced hiring manager. Create a profile of an ideal candidate for this job description.
            Then compare the provided resume against this ideal profile.
            
            Format your response in these sections:
            1. Ideal Candidate Profile: Key skills, experience, and qualifications
            2. Comparison: How the candidate meets or falls short of each key requirement
            3. Competitive Analysis: Where this candidate would rank against typical applicants (top 10%, average, etc.)
            """
            response = get_gemini_response(prompt, pdf_content, job_desc_input)
            
            if response:
                st.markdown(f"<div class='highlight'>{response}</div>", unsafe_allow_html=True)

# Skill Development
elif selected == "Skill Development" and pdf_content is not None and job_desc_input.strip() != "":
    st.markdown("<h2 class='sub-header'>Skill Development Plan</h2>", unsafe_allow_html=True)
    
    skill_button = st.button("Generate Skill Development Plan", type="primary")
    
    if skill_button:
        with st.spinner("Creating your personalized skill development plan..."):
            prompt = """
            You are a career development coach. Based on the resume and job description, create a 3-month skill development plan for the candidate.
            
            Include:
            1. Top 3-5 skills to develop based on gaps in the resume
            2. Specific resources to learn each skill (courses, certifications, projects)
            3. A timeline with weekly goals
            4. How to demonstrate these new skills on the resume
            
            Format your response in a clear, actionable plan.
            """
            response = get_gemini_response(prompt, pdf_content, job_desc_input)
            
            if response:
                st.markdown(f"<div class='highlight'>{response}</div>", unsafe_allow_html=True)

# Search
elif selected == "Search" and pdf_content is not None and job_desc_input.strip() != "":
    st.markdown("<h2 class='sub-header'>Job Search</h2>", unsafe_allow_html=True)
    
    # Number of results selector
    num_results = st.slider("Number of job results", 1, 10, 5)
    
    search_button = st.button("Find Matching Jobs", type="primary")
    
    if search_button:
        with st.spinner("Searching for personalized job opportunities..."):
            # Perform job search
            job_results = tavily_job_search(
                resume_text=resume_text, 
                job_desc_input=job_desc_input,
                count=num_results
            )
            
            # Display results
            st.markdown(job_results)
# Show homepage content
else:
    st.info("👈 Please upload your resume and enter a job description to get started.")
    
    # Display app features
    st.markdown("### Features of resai")
    features = [
        "🔍 **Resume Analysis**: Get detailed feedback on your resume's match with job requirements",
        "✨ **Resume Optimizer**: Get actionable suggestions to improve your resume",
        "📝 **Cover Letter Generator**: Create a tailored cover letter in seconds",
        "🎯 **Interview Prep**: Prepare with likely interview questions and answer strategies",
        "📈 **Market Position**: Compare your profile against an ideal candidate",
        "📚 **Skill Development**: Get a personalized skill development plan",
        "🔍 **Search**: Find relevant information to enhance your application"
    ]
    for feature in features:
        st.markdown(feature)

# Add model status checker
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Status Checker")
if st.sidebar.button("Check Model Status"):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello, please respond with 'OK' if you can receive this message.")
        if "OK" in response.text or "ok" in response.text.lower():
            st.sidebar.markdown("✅ **gemini-1.5-flash**: Available")
        else:
            st.sidebar.markdown("⚠️ **gemini-1.5-flash**: Unexpected response")
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
            st.sidebar.markdown("❌ **gemini-1.5-flash**: Rate limited")
        else:
            st.sidebar.markdown(f"❌ **gemini-1.5-flash**: Error: {error_msg[:50]}...")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and Google Gemini 1.5 Flash")
st.markdown("© ResAi - Whitman Hackathon Project")