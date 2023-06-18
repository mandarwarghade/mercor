# Import necessary libraries
import requests
from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import langchain
#from markupsafe import escape

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def analyze():
    github_url = request.form.get('github_url')
    most_complex_repo=get_most_complex_repository(github_url)

    if most_complex_repo is None:
        # Handle the case when no repositories are found
        error_message = "No repositories found. Please enter a valid GitHub URL."
        return render_template('error.html', error=error_message)

    # Generate GPT analysis for the most complex repository
    gpt_analysis = generate_gpt_analysis(most_complex_repo)

    return render_template('result.html', repository=most_complex_repo, analysis=gpt_analysis)


def get_most_complex_repository(username):
    # Fetch user's repositories
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return

    repositories = response.json()

    # Determine the most complex repository
    most_complex_repo = None
    complexity_score = 0

    for repo in repositories:
        # Fetch repository contents
        contents_url = repo['contents_url'].replace("{+path}", "")
        contents_response = requests.get(contents_url)
        if contents_response.status_code != 200:
            print(f"Error: {contents_response.status_code}")
            continue

        contents = contents_response.json()

        # Count the number of Jupyter notebook files
        jupyter_file_count = sum(1 for item in contents if item['name'].endswith('.ipynb'))

        # Update the most complex repository if necessary
        if jupyter_file_count > complexity_score:
            complexity_score = jupyter_file_count
            most_complex_repo = repo['name']

    return most_complex_repo

def generate_gpt_analysis(repository):
    # Load GPT model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Prepare input prompt for GPT analysis
    prompt = construct_prompt(repository)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate GPT analysis
    outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)
    gpt_analysis = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return gpt_analysis

def construct_prompt(repository):
    # Construct a prompt that incorporates repository information
    prompt = f"Analyzing repository:\n"
    prompt += f"Name: {repository}\n"
    # Add any other relevant information about the repository
    return prompt

if __name__ == '__main__':
    app.debug = True
    app.run()
