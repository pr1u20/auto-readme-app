import requests
import base64
import fnmatch
import os

def get_headers():
    """
    Returns headers containing the GitHub API authentication token.
    """
    token = os.environ['GITHUB_API_KEY']  # Replace with your actual PAT
    headers = {
        'Authorization': f'token {token}'
    }
    return headers

def fetch_gitignore_patterns(owner, repo):
    """
    Fetches the .gitignore file and extracts the patterns.
    
    Parameters:
    - owner (str): Owner of the repository
    - repo (str): Repository name
    
    Returns:
    - List of patterns from the .gitignore file
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/.gitignore"
    headers = get_headers()
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        gitignore_content = base64.b64decode(response.json()['content']).decode('utf-8')
        patterns = gitignore_content.splitlines()
        return [pattern for pattern in patterns if pattern and not pattern.startswith('#')]
    return []

def should_ignore(path, patterns):
    """
    Determines if a given path matches any of the .gitignore patterns, is a common directory to ignore,
    or ends with .pyc.
    
    Parameters:
    - path (str): The file path to check
    - patterns (list): The list of .gitignore patterns
    
    Returns:
    - True if the path matches any pattern, is in a common directory to ignore, or ends with .pyc; False otherwise.
    """
    common_ignores = ['__pycache__', '.git', 'node_modules', 'LICENSE', '.gitignore', '__init__.py']  # Add more if needed
    if any(ignore in path.split('/') for ignore in common_ignores):
        return True
    
    if path.endswith('.pyc'):
        return True

    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True

    return False

def get_files_contents(owner, repo, patterns, path=""):
    """
    Fetch and decode the content of all files in a GitHub repository, ignoring .gitignore patterns.
    
    Parameters:
    - owner (str): Owner of the repository
    - repo (str): Repository name
    - patterns (list): Patterns from .gitignore to ignore
    - path (str, optional): Directory path to start from, default is root
    
    Returns:
    - Dictionary of {file_path: content} for each file in the repository
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = get_headers()
    response = requests.get(api_url, headers=headers)
    items = response.json()
    
    contents = {}
    for item in items:
        if should_ignore(item['path'], patterns):
            continue  # Skip files matched by .gitignore patterns
        
        if item['type'] == 'file':
            file_content_response = requests.get(item['download_url'])
            file_content = file_content_response.text
            contents[item['path']] = file_content
        elif item['type'] == 'dir':
            contents.update(get_files_contents(owner, repo, patterns, item['path']))
    
    return contents

def generate_user_content(owner, repo, save_as_md=False):
    """
    Load all the files from a github repository and generate a str with all the content to be fed into ChatGPT.
    Optionally, save this content as a Markdown file.

    Parameters:
    - owner (str): Owner of the repository
    - repo (str): Repository name
    - save_as_md (bool): If True, save the content as a Markdown file

    Returns:
    - str with all the content of the files in the repository
    """
    
    # Fetch .gitignore patterns
    patterns = fetch_gitignore_patterns(owner, repo)

    # Fetch repository contents respecting .gitignore
    contents = get_files_contents(owner, repo, patterns)

    user_content = ""
    for file_path, content in contents.items():
        user_content += f"### Content of {file_path}:\n\n```python\n{content}\n```\n\n"

    if save_as_md:
        # Define the Markdown file name
        md_filename = f"{repo}_content.md"

        path = os.path.join("content", "user_content", md_filename)

        with open(path, 'w', encoding='utf-8') as md_file:
            md_file.write(user_content)
            print(f"Content saved to {md_filename}")

    return user_content

if __name__ == "__main__":
    owner = "pr1u20"  # Replace with the repository owner's username
    repo = "pyaocs"  # Replace with the repository name
    user_content = generate_user_content(owner, repo, save_as_md=True)

