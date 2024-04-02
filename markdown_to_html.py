import markdown

if __name__ == "__main__":

    # Load markdown file
    with open("content/readme_content/auto-readme-app_readme.md", 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)

    # Save HTML content
    with open("content/readme_content/auto-readme-app_readme.html", 'w', encoding='utf-8') as html_file:
        html_file.write(html_content)
        print("HTML content saved successfully.")