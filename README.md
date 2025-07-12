## your interview helper (chatbot)

### overview
This is a chatbot that helps you prepare for technical interviews using document-based and keyword-based questions. It leverages LLM (OpenAI) and Streamlit for an interactive experience.

### Requirments
- Python 3.12
- OpenAI API

### Installation

#### Common requirements
- Pull this repository
- Create `.env` file in the root directory and fill it with the following content.
  ```
  OPENAI_API_KEY={your-api-key}
  MODEL=gpt-4o-mini
  EMBEDDING=text-embedding-3-small
  ```

#### For local use
- Create and activate a virtual environment
  ```
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install the necessary Python packages listed in `requirements.txt`
  ```
  pip install -r requirements.txt
  ```
- Start Streamlit
  ```
  streamlit run main.py
  ```
  
#### For use with Docker
- Build Docker Image
  ```
  docker build -t interview-helper:latest .
  ```
- Run the Docker container
  ```
  docker run --rm -p 8501:8501 --env-file .env interview-helper:latest
  ```
You can use the Docker image via GitHub Packages at [packages](https://github.com/mardi2020/interview-helper/pkgs/container/interview-helper)
- Please pull the image and use it as follows
  ```
  docker pull ghcr.io/mardi2020/interview-helper:<tag>
  ```
  
Please access the app by visiting http://localhost:8501 in your browser.
